import os
import re
import json
import math
import asyncio
import datetime
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from .models import (
    TriageRequest,
    TriageResponse,
    TriageSummary,
    PackageTriageResult,
    AgentStep,
    AvailabilityStatus,
    SalesRecommendationTier,
    DeliverableType,
    DeliverableMatchStatus,
    RecommendationTrafficLight,
    SIMDPortingComplexity,
    TestDependency,
    ArchSupportScan,
    DockerfileImageFinding,
    ConfigImageFinding,
    _PARTIAL_EFFORT_MULTIPLIERS,
)
from .normalizer import UniversalNormalizer
from .prober import MultiSourceProber
from .build_analyzer import BuildAnalyzer, scale_arch_effort, get_simd_multipliers
from .repo_scanner import RepoScanner
from .gemini_helper import generate_gemini_content
from .url_extractor import URLExtractor

logger = logging.getLogger("agent")

BUFFER_PERSON_DAYS = 5          # one person-week of contingency added to the itemised total
EFFORT_CEILING_FACTOR = 1.30    # multiplier applied before rounding up to the next Fibonacci number

_SIMD_COMPLEXITY_BY_NAME = {
    "direct": SIMDPortingComplexity.DIRECT,
    "simde_compatible": SIMDPortingComplexity.SIMDE_COMPATIBLE,
    "partial_rewrite": SIMDPortingComplexity.PARTIAL_REWRITE,
    "full_redesign": SIMDPortingComplexity.FULL_REDESIGN,
}

FIBONACCI_NUMBERS = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]


def next_fibonacci_effort(effort: float) -> int:
    """Returns the single Fibonacci value that is strictly greater than the input effort.
    For example: 5 -> 8 (as requested), 26 -> 34, 34 -> 55, 0 -> 0.
    """
    if effort <= 0:
        return 0
    for fib in FIBONACCI_NUMBERS:
        if fib > effort:
            return fib
    return FIBONACCI_NUMBERS[-1]


def round_half_up(value: float) -> int:
    """Round .5 up (Python's round() is banker's rounding: 4.5 -> 4)."""
    return int(math.floor(value + 0.5))


KNOWN_ECOSYSTEM_DETAILS: Dict[str, Dict[str, str]] = {
    "rocksdb": {
        "git_repo_url": "https://github.com/facebook/rocksdb",
        "doc_url": "https://rocksdb.org",
        "ecosystem": "native_c",
        "version": "8.6",
    },
    "redis": {
        "git_repo_url": "https://github.com/redis/redis",
        "doc_url": "https://redis.io",
        "ecosystem": "container",
        "version": "7.2",
    },
    "nginx": {
        "git_repo_url": "https://github.com/nginx/nginx",
        "doc_url": "https://nginx.org",
        "ecosystem": "container",
        "version": "1.24",
    },
    "fastapi": {
        "git_repo_url": "https://github.com/tiangolo/fastapi",
        "doc_url": "https://fastapi.tiangolo.com",
        "ecosystem": "pypi",
        "version": "0.110.0",
    },
    "pytorch": {
        "git_repo_url": "https://github.com/pytorch/pytorch",
        "doc_url": "https://pytorch.org",
        "ecosystem": "pypi",
        "version": "2.1.0",
    },
    "torch": {
        "git_repo_url": "https://github.com/pytorch/pytorch",
        "doc_url": "https://pytorch.org",
        "ecosystem": "pypi",
        "version": "2.1.0",
    },
    "simdjson": {
        "git_repo_url": "https://github.com/simdjson/simdjson",
        "doc_url": "https://simdjson.org",
        "ecosystem": "native_c",
        "version": "3.6",
    },
    "snappy": {
        "git_repo_url": "https://github.com/google/snappy",
        "doc_url": "https://github.com/google/snappy",
        "ecosystem": "native_c",
        "version": "1.1.9",
    },
    "jemalloc": {
        "git_repo_url": "https://github.com/jemalloc/jemalloc",
        "doc_url": "http://jemalloc.net",
        "ecosystem": "native_c",
        "version": "5.3.0",
    },
}


class TriageAgent:
    """Agentic AI Orchestrator that triages workloads, probes repositories, scopes transitive dependencies,
    analyzes SIMD instructions & complexity, checks test dependencies, scans Dockerfiles & configs, and synthesizes reports.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_client = None

        if self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"):
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key.strip())
                logger.info("Gemini LLM Client successfully initialized with provided API key.")
            except Exception as e:
                logger.warning(f"Could not initialize Google GenAI Client: {e}")
        else:
            logger.info("Running in autonomous simulation mode (no live GEMINI_API_KEY provided).")

        self.prober = MultiSourceProber(gemini_client=self.gemini_client)

    # ------------------------------------------------------------------
    # Pipeline helpers (pure / synchronous unless noted)
    # ------------------------------------------------------------------

    class _Steps:
        """Builds SSE step events with an auto-incrementing step id."""

        def __init__(self) -> None:
            self.counter = 1

        def emit(self, agent_name: str, action_type: str, target: str, thought: str,
                 detail: Optional[str] = None) -> Dict[str, Any]:
            event = {
                "type": "step",
                "step": AgentStep(
                    step_id=self.counter,
                    agent_name=agent_name,
                    action_type=action_type,
                    target=target,
                    thought=thought,
                    detail=detail,
                ).model_dump(),
            }
            self.counter += 1
            return event

    @staticmethod
    def _component_effort(p: PackageTriageResult) -> float:
        """Sum of the itemised effort components for one package (person-days, unrounded)."""
        return (
            p.base_build_effort_pd
            + p.transitive_deps_effort_pd
            + p.arch_complexity_effort_pd
            + p.test_effort_pd
        )

    @staticmethod
    def _detect_urls(request: TriageRequest, raw_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns (git_url, doc_url) from the request, falling back to URLs embedded in raw text."""
        git_url: Optional[str] = request.git_repo_url
        doc_url: Optional[str] = request.doc_url
        if raw_text:
            github_matches = re.findall(r"https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", raw_text)
            if not git_url and github_matches:
                git_url = github_matches[0]
            all_urls = re.findall(r"https?://[a-zA-Z0-9_.-]+\.[a-zA-Z]{2,}(?:/[^\s)\"'>]*)?", raw_text)
            non_git = [u for u in all_urls if "github.com" not in u]
            if not doc_url and non_git:
                doc_url = non_git[0]
        return git_url, doc_url

    @staticmethod
    def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (item["name"].lower(), item.get("version", "latest"))
            if key not in seen and item["name"].strip():
                seen.add(key)
                unique.append(item)
        return unique

    @staticmethod
    def _infer_primary(
        unique_items: List[Dict[str, Any]],
        git_url: Optional[str],
        doc_url: Optional[str],
        name: Optional[str],
        version: Optional[str],
        ecosystem: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Infers the primary package and enriches provenance from the known-ecosystem table.

        Returns (name, version, ecosystem, git_url, doc_url).
        """
        if not name:
            if git_url:
                gh_m = re.search(r"github\.com/[^/]+/([^/\s?#]+)", git_url)
                if gh_m:
                    name = gh_m.group(1).replace(".git", "").lower()
            elif doc_url:
                doc_m = re.search(r"https?://(?:www\.)?([^/\.]+)", doc_url)
                if doc_m:
                    name = doc_m.group(1).lower()
            elif unique_items:
                name = unique_items[0]["name"]
                version = unique_items[0].get("version", "latest")
                ecosystem = unique_items[0].get("ecosystem", "container")

        if name:
            info = KNOWN_ECOSYSTEM_DETAILS.get(name.lower().strip())
            if info:
                git_url = git_url or info.get("git_repo_url")
                doc_url = doc_url or info.get("doc_url")
                if not version or version == "latest":
                    version = info.get("version", "latest")
                if not ecosystem:
                    ecosystem = info.get("ecosystem", "native_c")
        return name, version, ecosystem, git_url, doc_url

    @staticmethod
    def _clone_and_scan(url: str) -> Tuple[Optional[ArchSupportScan], List[DockerfileImageFinding], List[ConfigImageFinding]]:
        """Blocking: shallow-clone a repo and scan it. Run via asyncio.to_thread."""
        temp_repo = RepoScanner.shallow_clone_repo(url)
        if not temp_repo:
            return None, [], []
        try:
            return (
                RepoScanner.scan_arch_support(temp_repo),
                RepoScanner.scan_dockerfiles(temp_repo),
                RepoScanner.scan_config_images(temp_repo),
            )
        finally:
            RepoScanner.cleanup_repo(temp_repo)

    async def _probe_image_findings(
        self,
        docker_findings: List[DockerfileImageFinding],
        config_findings: List[ConfigImageFinding],
        effective_os: str,
        deliverable_type: DeliverableType,
    ) -> None:
        """Probes Dockerfile base images and compose/CI config images in place."""
        for df in docker_findings:
            probe = await self.prober.probe_component(df.base_image, df.tag, "container", effective_os, deliverable_type)
            df.status = probe["status"]
            if df.status in (AvailabilityStatus.UNPORTED_BUILD_REQUIRED, AvailabilityStatus.BLOCKER):
                df.effort_pd = 0.5
                df.notes = f"Base image {df.base_image}:{df.tag} not verified for ppc64le; container rebuild required."
            else:
                df.notes = f"Verified multi-arch ppc64le container ({probe.get('tier')})"

        for cf in config_findings:
            ref = cf.image_ref
            img_name, img_tag = ref.rsplit(":", 1) if ":" in ref else (ref, "latest")
            probe = await self.prober.probe_component(img_name, img_tag, "container", effective_os, deliverable_type)
            cf.status = probe["status"]
            if cf.status in (AvailabilityStatus.UNPORTED_BUILD_REQUIRED, AvailabilityStatus.BLOCKER):
                cf.effort_pd = 0.5
                cf.notes = f"Config image {ref} not verified for ppc64le; container rebuild required."
            else:
                cf.notes = f"Verified multi-arch ppc64le container ({probe.get('tier')})"

    async def _refine_arch_with_llm(self, pkg: PackageTriageResult) -> None:
        """Asks Gemini to size SIMD volume/complexity for a package and rescales its arch effort."""
        if not self.gemini_client or not pkg.arch_sensitivity:
            return
        arch = pkg.arch_sensitivity
        prompt = (
            f"You are an IBM Power (ppc64le) porting specialist. Analyze this unported library '{pkg.package_name}'.\n"
            f"Architecture issue: SIMD={arch.has_simd_avx}, 64KB Page={arch.has_64k_page_risk}, Asm={arch.has_inline_asm}.\n"
            f"Code snippet:\n{arch.code_snippet}\n"
            "Evaluate:\n"
            "1. simd_instruction_count: estimated count of x86 SIMD/AVX instructions (e.g. 0, 45, 150, 320, 600)\n"
            "2. simd_porting_complexity: one of 'direct' (macro drop-in), 'simde_compatible' (SIMDe header mapping), 'partial_rewrite' (mixed intrinsics requiring VSX rewrite), or 'full_redesign' (deep AVX-512 dependencies)\n"
            "3. remediation_strategy: concise 2-sentence porting remediation vector recommending SIMDe, Power VSX intrinsics, or 64KB page allocator configuration.\n"
            "Return ONLY JSON:\n"
            "{\n"
            '  "simd_instruction_count": int,\n'
            '  "simd_porting_complexity": "direct" | "simde_compatible" | "partial_rewrite" | "full_redesign",\n'
            '  "remediation_strategy": "string"\n'
            "}\n"
            "JSON:"
        )
        try:
            resp_text = await asyncio.to_thread(generate_gemini_content, self.gemini_client, prompt, json_mode=True)
            if not resp_text:
                return
            parsed = json.loads(resp_text.replace("```json", "").replace("```", "").strip())
            if "simd_instruction_count" in parsed:
                arch.simd_instruction_count = max(0, int(parsed["simd_instruction_count"]))
            complexity = _SIMD_COMPLEXITY_BY_NAME.get(str(parsed.get("simd_porting_complexity", "")).lower())
            if complexity:
                arch.simd_porting_complexity = complexity
            if "remediation_strategy" in parsed:
                arch.remediation_strategy = parsed["remediation_strategy"]

            base_arch = (
                pkg.base_arch_complexity_effort_pd
                if pkg.base_arch_complexity_effort_pd > 0
                else (arch.base_engineering_effort_pd if arch.base_engineering_effort_pd > 0 else 2.0)
            )
            count_m, comp_m = get_simd_multipliers(arch.simd_instruction_count, arch.simd_porting_complexity)
            arch.base_engineering_effort_pd = base_arch
            arch.simd_instruction_multiplier = count_m
            arch.simd_complexity_multiplier = comp_m
            pkg.arch_complexity_effort_pd = scale_arch_effort(
                base_arch, arch.simd_instruction_count, arch.simd_porting_complexity
            )
            pkg.total_effort_pd = round_half_up(self._component_effort(pkg))
        except Exception as llm_err:
            logger.warning(f"Gemini code analysis call skipped: {llm_err}")

    @classmethod
    def _apply_effort_adjustments(
        cls,
        packages: List[PackageTriageResult],
        arch_scan: Optional[ArchSupportScan],
        docker_findings: List[DockerfileImageFinding],
        config_findings: List[ConfigImageFinding],
    ) -> None:
        """Applies deliverable-match multipliers, repo-scan factor, and container-image effort in place."""
        for pkg in packages:
            dm = pkg.deliverable_match
            if dm in (DeliverableMatchStatus.PARTIAL_DIFFERENT_TYPE, DeliverableMatchStatus.PARTIAL_DIFFERENT_VERSION):
                mult = _PARTIAL_EFFORT_MULTIPLIERS[dm]
                if pkg.total_effort_pd > 0:
                    pkg.total_effort_pd = max(1, round_half_up(pkg.total_effort_pd * mult))
                elif pkg.status in (AvailabilityStatus.NATIVE_AVAILABLE, AvailabilityStatus.PLATFORM_AGNOSTIC):
                    # Partial match on a previously zero-effort package: add integration overhead
                    pkg.total_effort_pd = 1 if dm == DeliverableMatchStatus.PARTIAL_DIFFERENT_VERSION else 2

        if arch_scan and arch_scan.effort_adjustment_factor != 1.0:
            for p in packages:
                if p.arch_complexity_effort_pd > 0:
                    p.arch_complexity_effort_pd = round(p.arch_complexity_effort_pd * arch_scan.effort_adjustment_factor, 2)
                total = cls._component_effort(p)
                if total > 0:
                    p.total_effort_pd = max(1, round_half_up(total))

        container_extra_pd = sum(f.effort_pd for f in docker_findings) + sum(f.effort_pd for f in config_findings)
        if container_extra_pd > 0 and packages:
            packages[0].total_effort_pd += round_half_up(container_extra_pd)

    @staticmethod
    def _summarize(packages: List[PackageTriageResult], deliverable_type: DeliverableType) -> TriageSummary:
        """Aggregates readiness, Fibonacci-sized effort and the qualification tier."""
        total_count = len(packages)

        def count(status: AvailabilityStatus) -> int:
            return sum(1 for p in packages if p.status == status)

        native_count = count(AvailabilityStatus.NATIVE_AVAILABLE)
        agnostic_count = count(AvailabilityStatus.PLATFORM_AGNOSTIC)
        substitute_count = count(AvailabilityStatus.SUBSTITUTE_AVAILABLE)
        unported_count = count(AvailabilityStatus.UNPORTED_BUILD_REQUIRED)
        blocker_count = count(AvailabilityStatus.BLOCKER)
        partial_count = sum(
            1 for p in packages
            if p.deliverable_match in (
                DeliverableMatchStatus.PARTIAL_DIFFERENT_TYPE,
                DeliverableMatchStatus.PARTIAL_DIFFERENT_VERSION,
            )
        )
        unported_transitive_count = sum(
            1 for p in packages for d in p.build_dependencies
            if d.status == AvailabilityStatus.UNPORTED_BUILD_REQUIRED
        )
        unported_test_count = sum(
            1 for p in packages for td in p.test_dependencies
            if td.status != AvailabilityStatus.NATIVE_AVAILABLE
        )

        ready_units = (
            native_count + agnostic_count
            + (substitute_count * 0.9)
            + (unported_count * 0.3)
            - (partial_count * 0.25)   # partial matches reduce readiness
        )
        readiness_pct = round((max(ready_units, 0) / max(total_count, 1)) * 100.0, 1)

        raw_total_pd = sum(p.total_effort_pd for p in packages)
        if raw_total_pd == 0 and unported_count == 0:
            fib_pd = 0
        elif raw_total_pd == 0:
            fib_pd = next_fibonacci_effort(BUFFER_PERSON_DAYS)
        else:
            fib_pd = next_fibonacci_effort(round_half_up((raw_total_pd + BUFFER_PERSON_DAYS) * EFFORT_CEILING_FACTOR))

        if blocker_count > 0:
            rec = SalesRecommendationTier.NOT_POSSIBLE_AS_IS
            rec_reason = f"Workload contains {blocker_count} proprietary x86 component(s) requiring architectural substitution (e.g. OpenBLAS / ESSL alternatives recommended)."
        elif readiness_pct >= 90 and raw_total_pd == 0:
            rec = SalesRecommendationTier.MINIMAL_EFFORT
            rec_reason = f"Turnkey readiness ({readiness_pct}%). All components run out-of-the-box on IBM Power (0 Person-Days)."
        elif partial_count > 0 and readiness_pct >= 70:
            rec = SalesRecommendationTier.MODERATE_EFFORT
            rec_reason = (
                f"Moderate readiness ({readiness_pct}%) with {partial_count} partial deliverable match(es) "
                f"for '{deliverable_type.value}' deliverable type. Effort: {fib_pd} Person-Days (Fibonacci)."
            )
        elif fib_pd <= 8:
            rec = SalesRecommendationTier.MINOR_EFFORT
            rec_reason = f"High readiness ({readiness_pct}%). Light build verification estimated at {fib_pd} Person-Days (Fibonacci max estimate)."
        elif fib_pd <= 34:
            rec = SalesRecommendationTier.MODERATE_EFFORT
            rec_reason = f"Moderate porting scope ({readiness_pct}% ready). Transitive dependencies and build verification estimated at {fib_pd} Person-Days (Fibonacci max estimate)."
        else:
            rec = SalesRecommendationTier.SIGNIFICANT_EFFORT
            rec_reason = f"Extensive porting investment ({readiness_pct}% ready). Multi-tier transitive dependencies or vector kernels estimated at {fib_pd} Person-Days (Fibonacci max estimate)."

        return TriageSummary(
            total_packages=total_count,
            native_count=native_count,
            agnostic_count=agnostic_count,
            substitute_count=substitute_count,
            unported_count=unported_count,
            blocker_count=blocker_count,
            readiness_score_pct=readiness_pct,
            recommendation=rec,
            recommendation_reason=rec_reason,
            fibonacci_effort_pd=fib_pd,
            min_total_person_days=fib_pd,
            max_total_person_days=fib_pd,
            unported_transitive_deps_count=unported_transitive_count,
            test_deps_unresolved_count=unported_test_count,
            partial_support_count=partial_count,
        )

    @staticmethod
    def _attach_links(
        packages: List[PackageTriageResult],
        primary_pkg_name: Optional[str],
        git_url: Optional[str],
        doc_url: Optional[str],
    ) -> None:
        for p in packages:
            name = p.package_name.lower()
            if name in KNOWN_ECOSYSTEM_DETAILS:
                k = KNOWN_ECOSYSTEM_DETAILS[name]
                p.git_repo_url = k.get("git_repo_url")
                p.doc_url = k.get("doc_url")
                if not p.evidence_url:
                    p.evidence_url = p.git_repo_url or p.doc_url
            elif name == (primary_pkg_name or "").lower():
                p.git_repo_url = git_url
                p.doc_url = doc_url
                if not p.evidence_url:
                    p.evidence_url = git_url or doc_url

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    async def run_triage_stream(self, request: TriageRequest) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs the triage pipeline yielding live agent steps and the final comprehensive response."""
        if request.gemini_api_key and not self.gemini_client:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=request.gemini_api_key.strip())
                self.prober.set_gemini_client(self.gemini_client)
            except Exception as e:
                logger.warning(f"Could not initialize requested Gemini client: {e}")

        steps = self._Steps()
        effective_os = request.effective_os()
        deliverable_type: DeliverableType = request.deliverable_type

        # Step 1: Ingestion & normalization
        yield steps.emit(
            "IngestionAgent", "plan", "Input Manifests",
            "Ingesting provided input and normalizing it into a list of components (name, version, ecosystem).",
            f"Target: {request.target_environment.display} (ppc64le). "
            f"Deliverable: {deliverable_type.value.replace('_', ' ').title()}. "
            f"Depth: {request.triage_depth.value}.",
        )
        await asyncio.sleep(0.3)

        raw_text = request.raw_manifest or ""
        parsed_items = UniversalNormalizer.normalize(raw_text, request.manifest_type or "auto")
        detected_git_url, detected_doc_url = self._detect_urls(request, raw_text)
        primary_pkg_name: Optional[str] = None
        primary_version: Optional[str] = None
        primary_ecosystem: Optional[str] = None

        # Step 1.1: URL-based dependency inference
        candidate_url = detected_git_url or detected_doc_url
        if not candidate_url and raw_text and URLExtractor.is_url(raw_text):
            candidate_url = raw_text.strip()
            if "github.com" in candidate_url:
                detected_git_url = candidate_url
            else:
                detected_doc_url = candidate_url

        if candidate_url and URLExtractor.is_url(candidate_url):
            yield steps.emit(
                "URLScraperAgent", "url_inference", candidate_url,
                f"Crawling package website or repository ({candidate_url}) to extract dependencies and manifest declarations...",
            )
            await asyncio.sleep(0.3)

            url_pkgs, url_title, _url_note = await URLExtractor.extract_from_url(candidate_url, gemini_client=self.gemini_client)
            if url_pkgs:
                parsed_items.extend(url_pkgs)
                primary_pkg_name = url_pkgs[0]["name"]
                primary_version = url_pkgs[0].get("version", "latest")
                primary_ecosystem = url_pkgs[0].get("ecosystem", "native_c")
                yield steps.emit(
                    "LLMDependencyInferenceAgent", "url_inference", url_title,
                    f"Inferred {len(url_pkgs)} library requirements and version specifications from {candidate_url}.",
                    ", ".join(f"{p['name']} ({p.get('version', 'latest')})" for p in url_pkgs[:6]) + ("..." if len(url_pkgs) > 6 else ""),
                )
                await asyncio.sleep(0.3)

        for p in request.packages_list or []:
            parsed_items.append({
                "name": p.get("name", ""),
                "version": p.get("version", "latest"),
                "ecosystem": p.get("ecosystem", "container"),
            })

        unique_items = self._dedupe_items(parsed_items)
        primary_pkg_name, primary_version, primary_ecosystem, detected_git_url, detected_doc_url = self._infer_primary(
            unique_items, detected_git_url, detected_doc_url, primary_pkg_name, primary_version, primary_ecosystem
        )

        yield steps.emit(
            "IngestionAgent", "plan", "Dependency Normalizer",
            f"Successfully extracted and normalized {len(unique_items)} unique components across container, language, and system layers."
            + (f" Primary package: {primary_pkg_name}." if primary_pkg_name else ""),
            ", ".join(i["name"] for i in unique_items[:6]) + ("..." if len(unique_items) > 6 else ""),
        )
        await asyncio.sleep(0.3)

        # Step 1.5: Repository architecture & container scan
        arch_scan: Optional[ArchSupportScan] = None
        docker_findings: List[DockerfileImageFinding] = []
        config_findings: List[ConfigImageFinding] = []

        scan_repo_url = detected_git_url or request.git_repo_url
        if scan_repo_url:
            yield steps.emit(
                "RepoScanAgent", "scan", scan_repo_url,
                "Cloning repository shallowly to inspect architecture markers, Dockerfiles, and CI configurations...",
            )
            await asyncio.sleep(0.2)
            arch_scan, docker_findings, config_findings = await asyncio.to_thread(self._clone_and_scan, scan_repo_url)

        if not arch_scan:
            target_pkg_name = request.project_name or (unique_items[0]["name"] if unique_items else "workload")
            arch_scan = RepoScanner.llm_infer_arch_support(target_pkg_name, gemini_client=self.gemini_client)

        # Step 2: Multi-source availability probing
        triaged_packages: List[PackageTriageResult] = []
        unported_candidates: List[PackageTriageResult] = []

        for item in unique_items:
            name = item["name"]
            version = item.get("version", "latest")
            eco = item.get("ecosystem", "container")

            yield steps.emit(
                "AvailabilityProber", "lookup", f"{name} ({eco})",
                f"Consulting live registries (Quay.io/Docker Hub/PyPI), {effective_os} repos, and Gemini LLM for {name} ({version}) on ppc64le...",
            )
            await asyncio.sleep(0.15)

            probe_res = await self.prober.probe_component(name, version, eco, effective_os, deliverable_type)
            pkg_result = PackageTriageResult(
                package_name=name,
                requested_version=version,
                ecosystem=eco,
                status=probe_res["status"],
                tier_description=probe_res["tier"],
                evidence_source=probe_res.get("evidence"),
                evidence_url=probe_res.get("url"),
                substitute_package=probe_res.get("substitute"),
                version_warning=probe_res.get("version_warning"),
                deliverable_match=probe_res.get("deliverable_match", DeliverableMatchStatus.SUPPORTED),
                deliverable_detail=probe_res.get("deliverable_detail"),
            )

            if pkg_result.version_warning:
                yield steps.emit(
                    "AvailabilityProber", "lookup", f"{name} Version Notice",
                    f"⚠️ {pkg_result.version_warning}",
                    f"Requested: {version} | Verified Available on Power: {probe_res.get('available_version', 'N/A')}",
                )
                await asyncio.sleep(0.1)

            if pkg_result.deliverable_detail and pkg_result.deliverable_match != DeliverableMatchStatus.SUPPORTED:
                yield steps.emit(
                    "DeliverableClassifier", "lookup", f"{name} Deliverable Match",
                    f"Deliverable match [{deliverable_type.value}]: {pkg_result.deliverable_match.value.replace('_', ' ').title()}",
                    pkg_result.deliverable_detail,
                )
                await asyncio.sleep(0.1)
            elif probe_res.get("evidence"):
                yield steps.emit(
                    "AvailabilityProber", "lookup", f"{name} Verified",
                    f"Qualification Result: {pkg_result.status.value.upper()} • {pkg_result.tier_description}",
                    probe_res.get("evidence"),
                )
                await asyncio.sleep(0.1)

            if pkg_result.status == AvailabilityStatus.UNPORTED_BUILD_REQUIRED:
                unported_candidates.append(pkg_result)
            triaged_packages.append(pkg_result)

        await self._probe_image_findings(docker_findings, config_findings, effective_os, deliverable_type)

        if triaged_packages:
            triaged_packages[0].arch_support_scan = arch_scan
            triaged_packages[0].dockerfile_image_findings = docker_findings
            triaged_packages[0].config_image_findings = config_findings

        # Step 3: Deep source build & transitive dependency scoping
        if unported_candidates and request.triage_depth == "deep":
            yield steps.emit(
                "BuildScopingAgent", "deep_scan", "Unported Components",
                f"Detected {len(unported_candidates)} unported components. Initiating deep source build audit, inspecting build-time dependencies, SIMD instructions, and test harnesses...",
            )
            await asyncio.sleep(0.3)

            for pkg in unported_candidates:
                yield steps.emit(
                    "BuildScopingAgent", "deep_scan", pkg.package_name,
                    f"Parsing build configuration for {pkg.package_name}. Auditing BuildRequires, SIMD instructions, and test dependencies...",
                )
                await asyncio.sleep(0.25)

                BuildAnalyzer.analyze_package_build(pkg, effective_os)

                unported_sub_deps = [d for d in pkg.build_dependencies if d.status == AvailabilityStatus.UNPORTED_BUILD_REQUIRED]
                if unported_sub_deps:
                    yield steps.emit(
                        "BuildScopingAgent", "deep_scan", f"{pkg.package_name} Build-Tree",
                        f"⚠️ Uncovered transitive dependency iceberg in {pkg.package_name}: {len(unported_sub_deps)} build requirement(s) also unported on ppc64le!",
                        ", ".join(f"{d.name} (+{d.porting_effort_pd} PD)" for d in unported_sub_deps),
                    )
                    await asyncio.sleep(0.2)

                if pkg.test_dependencies:
                    yield steps.emit(
                        "BuildScopingAgent", "deep_scan", f"{pkg.package_name} Test Validation",
                        f"Scoped {len(pkg.test_dependencies)} test dependency/harness requirement(s). Test effort: {pkg.test_effort_pd} PD.",
                        ", ".join(f"{t.name} ({t.status.value})" for t in pkg.test_dependencies),
                    )
                    await asyncio.sleep(0.15)

                arch = pkg.arch_sensitivity
                if arch and (arch.has_simd_avx or arch.has_64k_page_risk or arch.has_inline_asm):
                    await self._refine_arch_with_llm(pkg)
                    yield steps.emit(
                        "ArchitectureHeuristicAgent", "llm_reasoning", f"{pkg.package_name} Code Audit",
                        f"Detected architecture friction in {pkg.package_name}: {arch.simd_instruction_count} SIMD instruction(s), "
                        f"complexity tier '{arch.simd_porting_complexity.value}'. Scaled engineering effort: {pkg.arch_complexity_effort_pd} PD.",
                        arch.remediation_strategy,
                    )
                    await asyncio.sleep(0.2)

            # Size the remaining components (blockers, substitutes, native) so their effort
            # is not silently dropped from the total when other packages are unported.
            for pkg in triaged_packages:
                if pkg.status != AvailabilityStatus.UNPORTED_BUILD_REQUIRED:
                    BuildAnalyzer.analyze_package_build(pkg, effective_os)
        else:
            for pkg in triaged_packages:
                BuildAnalyzer.analyze_package_build(pkg, effective_os)

        # Step 4: Aggregate metrics & scoring
        self._apply_effort_adjustments(triaged_packages, arch_scan, docker_findings, config_findings)
        summary = self._summarize(triaged_packages, deliverable_type)

        # Step 5: Final executive synthesis
        yield steps.emit(
            "ExecutiveSynthesizer", "synthesis", "Executive Deliverables",
            f"Synthesizing pre-sales qualification brief and engineering backlog. Recommendation: {summary.recommendation.value}.",
        )
        await asyncio.sleep(0.3)

        self._attach_links(triaged_packages, primary_pkg_name, detected_git_url, detected_doc_url)

        brief_md = await asyncio.to_thread(
            self._generate_executive_brief,
            request, summary, triaged_packages, arch_scan, docker_findings, config_findings,
            primary_package_name=primary_pkg_name,
            git_repo_url=detected_git_url,
            doc_url=detected_doc_url,
            package_version=primary_version,
            package_ecosystem=primary_ecosystem,
        )
        csv_data = self._generate_csv(triaged_packages)
        detected_source_url = detected_git_url or detected_doc_url or (raw_text if URLExtractor.is_url(raw_text) else None)

        response = TriageResponse(
            project_name=request.project_name or (f"{primary_pkg_name} Migration" if primary_pkg_name else "Customer Migration Triage"),
            target_os=effective_os.upper(),
            target_platform=request.effective_platform().upper(),
            target_environment=request.target_environment.display,
            deliverable_type=deliverable_type.value,
            summary=summary,
            packages=triaged_packages,
            agent_steps=[],
            executive_brief_markdown=brief_md,
            export_csv_data=csv_data,
            primary_package_name=primary_pkg_name,
            git_repo_url=detected_git_url,
            doc_url=detected_doc_url,
            source_url=detected_source_url,
            package_ecosystem=primary_ecosystem,
            package_version=primary_version,
        )

        yield {"type": "result", "data": response.model_dump()}

    def _generate_executive_brief(
        self,
        request: TriageRequest,
        summary: TriageSummary,
        packages: List[PackageTriageResult],
        arch_scan: Optional[ArchSupportScan] = None,
        docker_findings: Optional[List[DockerfileImageFinding]] = None,
        config_findings: Optional[List[ConfigImageFinding]] = None,
        primary_package_name: Optional[str] = None,
        git_repo_url: Optional[str] = None,
        doc_url: Optional[str] = None,
        package_version: Optional[str] = None,
        package_ecosystem: Optional[str] = None,
    ) -> str:
        rec_val = summary.recommendation.value
        traffic_emoji = "🟢" if "Minimal" in rec_val else (
            "🔵" if "Minor" in rec_val else (
                "🟡" if "Moderate" in rec_val else (
                    "🟠" if "Significant" in rec_val else "🟣"
                )
            )
        )
        
        has_custom_project = request.project_name and request.project_name not in ["Customer Migration Triage", "Migration_Triage", "Workload Migration", "Report"]
        if has_custom_project and primary_package_name and primary_package_name.lower() not in request.project_name.lower():
            display_title = f"{request.project_name} ({primary_package_name})"
        elif has_custom_project:
            display_title = request.project_name
        elif primary_package_name:
            display_title = f"Porting Qualification: {primary_package_name}"
        else:
            display_title = "Customer Migration Triage"

        md = f"""# Executive Porting Feasibility Assessment: {display_title}

**Target Architecture:** IBM Power (`ppc64le`)  
**Target Operating System:** {request.target_os.value.upper()} on {request.target_platform.value.upper()}  
**Primary Target Component:** `{primary_package_name or 'Multi-Component Stack'}` (Version: `{package_version or 'latest'}`)  
**GitHub Repository:** [{git_repo_url or 'Not specified'}]({git_repo_url or '#'})  
**Documentation Link:** [{doc_url or 'Not specified'}]({doc_url or '#'})  
**Assessment Date:** {datetime.date.today().isoformat()}
**Triage Basis:** Registry probing, curated knowledge base, and dependency/SIMD heuristics (see evidence per component)

---

## 1. Executive Summary & Qualification Recommendation

| Metric | Assessment Result |
| :--- | :--- |
| **Porting Readiness Score** | **{summary.readiness_score_pct}%** |
| **Sales Recommendation Tier** | **{traffic_emoji} {summary.recommendation.value}** |
| **Estimated Porting Sizing** | **{summary.fibonacci_effort_pd} Person-Days (Fibonacci Sized)** |
| **Total Components Analyzed** | **{summary.total_packages}** ({summary.native_count} Native, {summary.agnostic_count} Script/Bytecode, {summary.substitute_count} Substitute, {summary.unported_count} Unported) |
| **Transitive Build Dependencies Scoped** | **{summary.unported_transitive_deps_count} unported build-time requirements uncovered** |
| **Test Dependencies Unresolved** | **{summary.test_deps_unresolved_count} unported test suites/harnesses** |

### Strategic Recommendation:
> **{summary.recommendation_reason}**

---

## 2. Key Technical Findings & Sizing Breakdown

> **Effort Sizing Methodology:** Sizing estimates utilize a single Fibonacci-calibrated ceiling (e.g. 5→8, 26→34, 34→55) covering environment provisioning, source compilation, transitive dependencies, SIMD/VSX adaptation, and test validation on IBM Power architecture. All estimates are rounded to integer person-days.

"""
        # Section 2.1: Version Warnings (Sub-Task 1)
        version_warn_pkgs = [p for p in packages if p.version_warning]
        if version_warn_pkgs:
            md += "### ⚠️ Version-Aware Porting Warnings\n"
            for p in version_warn_pkgs:
                md += f"- **{p.package_name}** (Requested: `{p.requested_version}`): {p.version_warning}\n"
            md += "\n"

        # Section 2.2: Package Porting Deep Dives
        for p in packages:
            if p.status == AvailabilityStatus.UNPORTED_BUILD_REQUIRED:
                md += f"### ⚠️ {p.package_name} ({p.ecosystem}) — Estimated Effort: {p.total_effort_pd} PD\n"
                md += f"- **Build System:** {p.build_system or 'Standard'}\n"
                md += f"- **Base Compilation Effort:** {p.base_build_effort_pd} PD\n"
                
                # Build dependencies
                if p.build_dependencies:
                    md += f"- **Build-Time Dependencies Scoped ({len(p.build_dependencies)} total):**\n"
                    for d in p.build_dependencies:
                        icon = "✅" if d.status == AvailabilityStatus.NATIVE_AVAILABLE else "❌"
                        md += f"  - {icon} `{d.name}`: {d.status.value} ({d.porting_effort_pd} PD). {d.notes or ''}\n"

                # Sub-Tasks 2 & 3: SIMD scaling breakdown
                if p.arch_sensitivity and (p.arch_sensitivity.has_simd_avx or p.arch_sensitivity.has_64k_page_risk or p.arch_sensitivity.has_inline_asm):
                    md += f"- **Engineering Adaptation Effort (+{p.arch_complexity_effort_pd} PD):**\n"
                    if p.arch_sensitivity.has_simd_avx:
                        md += (
                            f"  - **SIMD/AVX Analysis:** {p.arch_sensitivity.simd_instruction_count} instruction(s) detected • "
                            f"Complexity Tier: `{p.arch_sensitivity.simd_porting_complexity.value.upper()}` • {p.arch_sensitivity.simd_details}\n"
                        )
                    if p.arch_sensitivity.has_64k_page_risk:
                        md += f"  - **64KB Page Size:** {p.arch_sensitivity.page_risk_details}\n"
                    if p.arch_sensitivity.has_inline_asm:
                        md += f"  - **Inline Assembly:** {p.arch_sensitivity.asm_details}\n"
                    md += f"  - **Remediation Strategy:** {p.arch_sensitivity.remediation_strategy}\n"

                # Sub-Task 4: Test dependencies
                if p.test_dependencies:
                    md += f"- **Testing & Verification Sizing (+{p.test_effort_pd} PD):**\n"
                    for td in p.test_dependencies:
                        icon = "✅" if td.status == AvailabilityStatus.NATIVE_AVAILABLE else "❌"
                        md += f"  - {icon} `{td.name}` ({td.test_dep_type}): {td.status.value} ({td.porting_effort_pd} PD). {td.notes or ''}\n"

                md += "\n"
            elif p.status == AvailabilityStatus.SUBSTITUTE_AVAILABLE:
                md += f"### 🔄 {p.package_name} — Substitute Available (0.5 PD)\n"
                md += f"- **Alternative / Recommended Package:** `{p.substitute_package}`\n"
                md += f"- **Source:** {p.evidence_source}\n\n"

        # Section 2.3: Repository Architecture Posture & CI Matrix (Sub-Task 6)
        if arch_scan and (arch_scan.ppc64le_markers_found or arch_scan.x86_only_markers_found or arch_scan.ci_matrix_has_power):
            md += "### 🏛️ Repository Architecture Support Posture\n"
            md += f"- **Scan Source:** `{arch_scan.scan_source}`\n"
            md += f"- **Upstream CI Matrix Has Power (ppc64le):** {'Yes ✅' if arch_scan.ci_matrix_has_power else 'No ❌'}\n"
            md += f"- **Effort Adjustment Factor:** {arch_scan.effort_adjustment_factor}x\n"
            if arch_scan.ppc64le_markers_found:
                md += f"- **Power/VSX Code Markers ({len(arch_scan.ppc64le_markers_found)}):**\n"
                for m in arch_scan.ppc64le_markers_found[:5]:
                    md += f"  - `{m}`\n"
            if arch_scan.x86_only_markers_found:
                md += f"- **x86-Only Code Markers ({len(arch_scan.x86_only_markers_found)}):**\n"
                for m in arch_scan.x86_only_markers_found[:5]:
                    md += f"  - `{m}`\n"
            md += "\n"

        # Section 2.4: Dockerfile & Config Container Audit (Sub-Tasks 5 & 7)
        if docker_findings:
            md += f"### 🐳 Dockerfile Base Image Audit ({len(docker_findings)} discovered)\n"
            for df in docker_findings:
                icon = "✅" if df.status == AvailabilityStatus.NATIVE_AVAILABLE else "⚠️"
                md += f"- {icon} `{df.dockerfile_path}`: `FROM {df.base_image}:{df.tag}` — Status: {df.status.value} ({df.effort_pd} PD). {df.notes or ''}\n"
            md += "\n"

        if config_findings:
            md += f"### ⚙️ Compose & CI Configuration Image Audit ({len(config_findings)} discovered)\n"
            for cf in config_findings:
                icon = "✅" if cf.status == AvailabilityStatus.NATIVE_AVAILABLE else "⚠️"
                md += f"- {icon} `{cf.config_file_path}` ({cf.config_file_type}): image `{cf.image_ref}` — Status: {cf.status.value} ({cf.effort_pd} PD). {cf.notes or ''}\n"
            md += "\n"

        # Strategic Sales Proposal via Gemini AI
        llm_sales_narrative = ""
        if self.gemini_client:
            try:
                prompt = (
                    f"You are an executive IBM Power pre-sales architect. Write a 2-paragraph strategic sales rationale for moving this customer workload ({request.project_name}) to IBM Power (ppc64le) on {request.target_os.value.upper()}.\n"
                    f"Readiness Score: {summary.readiness_score_pct}%, Recommendation: {summary.recommendation.value}, Sizing: {summary.min_total_person_days}-{summary.max_total_person_days} Person-Days.\n"
                    f"Key unported/scoped components: {[p.package_name for p in packages if p.status == AvailabilityStatus.UNPORTED_BUILD_REQUIRED]}.\n"
                    "Highlight IBM Power's specific hardware strengths (e.g. high memory bandwidth, SMT8 core threading, enterprise reliability, and Open-CE optimization). Keep it professional, crisp, and compelling."
                )
                resp_text = generate_gemini_content(self.gemini_client, prompt)
                if resp_text:
                    llm_sales_narrative = f"\n### 🤖 Gemini AI Strategic Sales Proposal:\n{resp_text}\n\n"
            except Exception as e:
                logger.warning(f"Gemini executive sales pitch generation skipped: {e}")

        if llm_sales_narrative:
            md += llm_sales_narrative

        md += """---

## 3. Next Steps for Technical Sales
1. Present this qualification memo to client architecture team to establish feasibility.
2. Hand over scoped transitive build dependency specs to the IBM Power Porting Lab for unported libraries.
3. Utilize IBM Open-CE channels for accelerated AI/data libraries.
"""
        return md

    def _generate_csv(self, packages: List[PackageTriageResult]) -> str:
        def q(value) -> str:
            return '"' + str(value if value is not None else "").replace('"', '""') + '"'

        lines = ["Package Name,Version,Ecosystem,Status,Tier,Effort (PD),Build System,Transitive Deps,Evidence Source"]
        for p in packages:
            num_deps = len(p.build_dependencies)
            build_sys = p.build_system or "N/A"
            evidence = p.evidence_source or ""
            lines.append(
                ",".join([
                    q(p.package_name), q(p.requested_version), q(p.ecosystem), q(p.status.value),
                    q(p.tier_description), str(p.total_effort_pd), q(build_sys), str(num_deps), q(evidence),
                ])
            )
        return "\n".join(lines)
