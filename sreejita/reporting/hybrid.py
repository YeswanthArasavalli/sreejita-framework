from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT ENGINE — UNIVERSAL (FINAL, LOCKED)
# Sreejita Framework v3.6
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid Report Engine (Authoritative)

    GUARANTEES:
    - Executive-safe Markdown
    - Deterministic ordering
    - No intelligence computation
    - Graceful handling of UNKNOWN / AMBIGUOUS domains
    - No empty or duplicated sections
    """

    name = "hybrid"

    # -------------------------------------------------
    # BUILD MARKDOWN REPORT
    # -------------------------------------------------
    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:

        if not isinstance(domain_results, dict) or not domain_results:
            raise RuntimeError("HybridReport requires non-empty domain_results")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"
        report_path = output_dir / f"Sreejita_Executive_Report_{run_id}.md"

        domains = self._sort_domains(list(domain_results.keys()))
        primary_domain = domains[0]
        primary_payload = domain_results.get(primary_domain) or {}

        with open(report_path, "w", encoding="utf-8") as f:
            self._write_header(f, run_id, metadata)

            # =================================================
            # GLOBAL EXECUTIVE SUMMARY (SAFE)
            # =================================================
            executive = primary_payload.get("executive")
            if isinstance(executive, dict):
                self._write_global_executive(f, executive)
                self._write_sub_domain_executives(f, executive)

            # =================================================
            # DOMAIN DEEP DIVES
            # =================================================
            for domain in domains:
                payload = domain_results.get(domain)
                if isinstance(payload, dict):
                    self._write_domain_section(f, domain, payload)

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # GLOBAL EXECUTIVE
    # -------------------------------------------------
    def _write_global_executive(self, f, executive: Dict[str, Any]):
        brief = executive.get("executive_brief")
        board = executive.get("board_readiness") or {}
        trend = executive.get("board_readiness_trend") or {}

        if isinstance(brief, str) and brief.strip():
            f.write("## Executive Summary\n\n")
            f.write(f"{brief.strip()}\n\n")

        if board:
            f.write("### Board Readiness\n")
            f.write(f"- **Score:** {board.get('score','—')} / 100\n")
            f.write(f"- **Status:** {board.get('band','—')}\n")

        if trend:
            f.write(
                f"- **Trend:** {trend.get('trend','→')} "
                f"(Prev: {trend.get('previous_score','—')}, "
                f"Current: {trend.get('current_score','—')})\n"
            )

        f.write("\n---\n\n")

    # -------------------------------------------------
    # SUB-DOMAIN EXECUTIVE
    # -------------------------------------------------
    def _write_sub_domain_executives(self, f, executive: Dict[str, Any]):
        exec_by_sub = executive.get("executive_by_sub_domain")

        if not isinstance(exec_by_sub, dict) or not exec_by_sub:
            return

        f.write("## Executive Summary by Operating Area\n\n")

        for sub, payload in exec_by_sub.items():
            if not isinstance(payload, dict):
                continue

            brief = payload.get("executive_brief")
            board = payload.get("board_readiness") or {}

            if not brief and not board:
                continue  # 🔒 avoid empty sections

            f.write(f"### {sub.replace('_',' ').title()}\n\n")

            if isinstance(brief, str) and brief.strip():
                f.write(f"{brief.strip()}\n\n")

            if board:
                f.write(
                    f"- **Board Readiness Score:** {board.get('score','—')} / 100  \n"
                    f"- **Status:** {board.get('band','—')}\n\n"
                )

        f.write("---\n\n")

    # -------------------------------------------------
    # DOMAIN SECTION
    # -------------------------------------------------
    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write(f"## Domain Deep Dive — {domain.replace('_',' ').title()}\n\n")

        # ---------------- KPIs ----------------
        raw_kpis = result.get("kpis") or {}
        kpis = {
            k: v for k, v in raw_kpis.items()
            if isinstance(k, str)
            and not k.startswith("_")
            and k not in {"sub_domains", "primary_sub_domain"}
        }

        if kpis:
            f.write("### Key Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            for i, (k, v) in enumerate(kpis.items()):
                if i >= 9:
                    break
                f.write(
                    f"| {k.replace('_',' ').title()} | {self._format_value(k, v)} |\n"
                )
            f.write("\n")

        # ---------------- VISUALS ----------------
        visuals = [
            v for v in (result.get("visuals") or [])
            if isinstance(v, dict) and v.get("path")
        ][:6]

        if visuals:
            f.write("### Visual Evidence\n")
            for vis in visuals:
                caption = vis.get("caption", "Visual evidence")
                path = vis.get("path")

                try:
                    conf = int(float(vis.get("confidence", 0)) * 100)
                except Exception:
                    conf = 0

                f.write(f"![{caption}]({path})\n")
                f.write(f"> {caption} (Confidence: {conf}%)\n\n")

        # ---------------- INSIGHTS ----------------
        raw_insights = result.get("insights") or []
        insights: List[Dict[str, Any]] = []

        if isinstance(raw_insights, list):
            insights = raw_insights
        elif isinstance(raw_insights, dict):
            insights = (
                raw_insights.get("risks", []) +
                raw_insights.get("warnings", []) +
                raw_insights.get("strengths", [])
            )

        insights = [i for i in insights if isinstance(i, dict)][:5]

        if insights:
            f.write("### Key Insights\n")
            for ins in insights:
                f.write(
                    f"- **{ins.get('level','INFO')}** — "
                    f"{ins.get('title','')}: {ins.get('so_what','')}\n"
                )
            f.write("\n")

        # ---------------- RECOMMENDATIONS ----------------
        recs = [
            r for r in (result.get("recommendations") or [])
            if isinstance(r, dict)
        ][:5]

        if recs:
            f.write("### Recommendations\n")
            for r in recs:
                f.write(
                    f"- **{r.get('priority','')}** — {r.get('action','')} "
                    f"(Owner: {r.get('owner','—')}, "
                    f"Timeline: {r.get('timeline','—')})\n"
                )
            f.write("\n")

        f.write("---\n\n")

    # -------------------------------------------------
    # HEADER & FOOTER
    # -------------------------------------------------
    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write("# Sreejita Executive Report\n\n")
        f.write(
            f"**Run ID:** `{run_id}` | "
            f"**Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n\n"
        )

        if isinstance(metadata, dict):
            for k, v in metadata.items():
                f.write(f"- **{str(k).replace('_',' ').title()}**: {v}\n")

        f.write("\n---\n\n")

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write("_Generated by **Sreejita Universal Domain Intelligence**_\n")

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def _sort_domains(self, domains: List[str]) -> List[str]:
        priority = ["healthcare", "finance", "retail", "marketing"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _format_value(self, key: str, v: Any) -> str:
        if isinstance(v, (int, float)):
            if "rate" in key:
                return f"{v:.1%}"
            if abs(v) >= 1_000_000:
                return f"{v / 1_000_000:.1f}M"
            if abs(v) >= 1_000:
                return f"{v / 1_000:.1f}K"
            return f"{v:.2f}"
        return str(v)


# =====================================================
# PUBLIC ENTRY POINT (BACKWARD COMPATIBILITY)
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    from sreejita.reporting.orchestrator import generate_report_payload

    run_dir = Path(config.get("run_dir", "./runs"))
    run_dir.mkdir(parents=True, exist_ok=True)

    domain_results = generate_report_payload(input_path, config)

    engine = HybridReport()
    md_path = engine.build(
        domain_results=domain_results,
        output_dir=run_dir,
        metadata=config.get("metadata"),
    )

    domains = list(domain_results.keys())
    primary_domain = engine._sort_domains(domains)[0]
    primary = domain_results.get(primary_domain) or {}

    return {
        "markdown": str(md_path),
        "domain_results": domain_results,
        "primary_domain": primary_domain,
        "executive": primary.get("executive", {}),
        "visuals": primary.get("visuals", []),
        "insights": primary.get("insights", []),
        "recommendations": primary.get("recommendations", []),
        "run_dir": str(run_dir),
    }
