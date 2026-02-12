from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from sreejita.reporting.base import BaseReport


# =====================================================
# HYBRID REPORT (v3.5 – MD SOURCE OF TRUTH)
# =====================================================

class HybridReport(BaseReport):
    """
    Hybrid v3.5 Report Engine

    - Deterministic intelligence (v3.4)
    - Optional AI-assisted narrative layer (v3.5)
    - Markdown remains the single source of truth
    """

    name = "hybrid"

    # -------------------------------------------------
    # ENGINE ENTRY POINT
    # -------------------------------------------------

    def build(
        self,
        domain_results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:

        config = config or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "Sreejita_Executive_Report.md"

        run_id = f"SR-{datetime.utcnow():%Y%m%d}-{uuid.uuid4().hex[:6]}"

        with open(report_path, "w", encoding="utf-8") as f:
            # Header + Executive Summary
            self._write_header(f, run_id, metadata)

            # v3.5 AI Narrative (clearly separated)
            self._write_optional_narrative(
                f,
                run_id,
                domain_results,
                config,
            )

            # Domain sections
            for domain in self._sort_domains(domain_results.keys()):
                self._write_domain_section(
                    f,
                    domain,
                    domain_results.get(domain, {}),
                )

            self._write_footer(f)

        return report_path

    # -------------------------------------------------
    # OPTIONAL NARRATIVE SECTION (v3.5)
    # -------------------------------------------------

    def _write_optional_narrative(
        self,
        f,
        run_id: str,
        domain_results: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
    ):
        narrative_cfg = config.get("narrative", {})
        if not narrative_cfg.get("enabled", False):
            return  # v3.4 behavior

        # Lazy imports (CRITICAL)
        from sreejita.narrative.schema import (
            NarrativeInput,
            NarrativeInsight,
            NarrativeAction,
        )
        from sreejita.narrative.llm import LLMClient
        from sreejita.narrative.composer import generate_narrative

        # Highest-priority domain only
        domain = self._sort_domains(domain_results.keys())[0]
        result = domain_results.get(domain, {})

        insights = self._prioritize_insights(result.get("insights", []))
        recs = result.get("recommendations", [])

        if not insights:
            return

        narrative_input = NarrativeInput(
            run_id=run_id,
            domain=domain.replace("_", " ").title(),
            insights=[
                NarrativeInsight(
                    level=i.get("level"),
                    title=i.get("title"),
                    description=i.get("so_what"),
                )
                for i in insights
                if i.get("title") and i.get("so_what")
            ],
            actions=[
                NarrativeAction(
                    action=r.get("action"),
                    priority=r.get("priority", "HIGH"),
                    timeline=r.get("timeline", "Immediate"),
                )
                for r in recs[:1]
                if r.get("action")
            ],
            confidence_band=narrative_cfg.get("confidence_band", "MEDIUM"),
        )

        llm_client = LLMClient(narrative_cfg)
        narrative_text = generate_narrative(narrative_input, llm_client)

        if not narrative_text:
            return

        f.write("\n## 🤖 AI-Assisted Narrative (Optional)\n\n")
        f.write(
            "> ⚠️ *This section is AI-assisted. It explains existing insights "
            "and recommendations in natural language. "
            "No new metrics or decisions are introduced.*\n\n"
        )
        f.write(narrative_text.strip() + "\n\n")

    # -------------------------------------------------
    # HEADER & EXEC SUMMARY
    # -------------------------------------------------

    def _write_header(self, f, run_id: str, metadata: Optional[Dict[str, Any]]):
        f.write("# 📊 Executive Decision Report\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Run ID:** {run_id}\n")
        f.write(f"- **Generated:** {datetime.utcnow():%Y-%m-%d %H:%M UTC}\n")
        f.write("- **Framework Version:** Sreejita v3.5\n")

        if metadata:
            for k, v in metadata.items():
                f.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")

        f.write(
            "\n> This report presents decision-grade insights generated using "
            "**Sreejita Composite Intelligence**, focusing on risks, "
            "opportunities, and recommended actions.\n\n"
        )

    # -------------------------------------------------
    # DOMAIN SECTIONS (UNCHANGED)
    # -------------------------------------------------

    def _write_domain_section(self, f, domain: str, result: Dict[str, Any]):
        f.write("\n---\n\n")
        f.write(f"## 🔹 {domain.replace('_', ' ').title()}\n\n")

        if not isinstance(result, dict):
            f.write("_Invalid domain output._\n\n")
            return

        kpis = result.get("kpis", {})
        insights = self._prioritize_insights(result.get("insights", []))
        recs = result.get("recommendations", [])
        visuals = result.get("visuals", [])

        # Strategic Intelligence
        f.write("### 🧠 Strategic Intelligence\n")
        if insights:
            for ins in insights:
                if not ins.get("title") or not ins.get("so_what"):
                    continue
                f.write(
                    f"#### {self._level_icon(ins.get('level'))} {ins['title']}\n"
                )
                f.write(f"{ins['so_what']}\n\n")
        else:
            f.write("_Operations within normal parameters._\n\n")

        # KPIs
        if isinstance(kpis, dict) and kpis:
            f.write("### 📉 Key Performance Indicators\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")

            for k, v in list(kpis.items())[:8]:
                f.write(
                    f"| {k.replace('_', ' ').title()} | "
                    f"**{self._format_value(k, v)}** |\n"
                )
            f.write("\n")

        # Visual Evidence
        if isinstance(visuals, list) and visuals:
            f.write("### 👁️ Visual Evidence\n")
            for idx, vis in enumerate(visuals[:2], start=1):
                path = vis.get("path")
                if not path:
                    continue
                img = Path(path).name
                caption = vis.get("caption", "Visualization")
                f.write(f"![{caption}]({img})\n")
                f.write(f"> *Fig {idx}.1 — {caption}*\n\n")

        # Action Plan
        if isinstance(recs, list) and recs:
            primary = recs[0]
            if "action" in primary:
                f.write("### 🚀 Action Plan\n")
                f.write("| Action | Priority | Timeline |\n")
                f.write("| :--- | :--- | :--- |\n")
                f.write(
                    f"| {primary['action']} | "
                    f"{primary.get('priority', 'HIGH')} | "
                    f"{primary.get('timeline', 'Immediate')} |\n\n"
                )

    # -------------------------------------------------
    # FOOTER & HELPERS
    # -------------------------------------------------

    def _write_footer(self, f):
        f.write("\n---\n")
        f.write(
            "_Prepared by **Sreejita Data Labs** · "
            "Framework v3.5 · Confidential_\n"
        )

    def _prioritize_insights(self, insights: List[Dict[str, Any]]):
        order = {"RISK": 0, "WARNING": 1, "INFO": 2}
        return sorted(insights, key=lambda i: order.get(i.get("level"), 3))[:5]

    def _sort_domains(self, domains):
        priority = ["finance", "retail", "ecommerce", "supply_chain", "healthcare"]
        return sorted(domains, key=lambda d: priority.index(d) if d in priority else 99)

    def _level_icon(self, level: str):
        return {"RISK": "🔴", "WARNING": "🟠", "INFO": "🔵"}.get(level, "ℹ️")

    def _format_value(self, key: str, v: Any):
        if isinstance(v, float):
            if any(
                x in key.lower()
                for x in ["rate", "ratio", "margin", "conversion"]
            ) and abs(v) <= 2:
                return f"{v:.1%}"
            return f"{v:,.2f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)


# =====================================================
# BACKWARD-COMPATIBLE ENTRY POINT
# =====================================================

def run(input_path: str, config: Dict[str, Any]) -> str:
    from sreejita.reporting.orchestrator import generate_report_payload

    domain_results = generate_report_payload(
        input_path=input_path,
        config=config,
    )

    output_dir = Path(config.get("output_dir", "runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = HybridReport()
    return str(engine.build(domain_results, output_dir, config.get("metadata"), config))
