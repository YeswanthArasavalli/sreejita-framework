"""
Sreejita Framework CLI
v3.6 — Universal Domain Intelligence
Markdown + ReportLab PDF (STABILIZATION MODE)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import importlib

from sreejita.__version__ import __version__
from sreejita.config.loader import load_config
from sreejita.automation.batch_runner import run_batch
from sreejita.automation.file_watcher import start_watcher
from sreejita.automation.scheduler import start_scheduler

logger = logging.getLogger("sreejita.cli")


# =====================================================
# PROGRAMMATIC ENTRY (CLI / UI / API)
# =====================================================
def run_single_file(
    input_path: str,
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    generate_pdf: bool = False,
    domain_hint: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Programmatic entry point.

    Returns:
    {
        "markdown": <path>,
        "pdf": <path or None>,
        "run_dir": <path>
    }
    """

    # -------------------------------------------------
    # Bootstrap domains (MANDATORY)
    # -------------------------------------------------
    importlib.import_module("sreejita.domains.bootstrap_v2")
    hybrid = importlib.import_module("sreejita.reporting.hybrid")

    # -------------------------------------------------
    # CONFIG & RUN DIRECTORY
    # -------------------------------------------------
    if config is not None:
        final_config = dict(config)
        run_dir = Path(final_config["run_dir"])
    else:
        final_config = load_config(config_path)
        run_dir = (
            Path("runs")
            / datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        )
        final_config["run_dir"] = str(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # -------------------------------------------------
    # DOMAIN HINT (ADVISORY ONLY)
    # -------------------------------------------------
    if domain_hint:
        final_config["domain_hint"] = domain_hint
        final_config["domain_hint_forced"] = False
        logger.info(
            "Domain hint provided (advisory only): %s",
            domain_hint,
        )

    # -------------------------------------------------
    # HYBRID REPORT
    # -------------------------------------------------
    result = hybrid.run(input_path, final_config)

    if not isinstance(result, dict):
        raise RuntimeError("Hybrid report returned invalid payload")

    required = {"markdown", "domain_results", "run_dir"}
    missing = required - set(result.keys())
    if missing:
        raise RuntimeError(
            f"Hybrid report missing required keys: {missing}"
        )

    primary_domain = result.get("primary_domain")
    domain_results = result["domain_results"]

    if primary_domain is None:
        logger.warning(
            "No primary domain selected due to low confidence or ambiguity"
        )

    primary_payload: Dict[str, Any] = {}
    if primary_domain and primary_domain in domain_results:
        primary_payload = domain_results[primary_domain]
        if not isinstance(primary_payload, dict):
            raise RuntimeError("Primary domain payload corrupted")

    md_path = Path(result["markdown"])
    pdf_path: Optional[Path] = None

    # -------------------------------------------------
    # EXECUTIVE PDF (GOVERNED)
    # -------------------------------------------------
    if generate_pdf and primary_domain:
        try:
            pdf_mod = importlib.import_module(
                "sreejita.reporting.pdf_renderer"
            )

            renderer = pdf_mod.ExecutivePDFRenderer()
            pdf_path = run_dir / "Sreejita_Executive_Report.pdf"

            pdf_payload = {
                "domain": primary_domain,
                "executive": primary_payload.get("executive", {}),
                "visuals": primary_payload.get("visuals", []),
                "insights": primary_payload.get("insights", []),
                "recommendations": primary_payload.get(
                    "recommendations", []
                ),
                "kpis": primary_payload.get("kpis", {}),
            }

            renderer.render(
                payload=pdf_payload,
                output_path=pdf_path,
            )

            logger.info("PDF generated: %s", pdf_path)

        except Exception:
            logger.exception("PDF generation failed")
            pdf_path = None

    elif generate_pdf:
        logger.warning(
            "PDF generation skipped due to ambiguous or missing domain"
        )

    # -------------------------------------------------
    # FINAL RETURN (UI SAFE)
    # -------------------------------------------------
    return {
        "markdown": str(md_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "run_dir": str(run_dir),
    }


# =====================================================
# CLI ENTRY POINT
# =====================================================
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Sreejita Framework v{__version__}"
    )

    parser.add_argument("input", nargs="?", help="CSV or Excel file")
    parser.add_argument("--config", required=False, help="Config YAML path")
    parser.add_argument("--batch", help="Batch input folder")
    parser.add_argument("--watch", help="Watch folder for new files")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF")
    parser.add_argument("--domain", help="Domain hint")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    # ---- VERSION ----
    if args.version:
        print(f"Sreejita Framework v{__version__}")
        return 0

    # ---- LOGGING ----
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ---- CONFIG VALIDATION ----
    if (args.batch or args.watch or args.schedule or args.input) and not args.config:
        parser.error("--config is required")

    config = load_config(args.config) if args.config else {}

    # ---- WATCH ----
    if args.watch:
        start_watcher(args.watch, args.config)
        return 0

    # ---- SCHEDULE ----
    if args.schedule:
        if not args.batch:
            parser.error("--schedule requires --batch")
        start_scheduler(
            config.get("automation", {}).get("schedule"),
            args.batch,
            args.config,
        )
        return 0

    # ---- BATCH ----
    if args.batch:
        run_batch(args.batch, args.config)
        return 0

    # ---- SINGLE FILE ----
    if not args.input:
        parser.error("Input file required")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    result = run_single_file(
        input_path=str(input_path),
        config_path=args.config,
        generate_pdf=args.pdf,
        domain_hint=args.domain,
    )

    print("\n✅ Report generated successfully")
    print(f"📝 Markdown: {result['markdown']}")
    if result["pdf"]:
        print(f"📄 PDF: {result['pdf']}")
    print(f"📁 Run folder: {result['run_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
