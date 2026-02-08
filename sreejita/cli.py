"""
Sreejita Framework CLI
v3.6 — Universal Domain Intelligence
STABILIZATION MODE
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

    GUARANTEES:
    - Never crashes on ambiguity
    - Never forces a domain
    - Always returns paths or None
    """

    # -------------------------------------------------
    # BOOTSTRAP DOMAINS (MANDATORY)
    # -------------------------------------------------
    importlib.import_module("sreejita.domains.bootstrap_v2")
    hybrid = importlib.import_module("sreejita.reporting.hybrid")

    # -------------------------------------------------
    # CONFIG
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
        logger.info("Domain hint provided (advisory): %s", domain_hint)

    # -------------------------------------------------
    # HYBRID REPORT (NEVER FAILS)
    # -------------------------------------------------
    try:
        result = hybrid.run(input_path, final_config)
    except Exception:
        logger.exception("Hybrid execution failed")
        return {
            "markdown": None,
            "pdf": None,
            "run_dir": str(run_dir),
        }

    if not isinstance(result, dict):
        logger.error("Hybrid returned invalid payload")
        return {
            "markdown": None,
            "pdf": None,
            "run_dir": str(run_dir),
        }

    md_path = result.get("markdown")
    domain_results = result.get("domain_results", {})
    primary_domain = result.get("primary_domain")

    # -------------------------------------------------
    # PDF (ONLY IF SAFE)
    # -------------------------------------------------
    pdf_path: Optional[Path] = None

    if generate_pdf and primary_domain and primary_domain in domain_results:
        payload = domain_results.get(primary_domain, {})
        executive = payload.get("executive")

        if isinstance(executive, dict):
            try:
                pdf_mod = importlib.import_module(
                    "sreejita.reporting.pdf_renderer"
                )
                renderer = pdf_mod.ExecutivePDFRenderer()
                pdf_path = run_dir / "Sreejita_Executive_Report.pdf"

                renderer.render(
                    payload={
                        "domain": primary_domain,
                        "executive": executive,
                        "visuals": payload.get("visuals", []),
                        "insights": payload.get("insights", []),
                        "recommendations": payload.get("recommendations", []),
                        "kpis": payload.get("kpis", {}),
                    },
                    output_path=pdf_path,
                )

                logger.info("PDF generated: %s", pdf_path)

            except Exception:
                logger.exception("PDF generation failed")
                pdf_path = None
        else:
            logger.warning("PDF skipped due to suppressed executive content")

    elif generate_pdf:
        logger.warning("PDF generation skipped (no confident domain)")

    # -------------------------------------------------
    # FINAL RETURN (UI SAFE)
    # -------------------------------------------------
    return {
        "markdown": str(md_path) if md_path else None,
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

    # ---- CONFIG ----
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
        print(f"File not found: {input_path}")
        return 1

    result = run_single_file(
        input_path=str(input_path),
        config_path=args.config,
        generate_pdf=args.pdf,
        domain_hint=args.domain,
    )

    print("\n✅ Run completed")
    if result["markdown"]:
        print(f"📝 Markdown: {result['markdown']}")
    if result["pdf"]:
        print(f"📄 PDF: {result['pdf']}")
    print(f"📁 Run folder: {result['run_dir']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
