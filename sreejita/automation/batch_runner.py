import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from sreejita.reporting.hybrid import run as run_hybrid
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger
from sreejita.automation.retry import retry

log = get_logger("batch-runner")

SUPPORTED_EXT = (".csv", ".xlsx")


# =====================================================
# PROCESS SINGLE FILE (BATCH SAFE, GOVERNED)
# =====================================================

@retry(times=3, delay=5)
def run_single_file(
    file_path: Path,
    config: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Stabilization-mode batch contract:

    - Markdown may always be generated
    - Executive PDF ONLY if domain is confidently detected
    - Ambiguity never crashes batch
    - No fabricated payloads
    """

    src = Path(file_path)

    # -------------------------------------------------
    # 1️⃣ Create isolated per-file run directory
    # -------------------------------------------------
    file_run_dir = run_dir / src.stem
    input_dir = file_run_dir / "input"
    failed_dir = file_run_dir / "failed"

    input_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    dst = input_dir / src.name
    dst.write_bytes(src.read_bytes())

    log.info("Processing file: %s", src.name)

    # -------------------------------------------------
    # 2️⃣ Force Hybrid output into this folder
    # -------------------------------------------------
    local_config = dict(config)
    local_config["run_dir"] = str(file_run_dir)

    # -------------------------------------------------
    # 3️⃣ Generate Hybrid Output
    # -------------------------------------------------
    result = run_hybrid(str(dst), local_config)

    if not isinstance(result, dict):
        raise RuntimeError("Hybrid returned invalid result")

    md_path = Path(result.get("markdown", ""))

    if not md_path.exists():
        raise RuntimeError(f"Markdown not generated: {md_path}")

    log.info("Markdown generated: %s", md_path.name)

    # -------------------------------------------------
    # 4️⃣ Inspect domain decision (if present)
    # -------------------------------------------------
    decision = result.get("decision")
    domain_status = getattr(decision, "status", None)
    primary_domain = getattr(decision, "selected_domain", None)

    pdf_path = None

    # -------------------------------------------------
    # 5️⃣ Executive PDF (ONLY IF SAFE)
    # -------------------------------------------------
    if domain_status == "detected" and primary_domain:
        payload = result.get("payload")

        if isinstance(payload, dict):
            try:
                from sreejita.reporting.pdf_renderer import (
                    ExecutivePDFRenderer,
                )

                pdf_renderer = ExecutivePDFRenderer()
                pdf_path = file_run_dir / "Sreejita_Executive_Report.pdf"

                pdf_renderer.render(
                    payload=payload,
                    output_path=pdf_path,
                )

                log.info("PDF generated: %s", pdf_path)

            except Exception as e:
                log.warning(
                    "PDF generation failed (non-blocking): %s",
                    e,
                )
        else:
            log.warning(
                "PDF skipped: payload missing or invalid for %s",
                src.name,
            )

    else:
        log.warning(
            "PDF skipped due to ambiguous or insufficient domain: %s",
            src.name,
        )

    log.info("Completed file: %s", src.name)

    return {
        "file": src.name,
        "markdown": str(md_path),
        "pdf": str(pdf_path) if pdf_path else None,
        "run_dir": str(file_run_dir),
    }


# =====================================================
# BATCH ENTRY POINT
# =====================================================

def run_batch(
    input_folder: str,
    config_path: Optional[str],
    output_root: str = "runs",
):
    """
    Stabilization-mode batch runner:

    - One timestamped run directory
    - One subfolder per file
    - Safe retries
    - Zero global failure risk
    - Governance preserved at scale
    """

    config = load_config(config_path)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

    log.info("Found %d input files", len(files))
    log.info("Batch run directory: %s", run_dir)

    for file in files:
        src = Path(input_folder) / file

        try:
            run_single_file(src, config, run_dir)

        except Exception as e:
            failed_path = (
                run_dir
                / "failed"
                / f"{src.stem}_{int(datetime.utcnow().timestamp())}{src.suffix}"
            )
            failed_path.parent.mkdir(exist_ok=True)
            failed_path.write_bytes(src.read_bytes())

            log.error(
                "File failed after retries: %s | Reason: %s",
                src.name,
                str(e),
            )

    log.info("Batch run completed successfully: %s", run_dir)
