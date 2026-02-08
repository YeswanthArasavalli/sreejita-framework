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
# PROCESS SINGLE FILE (BATCH SAFE, STABILIZED)
# =====================================================

@retry(times=3, delay=5)
def run_single_file(
    file_path: Path,
    config: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Stabilized batch contract:

    - Hybrid is authoritative
    - Ambiguity never crashes batch
    - PDF generation delegated to Hybrid
    - No domain inspection here
    """

    src = Path(file_path)

    # -------------------------------------------------
    # 1️⃣ Per-file run directory
    # -------------------------------------------------
    file_run_dir = run_dir / src.stem
    input_dir = file_run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    dst = input_dir / src.name
    dst.write_bytes(src.read_bytes())

    log.info("Processing file: %s", src.name)

    # -------------------------------------------------
    # 2️⃣ Localized config
    # -------------------------------------------------
    local_config = dict(config)
    local_config["run_dir"] = str(file_run_dir)

    # -------------------------------------------------
    # 3️⃣ Hybrid execution (NEVER TRUST DOMAIN HERE)
    # -------------------------------------------------
    result = run_hybrid(str(dst), local_config)

    if not isinstance(result, dict):
        raise RuntimeError("Hybrid returned invalid result")

    markdown = result.get("markdown")
    pdf = result.get("pdf")

    if markdown:
        md_path = Path(markdown)
        if not md_path.exists():
            raise RuntimeError(f"Markdown path invalid: {markdown}")
        log.info("Markdown generated: %s", md_path.name)
    else:
        log.warning("No markdown generated (ambiguous or insufficient data)")

    if pdf:
        log.info("PDF generated: %s", pdf)
    else:
        log.info("PDF not generated (by design)")

    log.info("Completed file: %s", src.name)

    return {
        "file": src.name,
        "markdown": markdown,
        "pdf": pdf,
        "run_dir": str(file_run_dir),
    }


# =====================================================
# BATCH ENTRY POINT (STABILIZED)
# =====================================================

def run_batch(
    input_folder: str,
    config_path: Optional[str],
    output_root: str = "runs",
):
    """
    Stabilization-mode batch runner:

    - One timestamped batch run
    - One subfolder per file
    - Hybrid governs intelligence
    - Zero cascade failures
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
            failed_dir = run_dir / "failed"
            failed_dir.mkdir(exist_ok=True)

            failed_path = (
                failed_dir
                / f"{src.stem}_{int(datetime.utcnow().timestamp())}{src.suffix}"
            )
            failed_path.write_bytes(src.read_bytes())

            log.error(
                "File failed after retries: %s | Reason: %s",
                src.name,
                str(e),
            )

    log.info("Batch run completed: %s", run_dir)
