from typing import Optional, Dict, Any
from apscheduler.schedulers.blocking import BlockingScheduler

from sreejita.automation.batch_runner import run_batch
from sreejita.utils.logger import get_logger

log = get_logger("scheduler")


# =====================================================
# SCHEDULER ENTRY POINT (STABILIZED)
# =====================================================

def start_scheduler(
    schedule_config: Optional[Dict[str, Any]],
    input_dir: str,
    config_path: Optional[str] = None,
    output_root: str = "runs",
) -> None:
    """
    Start time-based batch automation.

    GUARANTEES:
    - Never crashes on bad config
    - Logs misconfiguration clearly
    - Delegates execution to batch runner only
    - Safe shutdown

    schedule_config example:
    {
        "hour": 9,
        "minute": 0
    }
    """

    # -------------------------------------------------
    # 1️⃣ Validate schedule config (NON-BLOCKING)
    # -------------------------------------------------
    if not isinstance(schedule_config, dict) or not schedule_config:
        log.error(
            "Scheduler not started: invalid or missing schedule configuration"
        )
        return

    # Allow only valid cron keys
    allowed_keys = {
        "year",
        "month",
        "day",
        "week",
        "day_of_week",
        "hour",
        "minute",
        "second",
    }

    cron_args = {
        k: v
        for k, v in schedule_config.items()
        if k in allowed_keys and v is not None
    }

    if not cron_args:
        log.error(
            "Scheduler not started: no valid cron parameters found in %s",
            schedule_config,
        )
        return

    # -------------------------------------------------
    # 2️⃣ Initialize scheduler
    # -------------------------------------------------
    scheduler = BlockingScheduler()

    log.info(
        "Starting scheduler | input_dir=%s | output_root=%s | schedule=%s",
        input_dir,
        output_root,
        cron_args,
    )

    # -------------------------------------------------
    # 3️⃣ Register batch job
    # -------------------------------------------------
    scheduler.add_job(
        run_batch,
        trigger="cron",
        id="sreejita-batch-job",
        replace_existing=True,
        kwargs={
            "input_folder": input_dir,
            "config_path": config_path,
            "output_root": output_root,
        },
        **cron_args,
    )

    log.info("Scheduler started successfully. Press CTRL+C to stop.")

    # -------------------------------------------------
    # 4️⃣ Run loop (SAFE SHUTDOWN)
    # -------------------------------------------------
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopping...")
        scheduler.shutdown(wait=False)
        log.info("Scheduler stopped cleanly.")
