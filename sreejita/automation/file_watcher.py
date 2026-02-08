import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from sreejita.automation.batch_runner import run_single_file
from sreejita.config.loader import load_config
from sreejita.utils.logger import get_logger

log = get_logger("file-watcher")

SUPPORTED_EXT = (".csv", ".xlsx")
COOLDOWN_SECONDS = 10
FILE_STABILITY_WAIT = 1.5
FILE_STABILITY_RETRIES = 5


# =====================================================
# FILE SYSTEM EVENT HANDLER (STABILIZED)
# =====================================================

class NewFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        watch_dir: Path,
        config: Dict[str, Any],
        output_root: str = "runs",
    ) -> None:
        self.watch_dir = watch_dir
        self.config = config
        self.output_root = Path(output_root)
        self._cooldown: Dict[str, float] = {}

    # -------------------------------------------------
    # FILE CREATED EVENT
    # -------------------------------------------------
    def on_created(self, event) -> None:
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() not in SUPPORTED_EXT:
            return

        path_key = str(path.resolve())
        now = time.time()

        last_seen = self._cooldown.get(path_key)
        if last_seen and (now - last_seen) < COOLDOWN_SECONDS:
            return

        self._cooldown[path_key] = now

        log.info("New file detected: %s", path.name)

        # -------------------------------------------------
        # WAIT FOR FILE TO STABILIZE
        # -------------------------------------------------
        if not self._wait_for_file_ready(path):
            log.warning(
                "File not stable, skipping: %s",
                path.name,
            )
            return

        # -------------------------------------------------
        # CREATE RUN DIRECTORY
        # -------------------------------------------------
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # PROCESS FILE (SAFE)
        # -------------------------------------------------
        try:
            run_single_file(path, self.config, run_dir)
        except Exception as e:
            failed_dir = run_dir / "failed"
            failed_dir.mkdir(parents=True, exist_ok=True)

            failed_path = failed_dir / path.name
            try:
                failed_path.write_bytes(path.read_bytes())
            except Exception:
                pass

            log.error(
                "File failed after retries: %s | Reason: %s",
                path.name,
                str(e),
            )

    # -------------------------------------------------
    # FILE STABILITY CHECK
    # -------------------------------------------------
    def _wait_for_file_ready(self, path: Path) -> bool:
        """
        Wait until file size stabilizes.
        Prevents processing partially written files.
        """
        try:
            last_size = -1
            for _ in range(FILE_STABILITY_RETRIES):
                if not path.exists():
                    return False

                size = path.stat().st_size
                if size == last_size and size > 0:
                    return True

                last_size = size
                time.sleep(FILE_STABILITY_WAIT)

        except Exception:
            return False

        return False


# =====================================================
# WATCHER ENTRY POINT (STABILIZED)
# =====================================================

def start_watcher(
    watch_dir: str,
    config_path: Optional[str] = None,
    output_root: str = "runs",
) -> None:
    watch_path = Path(watch_dir)

    if not watch_path.exists():
        raise FileNotFoundError(f"Watch directory not found: {watch_path}")

    # -------------------------------------------------
    # LOAD CONFIG (NON-BLOCKING)
    # -------------------------------------------------
    try:
        config = load_config(config_path)
    except Exception as e:
        log.error("Failed to load config: %s", e)
        config = {}

    event_handler = NewFileHandler(
        watch_dir=watch_path,
        config=config,
        output_root=output_root,
    )

    observer = Observer()

    try:
        observer.schedule(
            event_handler,
            str(watch_path),
            recursive=False,
        )
        observer.start()
    except Exception as e:
        log.error("Failed to start file watcher: %s", e)
        return

    log.info("Watching folder: %s", watch_path)
    log.info("Press CTRL+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping file watcher...")
        observer.stop()
    finally:
        observer.join()
        log.info("File watcher stopped cleanly")
