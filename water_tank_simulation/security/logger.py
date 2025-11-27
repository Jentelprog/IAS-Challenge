# security/logger.py
import datetime
import os

LOG_PATH = "security_logs.txt"


def log(event: str, level: str = "INFO") -> None:
    """Append a timestamped log entry to the log file."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {level}: {event}\n"

    with open(LOG_PATH, "a") as f:
        f.write(line)


def clear_logs() -> None:
    """Delete all log content (truncate file)."""
    open(LOG_PATH, "w").close()
