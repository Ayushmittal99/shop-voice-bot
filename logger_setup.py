import json
import logging
from logging.handlers import RotatingFileHandler

from config import LOG_DIR


class JsonFormatter(logging.Formatter):
    """One JSON object per line, easy to grep or load into pandas."""

    def format(self, record):
        payload = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_data", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "shop_bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)

    # Human-readable console output
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    )

    # Rotating structured log file (5 MB x 5 backups)
    file_handler = RotatingFileHandler(
        LOG_DIR / "pipeline.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def log_event(logger: logging.Logger, message: str, **data):
    """Log a message with structured key/value data."""
    logger.info(message, extra={"extra_data": data})
