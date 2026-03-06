import logging
import sys
import json
from datetime import datetime, timezone

RESEARVED_KEYS = (
    "name","msg","args","levelname","levelno","pathname","filename",
    "module","exc_info","exc_text","stack_info","lineno","funcName",
    "created","msecs","relativeCreated","thread","threadName",
    "processName","process"
)

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        for key, value in record.__dict__.items():
            if key not in RESEARVED_KEYS and key not in log.keys():
                log[key] = value

        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log, default=str)


def setup_logging(
    level: str | int = logging.INFO,
    json_logs: bool = True,
    output_file: str | None = None,
): # TODO: Make the logger context aware (add contextvars)
    """Sets up a custom logger.

    ---
    Args:
        level: str|int = logging.INFO:
            NOTSET   = 0
            DEBUG    = 10
            INFO     = 20
            WARNING  = 30
            ERROR    = 40
            CRITICAL = 50
        json_logs: bool = True: Whetehr to create a JSON Formatter or not. If True, sets the logger's formatter to `JsonFormatter`
        output_file: str|None: If not None, configures a file handler utf-8 encoded with the same formatter as the stream handler.
    """
    stream_handler = logging.StreamHandler(sys.stdout)

    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)

    if output_file:
        file_handler = logging.FileHandler(output_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
