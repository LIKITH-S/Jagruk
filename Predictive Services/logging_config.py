"""
logging_config.py - Configures structured JSON logging for standard environment deployment.
"""
from __future__ import annotations

import logging
import sys
import json
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    """Encodes log records into JSON strings for log aggregation platforms."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def configure_logger(name: str = "predictive-service", level: int = logging.INFO, to_json: bool = False) -> logging.Logger:
    """Sets up a robust logger with optional structured JSON output format."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    if to_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)-8s in %(name)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        
    logger.addHandler(handler)
    logger.propagate = False
    return logger
