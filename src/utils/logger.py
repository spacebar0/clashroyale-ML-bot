"""
Logging Utilities
Structured logging for training and debugging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class Logger:
    """Structured logger with file and console output"""
    
    def __init__(
        self,
        name: str = "ClashRoyaleRL",
        log_dir: Optional[str] = None,
        level: int = logging.INFO
    ):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files (None = no file logging)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_path / f"training_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.info(f"Logging to {log_file}")
    
    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Log critical message"""
        self.logger.critical(msg)


# Global logger instance
_global_logger: Optional[Logger] = None


def get_logger(
    name: str = "ClashRoyaleRL",
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> Logger:
    """
    Get or create global logger
    
    Args:
        name: Logger name
        log_dir: Log directory
        level: Logging level
    
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = Logger(name, log_dir, level)
    
    return _global_logger


if __name__ == "__main__":
    # Test logger
    logger = get_logger(log_dir="logs")
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
