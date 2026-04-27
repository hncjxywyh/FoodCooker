import logging
from logging.handlers import TimedRotatingFileHandler
from food_cooker.settings import settings


def setup_logging() -> None:
    """Configure logging for the entire application.

    Sets up dual output:
    - stdout: INFO and above (visible in Chainlit console)
    - app.log: DEBUG and above, rotated daily at midnight, 7-day retention
    - error.log: ERROR and above only, no rotation (cleaned by app.log's backupCount)

    Unlike basicConfig(), this always takes effect — it injects handlers
    into the root logger even if chainlit has already configured logging.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()  # Clear chainlit's existing handlers

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Console — INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)

        # INFO+ log — rotated daily at midnight, 7-day retention
        info_handler = TimedRotatingFileHandler(
            settings.log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        info_handler.setLevel(logging.DEBUG)  # File records DEBUG for full trace
        info_handler.setFormatter(formatter)
        root.addHandler(info_handler)

        # ERROR log — separate file for error-only monitoring
        error_file = settings.log_file.parent / "error.log"
        error_handler = logging.FileHandler(error_file, encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root.addHandler(error_handler)
