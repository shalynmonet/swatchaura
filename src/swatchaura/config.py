"""Read local application settings from environment variables and .env."""

import os

from dotenv import load_dotenv


def mock_mode_enabled() -> bool:
    """Return whether local mock implementations should be used.

    MOCK_MODE defaults to true so a fresh checkout works without API access.
    """
    load_dotenv()
    value = os.getenv("MOCK_MODE", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}
