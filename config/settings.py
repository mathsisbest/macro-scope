"""Convenience re-export so `config.settings` and `mmi.settings` both work.

The canonical implementation lives in the installed package (``mmi.settings``).
"""

from mmi.settings import DEFAULT_EVENTS, Settings, get_settings, load_assets, load_events, settings

__all__ = ["DEFAULT_EVENTS", "Settings", "get_settings", "load_assets", "load_events", "settings"]
