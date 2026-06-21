"""Convenience re-export so `config.settings` and `mmi.settings` both work.

The canonical implementation lives in the installed package (``mmi.settings``).
"""

from mmi.settings import Settings, get_settings, load_assets, settings

__all__ = ["Settings", "get_settings", "load_assets", "settings"]
