"""Root pytest configuration: skip doctests of integrations whose framework is missing."""

from __future__ import annotations

import importlib.util

collect_ignore = (
    [] if importlib.util.find_spec("lightning") else ["src/aucreshape/integrations/lightning.py"]
)
