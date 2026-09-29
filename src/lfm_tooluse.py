"""Compatibility alias for the LFM parser in pixie_core."""
import importlib as _importlib
import sys as _sys

_sys.modules[__name__] = _importlib.import_module("pixie_core.lfm_tooluse")
