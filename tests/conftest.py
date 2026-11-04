"""Use a stand-in for yfinance only when the real package isn't installed."""
import sys
import types

try:
    import yfinance  # noqa: F401
except ImportError:
    sys.modules["yfinance"] = types.SimpleNamespace(download=lambda *a, **k: None)
