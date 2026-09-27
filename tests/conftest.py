"""Shared fixtures: synthetic OHLCV frames, so no test touches the network."""
import numpy as np
import pandas as pd
import pytest


def make_ohlcv(close: np.ndarray, volume: float = 1_000_000) -> pd.DataFrame:
    """OHLCV frame on business days, with a 1% high/low band around close."""
    idx = pd.bdate_range(end="2026-09-25", periods=len(close))
    close = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({
        "Open":   close,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": float(volume),
    })


@pytest.fixture
def uptrend() -> pd.DataFrame:
    """275 trading days rising 100 → 200: enough history for a real 200 SMA."""
    return make_ohlcv(np.linspace(100, 200, 275))


@pytest.fixture
def downtrend() -> pd.DataFrame:
    return make_ohlcv(np.linspace(200, 100, 275))
