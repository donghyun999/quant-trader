"""테스트 공용 픽스처: 네트워크 없이 합성 가격 데이터 생성."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_fake_prices(
    n_days: int = 600,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """기하 랜덤워크 OHLCV. data.get_prices 와 같은 형식(소문자 컬럼, tz-naive 정렬 인덱스)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    out: dict[str, pd.DataFrame] = {}
    for k, t in enumerate(tickers):
        drift = 0.0003 * (1 + k % 3) - 0.0002
        vol = 0.015 + 0.005 * (k % 3)
        rets = rng.normal(drift, vol, n_days)
        close = 100.0 * (1 + k * 0.5) * np.exp(np.cumsum(rets))
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * np.exp(rng.normal(0, vol / 3, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n_days)))
        volume = rng.integers(100_000, 1_000_000, n_days).astype(float)
        out[t] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
        )
    return out


@pytest.fixture
def make_fake_prices():
    """사용법: prices = make_fake_prices(n_days=600, tickers=("AAA","BBB"), seed=0)"""
    return _make_fake_prices
