"""기술적 지표. 모두 pd.Series 입력 -> pd.Series 출력 (bollinger 는 Series 3개 튜플).

모든 지표는 해당 시점까지의 데이터만 사용한다 (lookahead 없음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    """단순 이동평균."""
    return close.rolling(int(n), min_periods=int(n)).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """지수 이동평균 (처음 n개 이전은 NaN)."""
    out = close.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI (Wilder 평활). 0~100."""
    n = int(n)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # 손실이 전혀 없으면 RSI=100
    out = out.mask((avg_loss == 0) & avg_gain.notna(), 100.0)
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range (Wilder 평활)."""
    n = int(n)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=True)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def bollinger(
    close: pd.Series, n: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드. 반환: (middle, upper, lower)."""
    n = int(n)
    mid = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + n_std * std, mid - n_std * std


def momentum(close: pd.Series, n: int = 126) -> pd.Series:
    """n 거래일 수익률 (close / close.shift(n) - 1)."""
    return close / close.shift(int(n)) - 1.0
