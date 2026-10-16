"""가격 데이터 수집 (yfinance) + parquet 캐시.

- 미국: "SPY", "AAPL" / 한국: "005930.KS"(코스피), "035720.KQ"(코스닥)
- 캐시: data_cache/{ticker}.parquet (전체 기간을 저장해 두고 요청 구간만 잘라서 반환)
"""
from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_TTL_SECONDS = 24 * 60 * 60
COLUMNS = ["open", "high", "low", "close", "volume"]

# 자주 쓰는 종목 힌트 (선택 기능)
TICKER_HINTS: dict[str, str] = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "NAVER": "035420.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "셀트리온": "068270.KS",
    "POSCO홀딩스": "005490.KS",
    "KB금융": "105560.KS",
    "KODEX 200": "069500.KS",
    "TIGER 200": "102110.KS",
    "KODEX 코스닥150": "229200.KS",
    "KODEX 미국S&P500TR": "379800.KS",
    "TIGER 미국나스닥100": "133690.KS",
    "KOSEF 국고채10년": "148070.KS",
    "S&P500": "SPY",
    "나스닥100": "QQQ",
    "미국채20년": "TLT",
    "금": "GLD",
}


def search_ticker_hint(text: str) -> list[tuple[str, str]]:
    """이름 일부로 티커 후보를 찾는다. 반환: [(이름, 티커), ...]"""
    q = text.strip().lower()
    if not q:
        return []
    return [(name, t) for name, t in TICKER_HINTS.items() if q in name.lower() or q in t.lower()]


def normalize_ticker(t: str) -> str:
    """공백 제거/대문자화. 6자리 숫자만 입력하면 코스피(.KS)로 간주한다."""
    t = t.strip().upper()
    if re.fullmatch(r"\d{6}", t):
        t += ".KS"
    return t


def _cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return CACHE_DIR / f"{safe}.parquet"


def _clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance 결과를 표준 형식(소문자 OHLCV, tz-naive 정렬 인덱스)으로 정리."""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yf.download 는 단일 티커여도 (Price, Ticker) MultiIndex 를 반환한다.
        names = list(df.columns.names)
        if "Ticker" in names:
            tickers_in = df.columns.get_level_values("Ticker").unique()
            key = ticker if ticker in tickers_in else tickers_in[0]
            df = df.xs(key, axis=1, level="Ticker")
        else:
            df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: 필요한 컬럼이 없습니다 {missing}")
    df = df[COLUMNS].astype(float)
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    # 거래량 0 이고 시가=고가=저가=종가 인 행은 휴장일 잔재(거래 없음)로 보고 제거한다
    flat = (df["volume"] <= 0) & (df["open"] == df["close"]) & (df["high"] == df["low"])
    df = df[~flat]
    return df


def _download(ticker: str) -> pd.DataFrame:
    import yfinance as yf  # 지연 import (테스트에서는 네트워크 미사용)

    raw = yf.download(
        ticker, period="max", auto_adjust=True, progress=False, threads=False
    )
    return _clean(raw, ticker)


def _load_one(ticker: str, force: bool) -> pd.DataFrame:
    path = _cache_path(ticker)
    if not force and path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS:
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                return df
        except Exception:  # 손상된 캐시는 무시하고 재다운로드
            pass
    df = _download(ticker)
    if not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path)
        except Exception as e:  # 캐시 실패는 치명적이지 않음
            warnings.warn(f"{ticker}: 캐시 저장 실패 ({e})")
    elif path.exists():
        # 다운로드 실패 시 오래된 캐시라도 사용
        try:
            df = pd.read_parquet(path)
        except Exception:
            pass
    return df


def _period_start(period: str, end: pd.Timestamp) -> pd.Timestamp | None:
    p = (period or "max").strip().lower()
    if p == "max":
        return None
    if p == "ytd":
        return pd.Timestamp(year=end.year, month=1, day=1)
    m = re.fullmatch(r"(\d+)\s*(d|wk|mo|y)", p)
    if not m:
        raise ValueError(f"알 수 없는 기간 형식입니다: {period} (예: 1y, 3y, 6mo, max)")
    n, unit = int(m.group(1)), m.group(2)
    offset = {
        "d": pd.DateOffset(days=n),
        "wk": pd.DateOffset(weeks=n),
        "mo": pd.DateOffset(months=n),
        "y": pd.DateOffset(years=n),
    }[unit]
    return end - offset


def get_prices(
    tickers: list[str] | str,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """여러 티커의 일봉 OHLCV 를 가져온다.

    반환: {ticker: DataFrame[open, high, low, close, volume]} (수정주가, auto_adjust=True)
    start 가 주어지면 period 는 무시된다. 실패한 티커는 경고 후 제외, 전부 실패하면 ValueError.
    """
    if isinstance(tickers, str):
        tickers = [t for t in tickers.split(",")]
    tickers = list(dict.fromkeys(normalize_ticker(t) for t in tickers if t and t.strip()))
    if not tickers:
        raise ValueError("종목(티커)을 하나 이상 입력하세요.")

    end_ts = pd.Timestamp(end).normalize() if end else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start).normalize() if start else _period_start(period, end_ts)

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = _load_one(t, force)
        except Exception as e:
            warnings.warn(f"{t}: 데이터를 가져오지 못했습니다 ({e})")
            continue
        if df is None or df.empty:
            warnings.warn(f"{t}: 데이터가 없습니다. 티커를 확인하세요 (한국 주식은 005930.KS 형식).")
            continue
        mask = df.index <= end_ts
        if start_ts is not None:
            mask &= df.index >= start_ts
        sl = df.loc[mask]
        if sl.empty:
            warnings.warn(f"{t}: 요청한 기간에 데이터가 없습니다.")
            continue
        out[t] = sl.copy()
    if not out:
        raise ValueError(f"모든 종목의 데이터를 가져오지 못했습니다: {', '.join(tickers)}")
    return out


def fake_prices(tickers: list[str] | str, n_days: int = 600, seed: int = 0) -> dict[str, pd.DataFrame]:
    """네트워크 없이 쓰는 합성 가격 (테스트/데모용). 기하 랜덤워크 OHLCV."""
    import numpy as np

    if isinstance(tickers, str):
        tickers = tickers.split(",")
    tickers = [normalize_ticker(t) for t in tickers if t.strip()]
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days, name="date")
    out: dict[str, pd.DataFrame] = {}
    for k, t in enumerate(tickers):
        rets = rng.normal(0.0003 * (1 + k % 3) - 0.0002, 0.012 + 0.004 * (k % 3), n_days)
        close = 100.0 * (1 + k * 0.5) * np.exp(np.cumsum(rets))
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * np.exp(rng.normal(0, 0.004, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        out[t] = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                               "volume": rng.integers(1e5, 1e6, n_days).astype(float)}, index=idx)
    return out
