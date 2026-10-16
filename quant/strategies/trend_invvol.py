from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import sma
from .base import Param, Strategy, align_prices, common_index
from .dual_momentum import month_end_mask


def inverse_vol_weights(close: pd.DataFrame, i: int, mask: np.ndarray, vol_window: int) -> np.ndarray:
    """i 시점까지의 일별 수익률 변동성 역수로 비중 (mask=True 종목만). 합 1 (mask 전부 False 면 0)."""
    n = close.shape[1]
    w = np.zeros(n)
    if not mask.any() or i < vol_window:
        return w
    rets = close.iloc[i - vol_window:i + 1].pct_change().dropna()
    vol = rets.std(ddof=0).to_numpy()
    inv = np.where(mask & (vol > 0), 1.0 / np.where(vol > 0, vol, 1.0), 0.0)
    if inv.sum() <= 0:
        # 변동성 계산 불가면 균등
        w[mask] = 1.0 / mask.sum()
        return w
    return inv / inv.sum()


class TrendInverseVol(Strategy):
    name = "trend_invvol"
    label = "추세 필터 + 변동성 역가중"
    description = (
        "매달 말, 종목 묶음 전체의 평균 가격이 장기 이동평균(기본 200일) 위에 있을 때만 투자합니다(시장 추세 필터). "
        "투자할 때는 최근 변동성이 작은 종목에 더 큰 비중을 줍니다(변동성 역가중). "
        "시장 추세가 꺾이면 전부 현금으로 둡니다. 파라미터에 덜 민감한 보수적인 방식입니다."
    )
    params = [
        Param("trend_window", 200, 50, 300, 10, "추세 판단용 이동평균 기간(거래일)."),
        Param("vol_window", 60, 20, 252, 10, "변동성 계산 기간(거래일)."),
        Param("max_weight", 0.5, 0.1, 1.0, 0.05, "한 종목 최대 비중. 넘치는 몫은 현금."),
        Param("rebal_offset", 0, 0, 15, 1, "리밸런스 날짜: 월말에서 며칠(거래일) 뒤. 0=월말."),
        Param("market_filter", 1, 0, 1, 1, "1(기본)이면 묶음 전체(균등 지수)의 추세로 투자/현금을 결정, 0이면 종목별 추세."),
    ]
    multi_asset = True

    def validate(self) -> None:
        if self.p["trend_window"] < 2 or self.p["vol_window"] < 2:
            raise ValueError("기간은 2 이상이어야 합니다.")
        if not 0 < self.p["max_weight"] <= 1:
            raise ValueError("max_weight 는 0~1 사이여야 합니다.")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        if int(self.p["market_filter"]) == 1:
            # 묶음 전체의 균등 지수(각 종목 첫 유효가 기준 정규화 평균)로 추세 판단 → 전 종목에 동일 적용
            norm = close / close.bfill().iloc[0]
            index_px = norm.mean(axis=1)
            idx_trend = sma(index_px, self.p["trend_window"])
            on = ((index_px > idx_trend) & idx_trend.notna()).to_numpy()
            above = np.repeat(on[:, None], len(tickers), axis=1) & close.notna().to_numpy()
        else:
            trend = close.apply(lambda s: sma(s, self.p["trend_window"]))
            above = (close > trend).to_numpy() & trend.notna().to_numpy()
        cap = float(self.p["max_weight"])

        w = pd.DataFrame(np.nan, index=idx, columns=tickers)
        for i in np.flatnonzero(month_end_mask(idx, self.p['rebal_offset'])):
            row = inverse_vol_weights(close, i, above[i], self.p["vol_window"])
            w.iloc[i] = np.minimum(row, cap)
        return w.ffill().fillna(0.0)
