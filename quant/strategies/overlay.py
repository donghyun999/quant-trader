"""다른 전략 위에 얹는 오버레이 전략: 변동성 목표(vol targeting), 전략 앙상블."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Param, Strategy, align_prices, common_index


def _registry():
    from . import STRATEGIES  # 순환 import 회피

    return STRATEGIES


def _portfolio_returns(weights: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """어제 결정한 비중으로 오늘 얻는 수익률 (shift 1 → lookahead 없음)."""
    rets = close.pct_change().fillna(0.0)
    return (weights.shift(1).fillna(0.0) * rets).sum(axis=1)


class VolTarget(Strategy):
    name = "vol_target"
    label = "변동성 목표 오버레이"
    description = (
        "다른 전략(base)의 비중을 그대로 쓰되, 최근 포트폴리오 변동성이 목표(기본 연 10%)보다 크면 "
        "전체 비중을 줄여 현금을 늘리고, 작으면 최대 100%까지만 투자합니다. "
        "수익을 키우는 장치가 아니라 손실 폭을 일정하게 묶어 두는 장치입니다."
    )
    params = [
        Param("base", "trend_invvol", help="기반 전략 이름 (예: trend_invvol, momentum_ensemble, buy_hold)."),
        Param("target_vol", 0.10, 0.03, 0.30, 0.01, "목표 연 변동성."),
        Param("window", 60, 20, 252, 10, "변동성 측정 기간(거래일)."),
    ]
    multi_asset = True

    def validate(self) -> None:
        if self.p["base"] not in _registry() or self.p["base"] in ("vol_target", "ensemble"):
            raise ValueError(f"base 전략 이름이 잘못됐습니다: {self.p['base']}")
        if self.p["target_vol"] <= 0 or self.p["window"] < 5:
            raise ValueError("target_vol 은 양수, window 는 5 이상이어야 합니다.")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        base = _registry()[self.p["base"]]()
        w = base.generate_weights(aligned).reindex(index=idx, columns=tickers).fillna(0.0)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        pr = _portfolio_returns(w, close)
        realized = pr.rolling(int(self.p["window"]), min_periods=int(self.p["window"])).std(ddof=0) * np.sqrt(252)
        scale = (float(self.p["target_vol"]) / realized).clip(upper=1.0).fillna(1.0)
        # 투자 중이 아닐 때(비중 0)는 변동성이 0 → scale 1 이지만 비중이 0 이라 영향 없음
        return w.mul(scale, axis=0).clip(0.0, 1.0)


class Ensemble(Strategy):
    name = "ensemble"
    label = "전략 앙상블 (비중 평균)"
    description = (
        "여러 전략이 각각 계산한 목표 비중을 평균합니다. 한 전략이 틀려도 다른 전략이 완충해 주어 "
        "최고 전략보다 수익은 낮을 수 있지만 최악의 경우가 완만해집니다."
    )
    params = [
        Param("members", "trend_invvol,momentum_ensemble,buy_hold", help="쉼표로 구분한 전략 이름들."),
    ]
    multi_asset = True

    def validate(self) -> None:
        names = self._members()
        if not names:
            raise ValueError("members 가 비어 있습니다.")
        reg = _registry()
        bad = [n for n in names if n not in reg or n in ("vol_target", "ensemble")]
        if bad:
            raise ValueError(f"앙상블에 넣을 수 없는 전략: {bad}")

    def _members(self) -> list[str]:
        return [s.strip() for s in str(self.p["members"]).split(",") if s.strip()]

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        reg = _registry()
        total = None
        names = self._members()
        for n in names:
            w = reg[n]().generate_weights(aligned).reindex(index=idx, columns=tickers).fillna(0.0)
            total = w if total is None else total + w
        return (total / len(names)).clip(0.0, 1.0)
