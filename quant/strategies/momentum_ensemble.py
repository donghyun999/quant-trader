from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import momentum
from .base import Param, Strategy, align_prices, common_index
from .dual_momentum import month_end_mask
from .trend_invvol import inverse_vol_weights


class MomentumEnsemble(Strategy):
    name = "momentum_ensemble"
    label = "모멘텀 앙상블 (3·6·12개월 평균)"
    description = (
        "매달 말, 3개월·6개월·12개월 수익률의 평균으로 종목 순위를 매겨 상위 몇 개를 삽니다. "
        "한 가지 기간에만 의존하지 않아 듀얼 모멘텀보다 파라미터 선택에 덜 흔들립니다. "
        "평균 수익률이 마이너스인 종목은 사지 않고 현금으로 두며, 고른 종목 사이에서는 변동성이 낮을수록 비중을 더 줍니다."
    )
    params = [
        Param("top_n", 2, 1, 10, 1, "매달 보유할 상위 종목 수."),
        Param("vol_window", 60, 20, 252, 10, "변동성 역가중 계산 기간(거래일). 0이면 균등 비중."),
        Param("skip_recent", 0, 0, 21, 1, "최근 며칠을 모멘텀 계산에서 제외할지 (단기 되돌림 회피, 21=1개월)."),
    ]
    multi_asset = True
    LOOKBACKS = (63, 126, 252)

    def validate(self) -> None:
        if self.p["top_n"] < 1:
            raise ValueError("top_n 은 1 이상이어야 합니다.")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        skip = int(self.p["skip_recent"])
        base = close.shift(skip) if skip > 0 else close
        score = sum(base.apply(lambda s: momentum(s, lb)) for lb in self.LOOKBACKS) / len(self.LOOKBACKS)
        top_n = min(self.p["top_n"], len(tickers))
        vol_window = int(self.p["vol_window"])

        w = pd.DataFrame(np.nan, index=idx, columns=tickers)
        for i in np.flatnonzero(month_end_mask(idx)):
            valid = score.iloc[i].dropna()
            row = np.zeros(len(tickers))
            if not valid.empty:
                chosen = valid.sort_values(ascending=False).iloc[:top_n]
                chosen = chosen[chosen > 0]
                mask = np.zeros(len(tickers), dtype=bool)
                for t in chosen.index:
                    mask[tickers.index(t)] = True
                if mask.any():
                    if vol_window > 0:
                        iv = inverse_vol_weights(close, i, mask, vol_window)
                        # 빈 자리(top_n 미만 선택)는 현금: 선택 비율만큼만 투자
                        row = iv * (mask.sum() / top_n)
                    else:
                        row[mask] = 1.0 / top_n
            w.iloc[i] = row
        return w.ffill().fillna(0.0)
