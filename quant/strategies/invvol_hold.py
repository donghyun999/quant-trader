from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Param, Strategy, align_prices, common_index
from .dual_momentum import month_end_mask
from .trend_invvol import inverse_vol_weights


class InverseVolHold(Strategy):
    name = "invvol_hold"
    label = "변동성 역가중 보유 (항상 투자)"
    description = (
        "추세 판단 없이 항상 전부 투자하되, 종목 간 비중을 균등이 아니라 최근 변동성이 낮을수록 크게 둡니다(월간 갱신). "
        "단순 보유의 대안으로, 변동성 큰 종목(개별주)에 위험이 몰리는 것을 줄입니다."
    )
    params = [
        Param("vol_window", 60, 20, 252, 10, "변동성 계산 기간(거래일)."),
        Param("max_weight", 0.5, 0.1, 1.0, 0.05, "한 종목 최대 비중. 넘치는 몫은 현금."),
        Param("rebal_offset", 0, 0, 15, 1, "비중 갱신 날짜: 월말에서 며칠(거래일) 뒤. 0=월말.", advanced=True),
    ]
    multi_asset = True

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        valid = close.notna().to_numpy()
        cap = float(self.p["max_weight"])
        w = pd.DataFrame(np.nan, index=idx, columns=tickers)
        for i in np.flatnonzero(month_end_mask(idx, self.p["rebal_offset"])):
            row = inverse_vol_weights(close, i, valid[i], self.p["vol_window"])
            if not row.any():  # 변동성 계산 전 워밍업: 균등 보유
                row = valid[i].astype(float) / max(valid[i].sum(), 1)
            w.iloc[i] = np.minimum(row, cap)
        return w.ffill().fillna(0.0)
