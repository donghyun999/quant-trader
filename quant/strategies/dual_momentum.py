from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import momentum
from .base import Param, Strategy, align_prices, common_index


def month_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """각 달의 마지막 거래일이면 True.

    마지막 행은 다음 거래일을 모르므로 달력상 그 달의 마지막 영업일인지로 판단한다
    (가격 정보는 미래 것을 쓰지 않음).
    """
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    periods = idx.to_period("M")
    mask = np.zeros(len(idx), dtype=bool)
    mask[:-1] = periods[:-1] != periods[1:]
    last = pd.Timestamp(idx[-1]).normalize()
    mask[-1] = last >= (last + pd.offsets.BMonthEnd(0)).normalize()
    return mask


class DualMomentum(Strategy):
    name = "dual_momentum"
    label = "듀얼 모멘텀 (월간 순위)"
    description = (
        "매달 말, 최근 몇 달간 가장 많이 오른 종목 상위 몇 개를 골라 같은 비중으로 삽니다(상대 모멘텀). "
        "단, 그 종목의 수익률이 마이너스면 사지 않고 그 몫은 현금으로 둡니다(절대 모멘텀). "
        "여러 종목(예: 주식 ETF + 채권 ETF)을 함께 넣어야 효과가 있습니다."
    )
    params = [
        Param("lookback", 126, 20, 252, 1, "수익률을 비교할 기간(거래일). 126일은 약 6개월입니다."),
        Param("top_n", 2, 1, 10, 1, "매달 보유할 상위 종목 수."),
    ]
    multi_asset = True

    def validate(self) -> None:
        if self.p["lookback"] < 1 or self.p["top_n"] < 1:
            raise ValueError("lookback 과 top_n 은 1 이상이어야 합니다.")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        mom = close.apply(lambda s: momentum(s, self.p["lookback"]))
        top_n = min(self.p["top_n"], len(tickers))
        slot = 1.0 / top_n  # 각 자리 비중 (양수 모멘텀이 아니면 그 자리는 현금)

        w = pd.DataFrame(np.nan, index=idx, columns=tickers)
        for i in np.flatnonzero(month_end_mask(idx)):
            valid = mom.iloc[i].dropna()
            row = np.zeros(len(tickers))
            if not valid.empty:
                chosen = valid.sort_values(ascending=False).iloc[:top_n]
                for t in chosen[chosen > 0].index:
                    row[tickers.index(t)] = slot
            w.iloc[i] = row
        # 월말에 정한 비중을 다음 월말까지 그대로 유지
        return w.ffill().fillna(0.0)
