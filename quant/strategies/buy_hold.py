from __future__ import annotations

import pandas as pd

from .base import Strategy, common_index


class BuyHold(Strategy):
    name = "buy_hold"
    label = "사서 보유하기 (Buy & Hold)"
    description = (
        "처음에 모든 종목을 같은 금액씩 사서 계속 들고 있습니다. "
        "아무것도 하지 않는 가장 단순한 방법으로, 다른 전략이 이보다 나은지 비교하는 기준이 됩니다."
    )
    params = []
    multi_asset = True

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        idx = common_index(prices)
        n = len(prices)
        return pd.DataFrame({t: 1.0 / n for t in prices}, index=idx, dtype=float)
