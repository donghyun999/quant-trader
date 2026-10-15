from __future__ import annotations

import pandas as pd

from ..indicators import sma
from .base import Param, Strategy


class SmaCross(Strategy):
    name = "sma_cross"
    label = "이동평균 골든크로스"
    description = (
        "짧은 기간 평균가격(단기 이동평균)이 긴 기간 평균가격(장기 이동평균)보다 위에 있으면 "
        "상승 추세로 보고 주식을 보유하고, 아래로 내려가면 팔고 현금으로 쉽니다. "
        "큰 하락장을 피하는 데 도움이 되지만, 횡보장에서는 잦은 매매로 손해를 볼 수 있습니다."
    )
    params = [
        Param("fast", 20, 2, 100, 1, "단기 이동평균 기간(거래일). 짧을수록 신호가 빠르지만 잦아집니다."),
        Param("slow", 60, 10, 300, 5, "장기 이동평균 기간(거래일). 단기보다 커야 합니다."),
    ]

    def validate(self) -> None:
        if self.p["fast"] < 1 or self.p["slow"] < 2:
            raise ValueError("이동평균 기간은 양수여야 합니다.")
        if self.p["fast"] >= self.p["slow"]:
            raise ValueError("단기 이동평균(fast)은 장기(slow)보다 작아야 합니다.")

    def signal(self, df: pd.DataFrame) -> pd.Series:
        f = sma(df["close"], self.p["fast"])
        s = sma(df["close"], self.p["slow"])
        return (f > s).astype(float).where(s.notna() & f.notna(), 0.0)
