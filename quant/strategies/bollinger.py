from __future__ import annotations

import pandas as pd

from ..indicators import bollinger
from .base import Param, Strategy, hysteresis


class BollingerReversion(Strategy):
    name = "bollinger"
    label = "볼린저밴드 반등"
    description = (
        "최근 평균가격 주변에 정상 범위(볼린저 밴드)를 그립니다. "
        "가격이 범위 아래로 벗어나면 일시적으로 싸졌다고 보고 사고, 평균선(중심선) 위로 돌아오면 팝니다. "
        "하락 추세가 길게 이어지면 손실이 커질 수 있어 손절 설정을 권합니다."
    )
    params = [
        Param("period", 20, 5, 100, 1, "평균과 표준편차를 계산할 기간(거래일)."),
        Param("n_std", 2.0, 0.5, 4.0, 0.25, "밴드 폭(표준편차 배수). 클수록 매수 기회가 드물어집니다."),
    ]

    def validate(self) -> None:
        if self.p["period"] < 2 or self.p["n_std"] <= 0:
            raise ValueError("기간은 2 이상, 밴드 폭은 0보다 커야 합니다.")

    def signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mid, _upper, lower = bollinger(close, self.p["period"], self.p["n_std"])
        return hysteresis(close < lower, close > mid)
