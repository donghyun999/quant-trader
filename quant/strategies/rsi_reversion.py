from __future__ import annotations

import pandas as pd

from ..indicators import rsi
from .base import Param, Strategy, hysteresis


class RsiReversion(Strategy):
    name = "rsi_reversion"
    label = "RSI 과매도 반등"
    description = (
        "RSI는 최근 오르내림의 강도를 0~100으로 나타낸 값입니다. "
        "RSI가 과매도 기준(기본 30) 아래로 떨어지면 너무 많이 빠졌다고 보고 사고, "
        "청산 기준(기본 55) 위로 회복하면 팝니다. 단기 반등을 노리는 전략입니다."
    )
    params = [
        Param("period", 14, 2, 50, 1, "RSI 계산 기간(거래일)."),
        Param("oversold", 30, 5, 50, 1, "RSI가 이 값보다 낮아지면 매수합니다."),
        Param("exit_level", 55, 30, 95, 1, "RSI가 이 값보다 높아지면 매도합니다."),
    ]

    def validate(self) -> None:
        if self.p["period"] < 2:
            raise ValueError("RSI 기간은 2 이상이어야 합니다.")
        if self.p["oversold"] >= self.p["exit_level"]:
            raise ValueError("과매도 기준(oversold)은 청산 기준(exit_level)보다 작아야 합니다.")

    def signal(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.p["period"])
        return hysteresis(r < self.p["oversold"], r > self.p["exit_level"])
