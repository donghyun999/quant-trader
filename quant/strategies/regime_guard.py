"""regime_guard: 완화 추세 필터 (단계적 노출).

회차 28 가설과 결과 (research/LOG.md 참고):
- 추세 ON                    -> 100% 투자
- 추세 OFF                   -> soft_exposure(기본 50%) 만 투자
- (use_stress=1) 추세 OFF + 변동성 스트레스 ON -> 0%. 평가 결과 Sharpe 를 깎고 보호도 못 더해 기본 꺼짐.
추세·스트레스는 묶음 전체의 균등 지수로 판단한다. 노출은 매일, 종목 간 비중(변동성 역가중)은
월간(트랜치 날)에만 갱신한다. 모든 계산은 당일 종가까지의 정보만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import sma
from .base import Param, Strategy, align_prices, common_index
from .dual_momentum import month_end_mask
from .trend_invvol import inverse_vol_weights


class RegimeGuard(Strategy):
    name = "regime_guard"
    label = "완화 추세 필터 (추세 꺾이면 절반만)"
    description = (
        "종목 묶음 전체의 평균 가격이 200일 평균 위면 전부 투자하고, 아래로 내려가면 다 팔지 않고 절반(soft_exposure)만 남깁니다. "
        "노출은 매일 확인하고 종목 간 비중(변동성 역가중)은 월간으로 갱신합니다. "
        "추세 필터(trend_invvol)보다 상승 복귀를 덜 놓치는 대신 위기 때 보호는 약합니다. "
        "변동성 스트레스로 0%까지 내리는 옵션(use_stress)은 평가에서 효과가 없어 기본 꺼짐입니다."
    )
    params = [
        Param("trend_window", 200, 50, 300, 10, "추세 판단용 이동평균 기간(거래일)."),
        Param("vol_short", 20, 5, 60, 5, "단기 변동성 기간(거래일).", advanced=True),
        Param("vol_long", 252, 60, 504, 21, "장기 변동성 기간(거래일).", advanced=True),
        Param("use_stress", 0, 0, 1, 1, "1이면 변동성 스트레스 확인 시 0%로 내려갑니다. 평가 결과 기여가 없어 기본 0 (회차 28).", advanced=True),
        Param("stress_mult", 1.5, 1.0, 3.0, 0.1, "단기 변동성이 장기 변동성의 몇 배를 넘으면 스트레스로 봅니다 (use_stress=1 일 때).", advanced=True),
        Param("soft_exposure", 0.5, 0.0, 1.0, 0.1, "추세 OFF·스트레스 OFF 일 때 투자 비율."),
        Param("vol_window", 60, 20, 252, 10, "종목 간 변동성 역가중 계산 기간(거래일).", advanced=True),
        Param("max_weight", 0.5, 0.1, 1.0, 0.05, "한 종목 최대 비중.", advanced=True),
        Param("rebal_offset", 0, 0, 15, 1, "종목 간 비중 갱신 날짜: 월말에서 며칠(거래일) 뒤. 0=월말.", advanced=True),
        Param("ramp", 0, 0, 2, 1, "0=계단(추세 OFF 면 soft_exposure), 1=추세선 ±band 구간에서 0~100% 대칭 선형, 2=추세선 위 100%, 아래 band 까지 soft_exposure 로 선형 감소.", advanced=True),
        Param("band", 0.05, 0.01, 0.30, 0.01, "ramp 의 거리 폭 (예: 0.05 = 추세선 대비 5%).", advanced=True),
        Param("reentry_window", 0, 0, 200, 10, "0=이탈·재진입 모두 trend_window. N이면 추세 OFF 상태에서 지수가 N일 평균 위로 올라오면 다시 ON (빠른 재진입).", advanced=True),
        Param("stress_days", 1, 1, 20, 1, "스트레스가 이 일수만큼 연속돼야 인정합니다(깜빡임 방지). 1=즉시.", advanced=True),
        Param("daily_exposure", 1, 0, 1, 1, "1이면 노출을 매일 갱신, 0이면 종목 비중 갱신일(트랜치 날)에만 갱신.", advanced=True),
    ]
    multi_asset = True

    def validate(self) -> None:
        if self.p["vol_short"] >= self.p["vol_long"]:
            raise ValueError("vol_short 는 vol_long 보다 작아야 합니다.")
        if not 0.0 <= self.p["soft_exposure"] <= 1.0:
            raise ValueError("soft_exposure 는 0~1 사이여야 합니다.")
        if self.p["stress_mult"] <= 0 or self.p["trend_window"] < 2:
            raise ValueError("stress_mult 는 양수, trend_window 는 2 이상이어야 합니다.")

    def regime(self, close: pd.DataFrame) -> pd.DataFrame:
        """일별 상태표: index_px, trend_on, stress, exposure."""
        index_px = (close / close.bfill().iloc[0]).mean(axis=1)
        trend_ma = sma(index_px, self.p["trend_window"])
        trend_on = (index_px > trend_ma) & trend_ma.notna()
        rw = int(self.p["reentry_window"])
        if rw > 0 and rw != int(self.p["trend_window"]):
            from .base import hysteresis

            fast_ma = sma(index_px, rw)
            # ON 상태: 장기선 아래로 내려가면 OFF. OFF 상태: 단기선 위로 올라오면 ON.
            state = hysteresis(index_px > fast_ma, index_px < trend_ma)
            trend_on = (state > 0) & trend_ma.notna()
        rets = index_px.pct_change()
        vs = rets.rolling(int(self.p["vol_short"]), min_periods=int(self.p["vol_short"])).std(ddof=0)
        vl = rets.rolling(int(self.p["vol_long"]), min_periods=int(self.p["vol_long"])).std(ddof=0)
        stress = (vs > float(self.p["stress_mult"]) * vl) & vl.notna()
        if int(self.p["use_stress"]) == 0:
            stress = pd.Series(False, index=close.index)
        n_days = int(self.p["stress_days"])
        if n_days > 1:
            stress = stress.astype(float).rolling(n_days, min_periods=n_days).sum().eq(n_days)
        soft = float(self.p["soft_exposure"])
        ramp, band = int(self.p["ramp"]), float(self.p["band"])
        if ramp == 0:
            exposure = np.where(trend_on, 1.0, np.where(stress, 0.0, soft))
        else:
            dist = (index_px / trend_ma - 1.0).to_numpy()
            if ramp == 1:
                cont = np.clip(0.5 + dist / (2.0 * band), 0.0, 1.0)
            else:
                cont = np.where(dist >= 0, 1.0, np.clip(1.0 + dist * (1.0 - soft) / band, soft, 1.0))
            exposure = np.where(stress & ~trend_on.to_numpy(), 0.0, cont)
        exposure = pd.Series(exposure, index=close.index)
        exposure[trend_ma.isna()] = 0.0  # 워밍업 구간은 다른 전략과 같이 미투자
        return pd.DataFrame({"index_px": index_px, "trend_on": trend_on, "stress": stress, "exposure": exposure})

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        exposure = self.regime(close)["exposure"]
        cap = float(self.p["max_weight"])
        valid = close.notna().to_numpy()
        rebal_days = month_end_mask(idx, self.p["rebal_offset"])

        base = pd.DataFrame(np.nan, index=idx, columns=tickers)
        for i in np.flatnonzero(rebal_days):
            row = inverse_vol_weights(close, i, valid[i], self.p["vol_window"])
            base.iloc[i] = np.minimum(row, cap)
        base = base.ffill().fillna(0.0)
        if int(self.p["daily_exposure"]) == 0:
            exposure = exposure.where(rebal_days).ffill().fillna(0.0)
        return base.mul(exposure, axis=0).clip(0.0, 1.0)
