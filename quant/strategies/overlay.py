"""다른 전략 위에 얹는 오버레이 전략: 변동성 목표(vol targeting), 전략 앙상블."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import Param, Strategy, align_prices, common_index


def _registry():
    from . import STRATEGIES  # 순환 import 회피

    return STRATEGIES


def _split_top(text: str, seps: str) -> list[str]:
    """괄호 깊이 0 인 위치의 구분자로만 분리."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch in seps and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return [x for x in out if x]


def parse_spec(spec: str) -> tuple[str, dict]:
    """'trend_invvol' 또는 'trend_invvol(trend_window=100,max_weight=0.5)' -> (이름, 파라미터).

    중첩 허용: 'vol_target(base=ensemble(members=tranched(base=trend_invvol)|buy_hold))'.
    앙상블 members 처럼 쉼표가 바깥 구분자로 쓰이는 곳에서는 파라미터 구분자로 '|' 도 허용한다.
    """
    spec = spec.strip()
    if "(" not in spec:
        return spec, {}
    name, rest = spec.split("(", 1)
    if not rest.endswith(")"):
        raise ValueError(f"괄호가 닫히지 않았습니다: {spec}")
    rest = rest[:-1].strip()
    params: dict = {}
    for kv in _split_top(rest, ","):
        if "=" not in kv:
            raise ValueError(f"파라미터 형식 오류: {kv} (예: key=value)")
        k, v = (x.strip() for x in kv.split("=", 1))
        if re.fullmatch(r"-?\d+", v):
            params[k] = int(v)
        else:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    return name.strip(), params


def build(spec: str) -> Strategy:
    name, params = parse_spec(spec)
    reg = _registry()
    if name not in reg:
        raise ValueError(f"알 수 없는 전략: {name}")
    return reg[name](**params)


def split_members(text: str) -> list[str]:
    """최상위 쉼표 또는 '|' 로만 분리 (괄호 안은 보존). 중첩 스펙 안에서는 '|' 를 쓴다."""
    return _split_top(str(text), ",|")


def _portfolio_returns(weights: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """어제 결정한 비중으로 오늘 얻는 수익률 (shift 1 → lookahead 없음)."""
    rets = close.pct_change().fillna(0.0)
    return (weights.shift(1).fillna(0.0) * rets).sum(axis=1)


class VolTarget(Strategy):
    name = "vol_target"
    label = "변동성 목표 오버레이"
    description = (
        "다른 전략(base)의 비중을 그대로 쓰되, 최근 포트폴리오 변동성이 목표(기본 연 15%)보다 크면 "
        "전체 비중을 줄여 현금을 늘리고, 작으면 최대 100%까지만 투자합니다. "
        "수익을 키우는 장치가 아니라 손실 폭을 일정하게 묶어 두는 장치입니다."
    )
    params = [
        Param("base", "trend_invvol", help="기반 전략 이름 (예: trend_invvol, momentum_ensemble, buy_hold)."),
        Param("target_vol", 0.15, 0.03, 0.30, 0.01, "목표 연 변동성. 10%는 한국 개별주처럼 변동성 큰 자산에, 15%는 범용."),
        Param("window", 60, 20, 252, 10, "변동성 측정 기간(거래일)."),
    ]
    multi_asset = True

    def validate(self) -> None:
        name, _ = parse_spec(str(self.p["base"]))
        if name not in _registry() or name == "vol_target":
            raise ValueError(f"base 전략 이름이 잘못됐습니다: {self.p['base']}")
        if self.p["target_vol"] <= 0 or self.p["window"] < 5:
            raise ValueError("target_vol 은 양수, window 는 5 이상이어야 합니다.")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        base = build(str(self.p["base"]))
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
        Param("members", "tranched(base=trend_invvol),tranched(base=momentum_ensemble),buy_hold",
              help="쉼표로 구분한 전략 이름들. 괄호로 파라미터 지정 가능: tranched(base=trend_invvol)"),
        Param("weights", "", help="멤버별 가중치 (쉼표 또는 | 구분, 예: 2,1,1). 비우면 균등."),
    ]
    multi_asset = True

    def validate(self) -> None:
        names = self._members()
        if not names:
            raise ValueError("members 가 비어 있습니다.")
        reg = _registry()
        bad = [n for n in names if parse_spec(n)[0] not in reg or parse_spec(n)[0] == "ensemble"]
        if bad:
            raise ValueError(f"앙상블에 넣을 수 없는 전략: {bad}")
        w = self._weights()
        if w is not None and (len(w) != len(names) or min(w) < 0 or sum(w) <= 0):
            raise ValueError("weights 는 members 수와 같아야 하고 음수가 아니어야 합니다.")

    def _weights(self) -> list[float] | None:
        raw = str(self.p.get("weights", "")).strip()
        if not raw:
            return None
        return [float(x) for x in re.split(r"[,|]", raw) if x.strip()]

    def _members(self) -> list[str]:
        return split_members(str(self.p["members"]))

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        total = None
        names = self._members()
        wts = self._weights() or [1.0] * len(names)
        norm = sum(wts)
        for n, wt in zip(names, wts):
            w = build(n).generate_weights(aligned).reindex(index=idx, columns=tickers).fillna(0.0) * (wt / norm)
            total = w if total is None else total + w
        return total.clip(0.0, 1.0)


class Tranched(Strategy):
    name = "tranched"
    label = "분할 리밸런스 (트랜치)"
    description = (
        "월간 전략을 여러 조각(트랜치)으로 나눠 각 조각을 서로 다른 날(월말, +5일, +10일, ...)에 리밸런스하고 "
        "비중을 평균합니다. '어느 날에 리밸런스했느냐'는 운을 제거해 결과를 안정시킵니다. "
        "base 전략은 rebal_offset 파라미터가 있는 월간 전략이어야 합니다."
    )
    params = [
        Param("base", "trend_invvol", help="기반 월간 전략 (trend_invvol, momentum_ensemble, dual_momentum)."),
        Param("n_tranches", 4, 1, 8, 1, "조각 수. 4면 월말, +5, +10, +15 거래일."),
        Param("spacing", 5, 1, 10, 1, "조각 간 간격(거래일)."),
    ]
    multi_asset = True

    def validate(self) -> None:
        name, params = parse_spec(str(self.p["base"]))
        reg = _registry()
        if name not in reg or not any(p.name == "rebal_offset" for p in reg[name].params):
            raise ValueError(f"tranched 의 base 는 rebal_offset 이 있는 월간 전략이어야 합니다: {self.p['base']}")
        if "rebal_offset" in params:
            raise ValueError("base 스펙에 rebal_offset 을 직접 주지 마세요. tranched 가 정합니다.")
        if self.p["n_tranches"] < 1 or self.p["spacing"] < 1:
            raise ValueError("n_tranches, spacing 은 1 이상이어야 합니다.")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        name, params = parse_spec(str(self.p["base"]))
        reg = _registry()
        total = None
        n = int(self.p["n_tranches"])
        for k in range(n):
            strat = reg[name](**params, rebal_offset=k * int(self.p["spacing"]))
            w = strat.generate_weights(aligned).reindex(index=idx, columns=tickers).fillna(0.0)
            total = w if total is None else total + w
        return (total / n).clip(0.0, 1.0)


class DrawdownControl(Strategy):
    name = "dd_control"
    label = "낙폭 제어 오버레이"
    description = (
        "다른 전략(base)의 비중을 그대로 쓰되, 그 전략의 자산 곡선이 고점 대비 일정 비율(기본 10%) 이상 떨어져 있는 동안 "
        "전체 비중을 절반으로 줄입니다. 낙폭이 고점의 절반 이하로 회복되면 원래 비중으로 돌아갑니다."
    )
    params = [
        Param("base", "ensemble", help="기반 전략 스펙."),
        Param("dd_trigger", 0.10, 0.03, 0.30, 0.01, "이 낙폭을 넘으면 비중 축소."),
        Param("scale", 0.5, 0.0, 1.0, 0.05, "축소 시 비중 배수."),
    ]
    multi_asset = True

    def validate(self) -> None:
        name, _ = parse_spec(str(self.p["base"]))
        if name not in _registry() or name == "dd_control":
            raise ValueError(f"base 전략 이름이 잘못됐습니다: {self.p['base']}")

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        aligned = align_prices(prices)
        idx = common_index(prices)
        tickers = list(aligned)
        w = build(str(self.p["base"])).generate_weights(aligned).reindex(index=idx, columns=tickers).fillna(0.0)
        close = pd.DataFrame({t: aligned[t]["close"] for t in tickers}, index=idx)
        eq = (1.0 + _portfolio_returns(w, close)).cumprod()
        dd = eq / eq.cummax() - 1.0
        trig, half = -float(self.p["dd_trigger"]), -float(self.p["dd_trigger"]) / 2
        state = hysteresis_series(dd < trig, dd > half)   # 낙폭 깊어지면 1(축소), 절반 회복하면 0
        scale = pd.Series(np.where(state > 0, float(self.p["scale"]), 1.0), index=idx)
        return w.mul(scale, axis=0).clip(0.0, 1.0)


def hysteresis_series(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    from .base import hysteresis

    return hysteresis(entry, exit_)
