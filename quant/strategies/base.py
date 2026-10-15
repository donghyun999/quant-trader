"""전략 베이스 클래스와 파라미터 메타 정의."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd


@dataclass
class Param:
    """전략 파라미터 메타데이터 (UI 자동 생성용)."""

    name: str
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    help: str = ""


def common_index(prices: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """모든 종목에 공통으로 존재하는 날짜 (inner join)."""
    if not prices:
        raise ValueError("가격 데이터가 비어 있습니다.")
    idx: pd.Index | None = None
    for df in prices.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    assert idx is not None
    return pd.DatetimeIndex(idx).sort_values()


def align_prices(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """모든 종목을 공통 기간으로 맞춘다."""
    idx = common_index(prices)
    return {t: df.reindex(idx) for t, df in prices.items()}


def hysteresis(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """진입 조건이 참이면 1, 청산 조건이 참이면 0, 그 외에는 직전 상태 유지 (시작 상태 0)."""
    state = pd.Series(float("nan"), index=entry.index)
    state[exit_.fillna(False).astype(bool)] = 0.0
    state[entry.fillna(False).astype(bool)] = 1.0
    return state.ffill().fillna(0.0)


class Strategy:
    """모든 전략의 베이스.

    단일 종목 전략은 `signal(df) -> pd.Series(0/1)` 만 구현하면 되고,
    여러 종목이 주어지면 종목별로 적용한 뒤 각 1/N 씩 비중을 나눈다.
    다종목 전략(multi_asset=True)은 generate_weights 를 직접 구현한다.
    """

    name: ClassVar[str] = "base"
    label: ClassVar[str] = "기본 전략"
    description: ClassVar[str] = ""
    params: ClassVar[list[Param]] = []
    multi_asset: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        known = {p.name: p for p in self.params}
        unknown = set(params) - set(known)
        if unknown:
            raise ValueError(f"{self.name}: 알 수 없는 파라미터 {sorted(unknown)}")
        self.p: dict[str, Any] = {}
        for name, meta in known.items():
            val = params.get(name, meta.default)
            # UI 에서 float 로 넘어와도 기본값 타입으로 맞춘다
            if isinstance(meta.default, bool):
                val = bool(val)
            elif isinstance(meta.default, int):
                val = int(round(float(val)))
            elif isinstance(meta.default, float):
                val = float(val)
            self.p[name] = val
        self.validate()

    def validate(self) -> None:
        """파라미터 유효성 검사 (잘못되면 ValueError)."""

    def signal(self, df: pd.DataFrame) -> pd.Series:
        """단일 종목 신호 (1=보유, 0=현금). 당일 종가까지만 사용해야 한다."""
        raise NotImplementedError

    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """반환: index=날짜, columns=ticker, 값=목표 비중(0~1). 행 합 <= 1."""
        aligned = align_prices(prices)
        idx = common_index(prices)
        n = len(aligned)
        cols: dict[str, pd.Series] = {}
        for t, df in aligned.items():
            sig = self.signal(df).reindex(idx)
            cols[t] = sig.fillna(0.0).clip(0.0, 1.0).astype(float) / n
        return pd.DataFrame(cols, index=idx)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.p.items())
        return f"{type(self).__name__}({args})"
