"""전략 레지스트리."""
from __future__ import annotations

from typing import Any

from .base import Param, Strategy
from .bollinger import BollingerReversion
from .buy_hold import BuyHold
from .dual_momentum import DualMomentum
from .momentum_ensemble import MomentumEnsemble
from .overlay import DrawdownControl, Ensemble, Tranched, VolTarget
from .rsi_reversion import RsiReversion
from .sma_cross import SmaCross
from .trend_invvol import TrendInverseVol

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (SmaCross, RsiReversion, BollingerReversion, DualMomentum, MomentumEnsemble, TrendInverseVol, VolTarget, Tranched, Ensemble, DrawdownControl, BuyHold)
}


def get_strategy(name: str, **params: Any) -> Strategy:
    """이름으로 전략 인스턴스를 만든다. 지정하지 않은 파라미터는 기본값을 쓴다."""
    if name not in STRATEGIES:
        raise ValueError(f"알 수 없는 전략입니다: {name} (가능: {', '.join(STRATEGIES)})")
    return STRATEGIES[name](**params)


__all__ = ["STRATEGIES", "get_strategy", "Strategy", "Param"]
