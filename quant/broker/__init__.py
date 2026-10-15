"""브로커 구현 모음."""
from __future__ import annotations

from .base import Balance, Broker, BrokerError, OrderResult, Position
from .paper import PaperBroker


def make_broker(kind: str, **kw) -> Broker:
    """이름으로 브로커 생성. kind: "paper" | "kis"."""
    kind = (kind or "paper").lower()
    if kind == "paper":
        return PaperBroker(**kw)
    if kind == "kis":
        from .kis import KISBroker  # requests 지연 import

        return KISBroker(**kw)
    raise ValueError(f"알 수 없는 브로커: {kind} (가능: paper, kis)")


__all__ = ["Balance", "Broker", "BrokerError", "OrderResult", "Position", "PaperBroker", "make_broker"]
