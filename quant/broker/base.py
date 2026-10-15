"""브로커(증권사) 공통 인터페이스.

실전 자동매매 파이프라인(quant/live.py)은 이 인터페이스만 보고 동작한다.
구현체: PaperBroker(가상), KISBroker(한국투자증권 Open API).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass
class Position:
    ticker: str          # 프로그램 내부 티커 (예: 005930.KS, SPY)
    qty: int
    avg_price: float
    price: float         # 현재가 (조회 시점)

    @property
    def value(self) -> float:
        return self.qty * self.price


@dataclass
class Balance:
    cash: float                           # 주문 가능 현금
    positions: dict[str, Position] = field(default_factory=dict)
    currency: str = "KRW"

    @property
    def positions_value(self) -> float:
        return sum(p.value for p in self.positions.values())

    @property
    def total(self) -> float:
        return self.cash + self.positions_value

    def weights(self) -> dict[str, float]:
        tot = self.total
        return {t: (p.value / tot if tot > 0 else 0.0) for t, p in self.positions.items()}


@dataclass
class OrderResult:
    ticker: str
    side: Side
    qty: int
    price: float                 # 주문가 (시장가면 참고 현재가)
    ok: bool
    order_id: str = ""
    message: str = ""
    broker: str = ""
    dry_run: bool = False
    time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class BrokerError(RuntimeError):
    """증권사 API 오류."""


class Broker:
    """브로커 베이스 클래스. 모든 메서드는 동기 호출."""

    name: str = "base"
    #: 실제 돈이 오가는 브로커이면 True (안전장치 분기용)
    real_money: bool = False

    def connect(self) -> None:
        """토큰 발급 등 연결 준비. 실패 시 BrokerError."""

    def get_balance(self, tickers: list[str] | None = None) -> Balance:
        """현금 + 보유 종목. tickers 를 주면 현재가 조회 대상 힌트로 쓴다."""
        raise NotImplementedError

    def get_price(self, ticker: str) -> float:
        """현재가(최근 체결가)."""
        raise NotImplementedError

    def submit_order(self, ticker: str, side: Side, qty: int, price: float | None = None) -> OrderResult:
        """주문 전송. price=None 이면 시장가(또는 브로커가 허용하는 가장 가까운 방식)."""
        raise NotImplementedError

    def is_market_open(self) -> bool:
        """장 운영 시간이면 True. 모르면 True."""
        return True

    def close(self) -> None:
        """자원 정리."""
