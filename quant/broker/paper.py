"""PaperAccount 를 Broker 인터페이스로 감싼 가상 브로커 (실제 돈 없음)."""
from __future__ import annotations

from ..paper import PaperAccount
from .base import Balance, Broker, OrderResult, Position, Side


class PaperBroker(Broker):
    name = "paper"
    real_money = False

    def __init__(self, account: PaperAccount | None = None, path: str = "paper_account.json",
                 fee_pct: float = 0.0015, prices: dict[str, float] | None = None):
        self.account = account or PaperAccount(path)
        self.fee_pct = float(fee_pct)
        self._prices: dict[str, float] = dict(prices or {})

    def set_prices(self, prices: dict[str, float]) -> None:
        """가상 브로커는 외부(데이터 모듈)에서 받은 종가를 현재가로 쓴다."""
        self._prices.update({k: float(v) for k, v in prices.items()})

    def get_price(self, ticker: str) -> float:
        if ticker in self._prices:
            return self._prices[ticker]
        pos = self.account.positions.get(ticker)
        if pos:
            return float(pos["avg_price"])
        raise KeyError(f"{ticker}: 현재가가 없습니다. set_prices 로 먼저 넣어 주세요.")

    def get_balance(self, tickers: list[str] | None = None) -> Balance:
        positions = {}
        for t, pos in self.account.positions.items():
            px = self._prices.get(t, pos["avg_price"])
            positions[t] = Position(t, int(pos["qty"]), float(pos["avg_price"]), float(px))
        return Balance(cash=float(self.account.cash), positions=positions)

    def submit_order(self, ticker: str, side: Side, qty: int, price: float | None = None) -> OrderResult:
        px = float(price) if price is not None else self.get_price(ticker)
        try:
            if side == "BUY":
                o = self.account.buy(ticker, qty, px, self.fee_pct)
            else:
                o = self.account.sell(ticker, qty, px, self.fee_pct)
            self.account.save()
            return OrderResult(ticker, side, qty, px, True, order_id=f"paper-{len(self.account.orders)}",
                               message=f"가상 체결 {o['value']:,.0f}", broker=self.name)
        except ValueError as e:
            return OrderResult(ticker, side, qty, px, False, message=str(e), broker=self.name)
