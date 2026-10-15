"""페이퍼(가상) 계좌. JSON 파일로 저장하며 실제 주문은 전송하지 않는다."""
from __future__ import annotations

import json
import math
from datetime import date as _date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIN_TRADE_FRAC = 0.01  # 총자산의 1% 미만 거래는 건너뜀(잔돈 방지)


class PaperAccount:
    def __init__(self, path: str = "paper_account.json", initial_cash: float = 10_000_000):
        p = Path(path)
        self.path = p if p.is_absolute() else PROJECT_ROOT / p
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, dict] = {}
        self.history: list[dict] = []
        self.realized_pnl = 0.0
        self.total_deposits = float(initial_cash)
        self.orders: list[dict] = []
        if self.path.exists():
            self.load()

    # ---- 저장/불러오기 ----
    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> None:
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.initial_cash = float(d.get("initial_cash", self.initial_cash))
        self.cash = float(d.get("cash", 0.0))
        self.positions = {
            k: {"qty": int(v["qty"]), "avg_price": float(v["avg_price"])}
            for k, v in d.get("positions", {}).items()
        }
        self.history = list(d.get("history", []))
        self.realized_pnl = float(d.get("realized_pnl", 0.0))
        self.total_deposits = float(d.get("total_deposits", self.initial_cash))
        self.orders = list(d.get("orders", []))

    def save(self) -> None:
        d = {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "positions": self.positions,
            "history": self.history,
            "realized_pnl": self.realized_pnl,
            "total_deposits": self.total_deposits,
            "orders": self.orders,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self, initial_cash: float | None = None) -> None:
        """계좌를 초기화하고 저장한다."""
        if initial_cash is not None:
            self.initial_cash = float(initial_cash)
        self.cash = self.initial_cash
        self.total_deposits = self.initial_cash
        self.positions = {}
        self.history = []
        self.orders = []
        self.realized_pnl = 0.0
        self.save()

    # ---- 입출금/주문 ----
    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("입금액은 0보다 커야 합니다.")
        self.cash += amount
        self.total_deposits += amount

    def buy(self, ticker: str, qty: int, price: float, fee_pct: float = 0.0015,
            date: str | None = None) -> dict:
        qty = int(qty)
        if qty <= 0:
            raise ValueError("수량은 1 이상이어야 합니다.")
        value = qty * price
        fee = value * fee_pct
        if value + fee > self.cash + 1e-9:
            raise ValueError("현금이 부족합니다.")
        self.cash -= value + fee
        pos = self.positions.get(ticker, {"qty": 0, "avg_price": 0.0})
        new_qty = pos["qty"] + qty
        pos["avg_price"] = (pos["avg_price"] * pos["qty"] + value) / new_qty
        pos["qty"] = new_qty
        self.positions[ticker] = pos
        order = self._order(date, ticker, "BUY", qty, price, value, fee)
        self.orders.append(order)
        return order

    def sell(self, ticker: str, qty: int, price: float, fee_pct: float = 0.0015,
             date: str | None = None) -> dict:
        qty = int(qty)
        pos = self.positions.get(ticker)
        if qty <= 0:
            raise ValueError("수량은 1 이상이어야 합니다.")
        if not pos or pos["qty"] < qty:
            raise ValueError("보유 수량이 부족합니다.")
        value = qty * price
        fee = value * fee_pct
        self.cash += value - fee
        self.realized_pnl += (price - pos["avg_price"]) * qty - fee
        pos["qty"] -= qty
        if pos["qty"] == 0:
            del self.positions[ticker]
        order = self._order(date, ticker, "SELL", qty, price, value, fee)
        self.orders.append(order)
        return order

    @staticmethod
    def _order(date, ticker, side, qty, price, value, fee) -> dict:
        return {
            "date": date or _date.today().isoformat(),
            "ticker": ticker, "side": side, "qty": int(qty),
            "price": float(price), "value": float(value), "fee": float(fee),
        }

    # ---- 평가 ----
    def value(self, prices: dict[str, float]) -> dict:
        det = {}
        pv = 0.0
        for t, pos in self.positions.items():
            px = float(prices.get(t, pos["avg_price"]))
            mv = pos["qty"] * px
            pv += mv
            det[t] = {
                "qty": pos["qty"], "avg_price": pos["avg_price"], "price": px,
                "value": mv,
                "pnl": (px - pos["avg_price"]) * pos["qty"],
                "pnl_pct": (px / pos["avg_price"] - 1) if pos["avg_price"] else 0.0,
                "weight": 0.0,
            }
        total = self.cash + pv
        for d in det.values():
            d["weight"] = d["value"] / total if total > 0 else 0.0
        pnl = total - self.total_deposits
        return {
            "total": total, "cash": self.cash, "positions_value": pv,
            "pnl": pnl,
            "pnl_pct": pnl / self.total_deposits if self.total_deposits else 0.0,
            "realized_pnl": self.realized_pnl,
            "positions": det,
        }

    def snapshot(self, prices: dict[str, float], date: str | None = None) -> None:
        v = self.value(prices)
        d = date or _date.today().isoformat()
        entry = {"date": d, "total": v["total"], "cash": v["cash"]}
        self.history = [h for h in self.history if h.get("date") != d]
        self.history.append(entry)
        self.history.sort(key=lambda h: h["date"])

    # ---- 리밸런스 ----
    def rebalance_to(self, target_weights: dict[str, float], prices: dict[str, float],
                     fee_pct: float = 0.0015, date: str | None = None) -> list[dict]:
        """목표 비중에 맞춰 매도 먼저, 매수 나중에 실행. 실행된 주문 리스트 반환."""
        executed: list[dict] = []
        total = self.value(prices)["total"]
        if total <= 0:
            return executed
        min_val = total * MIN_TRADE_FRAC
        tickers = set(target_weights) | set(self.positions)

        # 1) 매도
        for t in sorted(tickers):
            px = prices.get(t)
            if not px or px <= 0 or t not in self.positions:
                continue
            cur_qty = self.positions[t]["qty"]
            tgt_val = float(target_weights.get(t, 0.0)) * total
            excess = cur_qty * px - tgt_val
            if excess <= 0:
                continue
            qty = cur_qty if tgt_val <= 0 else min(cur_qty, int(math.ceil(excess / px - 1e-9)))
            # 완전 청산은 소액이어도 허용, 부분 매도는 1% 미만이면 건너뜀
            if qty <= 0 or (qty < cur_qty and qty * px < min_val):
                continue
            executed.append(self.sell(t, qty, px, fee_pct, date))

        # 2) 매수
        for t in sorted(tickers):
            px = prices.get(t)
            w = float(target_weights.get(t, 0.0))
            if not px or px <= 0 or w <= 0:
                continue
            cur_val = self.positions.get(t, {"qty": 0})["qty"] * px
            want = w * total - cur_val
            if want < min_val:
                continue
            want = min(want, self.cash)
            qty = int(want / (px * (1 + fee_pct)))
            if qty <= 0 or qty * px < min_val:
                continue
            executed.append(self.buy(t, qty, px, fee_pct, date))
        return executed
