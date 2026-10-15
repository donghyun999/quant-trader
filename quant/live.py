"""실전 자동매매 실행기.

흐름: 가격 데이터 → 전략 목표 비중 → 브로커 잔고 → 주문 계획(diff) → 리스크 가드 → 주문 전송 → 로그.

안전장치
- dry_run=True(기본)면 주문을 계산만 하고 보내지 않는다. 실제 전송은 execute=True 를 명시해야 한다.
- 실계좌(broker.real_money)면 config 의 리스크 한도를 반드시 거친다.
- 프로젝트 루트에 STOP_TRADING 파일이 있으면 어떤 주문도 나가지 않는다 (킬 스위치).
- 모든 실행은 live_log/YYYY-MM-DD.jsonl 에 기록된다.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .broker import Broker, PaperBroker, make_broker
from .broker.base import Balance, OrderResult
from .config import LiveConfig
from .risk import DailyState, PlannedOrder
from .signals import Signal, today_signals
from .strategies import get_strategy


@dataclass
class LiveReport:
    time: str
    broker: str
    dry_run: bool
    balance_before: dict
    signals: list[dict]
    planned: list[dict]
    rejected: list[str]
    results: list[dict]
    balance_after: dict | None = None
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_korean(self) -> str:
        lines = [f"[{self.time}] 브로커 {self.broker} / {'모의 실행(주문 전송 안 함)' if self.dry_run else '실제 주문 전송'}"]
        b = self.balance_before
        lines.append(f" 실행 전 총자산 {b['total']:,.0f} (현금 {b['cash']:,.0f})")
        for s in self.signals:
            lines.append(f"  신호 {s['ticker']}: {s['action']} 목표 {s['target_weight']:.0%} / 현재 {s['current_weight']:.0%}")
        if self.planned:
            lines.append(" 주문 계획:")
            for o in self.planned:
                lines.append(f"  {o['side']} {o['ticker']} {o['qty']}주 @ {o['price']:,.2f} (약 {o['qty'] * o['price']:,.0f})")
        else:
            lines.append(" 주문 계획: 없음 (목표 비중과 이미 비슷함)")
        for r in self.rejected:
            lines.append(f"  ✖ {r}")
        for r in self.results:
            mark = "✔" if r["ok"] else "✖"
            lines.append(f"  {mark} {r['side']} {r['ticker']} {r['qty']}주 → {r['message']} {('#' + r['order_id']) if r['order_id'] else ''}")
        if self.balance_after:
            lines.append(f" 실행 후 총자산 {self.balance_after['total']:,.0f} (현금 {self.balance_after['cash']:,.0f})")
        for n in self.notes:
            lines.append(f" ※ {n}")
        if self.error:
            lines.append(f" 오류: {self.error}")
        return "\n".join(lines)


def _balance_dict(b: Balance) -> dict:
    return {
        "total": b.total, "cash": b.cash, "currency": b.currency,
        "positions": {t: {"qty": p.qty, "avg_price": p.avg_price, "price": p.price, "value": p.value}
                      for t, p in b.positions.items()},
    }


def plan_orders(signals: list[Signal], balance: Balance, prices: dict[str, float],
                fee_pct: float, threshold: float = 0.02, min_frac: float = 0.01) -> list[PlannedOrder]:
    """목표 비중과 현재 잔고의 차이를 정수 주문으로 바꾼다. 매도 먼저, 매수 나중."""
    total = balance.total
    if total <= 0:
        return []
    sells: list[PlannedOrder] = []
    buys: list[PlannedOrder] = []
    cash = balance.cash
    for s in signals:
        px = prices.get(s.ticker)
        if not px or px <= 0:
            continue
        pos = balance.positions.get(s.ticker)
        cur_qty = pos.qty if pos else 0
        cur_val = cur_qty * px
        tgt_val = s.target_weight * total
        diff = tgt_val - cur_val
        if abs(diff) < max(threshold * total, min_frac * total):
            continue
        if diff < 0:
            qty = cur_qty if s.target_weight <= 0 else min(cur_qty, int(math.ceil(-diff / px - 1e-9)))
            if qty > 0:
                sells.append(PlannedOrder(s.ticker, "SELL", qty, px))
                cash += qty * px * (1 - fee_pct)
    for s in signals:
        px = prices.get(s.ticker)
        if not px or px <= 0 or s.target_weight <= 0:
            continue
        pos = balance.positions.get(s.ticker)
        cur_val = (pos.qty if pos else 0) * px
        diff = s.target_weight * total - cur_val
        if diff < max(threshold * total, min_frac * total):
            continue
        spend = min(diff, cash)
        qty = int(spend / (px * (1 + fee_pct)))
        if qty > 0:
            buys.append(PlannedOrder(s.ticker, "BUY", qty, px))
            cash -= qty * px * (1 + fee_pct)
    return sells + buys


class LiveTrader:
    def __init__(self, cfg: LiveConfig, broker: Broker | None = None, prices: dict[str, pd.DataFrame] | None = None):
        self.cfg = cfg
        self.broker = broker or self._make_broker(cfg)
        self._prices = prices
        self.log_dir = cfg.root / cfg.log_dir
        self.state = DailyState(self.log_dir / "live_state.json")

    @staticmethod
    def _make_broker(cfg: LiveConfig) -> Broker:
        if cfg.broker == "kis":
            k = cfg.kis
            return make_broker("kis", app_key=k.get("app_key", ""), app_secret=k.get("app_secret", ""),
                               account_no=k.get("account_no", ""), mode=k.get("mode", "vts"),
                               us_exchange=k.get("us_exchange"), token_cache=k.get("token_cache"))
        return make_broker("paper", fee_pct=cfg.fee_pct)

    def load_prices(self) -> dict[str, pd.DataFrame]:
        if self._prices is None:
            from .data import get_prices

            self._prices = get_prices(self.cfg.tickers, period=self.cfg.period)
        return self._prices

    def current_prices(self, hist: dict[str, pd.DataFrame]) -> dict[str, float]:
        """실계좌면 브로커 현재가, 아니면 마지막 종가."""
        last = {t: float(df["close"].iloc[-1]) for t, df in hist.items()}
        if isinstance(self.broker, PaperBroker):
            self.broker.set_prices(last)
            return last
        out = {}
        for t in hist:
            try:
                out[t] = self.broker.get_price(t)
            except Exception:  # noqa: BLE001
                out[t] = last[t]
        return out

    def run(self, execute: bool = False) -> LiveReport:
        """execute=False 면 dry-run. True 면 실제로 주문을 보낸다."""
        dry_run = not execute
        now = datetime.now().isoformat(timespec="seconds")
        rep = LiveReport(now, f"{self.broker.name}{':' + getattr(self.broker, 'mode', '') if getattr(self.broker, 'mode', '') else ''}",
                         dry_run, {"total": 0.0, "cash": 0.0}, [], [], [], [])
        try:
            self.broker.connect()
            hist = self.load_prices()
            prices = self.current_prices(hist)
            balance = self.broker.get_balance(self.cfg.tickers)
            rep.balance_before = _balance_dict(balance)
            strategy = get_strategy(self.cfg.strategy, **self.cfg.params)
            sigs = today_signals(strategy, hist, balance.weights(), threshold=self.cfg.threshold)
            rep.signals = [asdict(s) for s in sigs]
            planned = plan_orders(sigs, balance, prices, self.cfg.fee_pct, self.cfg.threshold)

            start_eq = self.state.day_start_equity(balance.total)
            passed, rejected = self.cfg.risk.check_orders(planned, balance, start_eq, self.cfg.root)
            rep.planned = [asdict(o) for o in passed]
            rep.rejected = rejected

            if not self.broker.is_market_open():
                rep.notes.append("장 운영 시간이 아닙니다. 주문은 다음 장에 접수/체결될 수 있습니다.")
            if dry_run:
                rep.notes.append("dry-run: 주문을 전송하지 않았습니다. 실제 전송은 --execute 옵션을 쓰세요.")
            else:
                for o in passed:
                    price = None if self.cfg.order_type == "market" else o.price
                    res: OrderResult = self.broker.submit_order(o.ticker, o.side, o.qty, price)
                    res.dry_run = False
                    rep.results.append(res.to_dict())
                try:
                    rep.balance_after = _balance_dict(self.broker.get_balance(self.cfg.tickers))
                except Exception as e:  # noqa: BLE001
                    rep.notes.append(f"실행 후 잔고 조회 실패: {e}")
        except Exception as e:  # noqa: BLE001
            rep.error = f"{type(e).__name__}: {e}"
        finally:
            self._log(rep)
        return rep

    def _log(self, rep: LiveReport) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{rep.time[:10]}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rep.to_dict(), ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass


def read_logs(log_dir: Path, limit: int = 20) -> list[dict]:
    rows: list[dict] = []
    if not log_dir.exists():
        return rows
    for p in sorted(log_dir.glob("*.jsonl"), reverse=True):
        for line in reversed(p.read_text(encoding="utf-8").splitlines()):
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
            if len(rows) >= limit:
                return rows
    return rows
