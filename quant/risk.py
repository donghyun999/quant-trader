"""실전 주문 전 리스크 가드.

모든 실전 주문은 RiskLimits.check_orders 를 통과해야 한다. 하나라도 위반하면 그 주문은 거부되고
(치명적 위반이면 전체 배치 거부), 이유가 한국어로 기록된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .broker.base import Balance


@dataclass
class PlannedOrder:
    ticker: str
    side: str          # BUY / SELL
    qty: int
    price: float       # 참고 가격

    @property
    def value(self) -> float:
        return self.qty * self.price


@dataclass
class RiskLimits:
    max_position_weight: float = 0.5      # 한 종목 최대 비중 (매수 후 기준)
    max_order_value: float = 0.0          # 1회 주문 최대 금액 (0 = 제한 없음)
    max_orders_per_run: int = 20          # 한 번 실행에서 낼 수 있는 최대 주문 수
    max_daily_loss_pct: float = 0.05      # 당일 시작 자산 대비 손실이 이 이상이면 신규 매수 금지
    allowed_tickers: list[str] = field(default_factory=list)  # 비어 있으면 전부 허용
    kill_switch_file: str = "STOP_TRADING"  # 이 파일이 프로젝트 루트에 있으면 모든 주문 중단
    min_order_value: float = 0.0          # 이 금액 미만 주문은 건너뜀 (잔돈 방지)

    def check_orders(self, orders: list[PlannedOrder], balance: Balance,
                     day_start_equity: float | None, root: Path) -> tuple[list[PlannedOrder], list[str]]:
        """통과한 주문과 거부 사유 목록을 반환."""
        reasons: list[str] = []
        if (root / self.kill_switch_file).exists():
            return [], [f"킬 스위치 파일({self.kill_switch_file})이 있어 모든 주문을 중단합니다."]
        total = balance.total
        if total <= 0:
            return [], ["계좌 총자산이 0 이하라 주문하지 않습니다."]

        loss_lock = False
        if day_start_equity and day_start_equity > 0:
            dd = total / day_start_equity - 1.0
            if dd <= -self.max_daily_loss_pct:
                loss_lock = True
                reasons.append(f"당일 손실 {dd:.2%} 가 한도(-{self.max_daily_loss_pct:.0%})를 넘어 신규 매수를 막습니다.")

        allowed = {t.upper() for t in self.allowed_tickers}
        passed: list[PlannedOrder] = []
        # 매도 먼저 평가해 누적 비중 계산을 단순화
        for o in sorted(orders, key=lambda x: 0 if x.side == "SELL" else 1):
            t = o.ticker.upper()
            if allowed and t not in allowed:
                reasons.append(f"{t}: 허용 종목 목록에 없어 거부.")
                continue
            if o.qty <= 0 or o.price <= 0:
                reasons.append(f"{t}: 수량/가격이 0 이라 거부.")
                continue
            if self.min_order_value and o.value < self.min_order_value:
                reasons.append(f"{t}: 주문 금액 {o.value:,.0f} 이 최소 {self.min_order_value:,.0f} 미만이라 건너뜀.")
                continue
            if self.max_order_value and o.value > self.max_order_value:
                reasons.append(f"{t}: 주문 금액 {o.value:,.0f} 이 1회 한도 {self.max_order_value:,.0f} 초과라 거부.")
                continue
            if o.side == "BUY":
                if loss_lock:
                    reasons.append(f"{t}: 당일 손실 한도 초과로 매수 거부.")
                    continue
                cur = balance.positions.get(t)
                cur_val = cur.value if cur else 0.0
                after_w = (cur_val + o.value) / total
                if after_w > self.max_position_weight + 1e-9:
                    reasons.append(f"{t}: 매수 후 비중 {after_w:.1%} 가 한도 {self.max_position_weight:.0%} 초과라 거부.")
                    continue
            else:
                cur = balance.positions.get(t)
                if not cur or cur.qty < o.qty:
                    reasons.append(f"{t}: 보유 수량({cur.qty if cur else 0})보다 많이 팔 수 없어 거부.")
                    continue
            passed.append(o)
        if len(passed) > self.max_orders_per_run:
            reasons.append(f"주문 수 {len(passed)} 가 1회 한도 {self.max_orders_per_run} 를 넘어 전체 배치를 거부합니다.")
            return [], reasons
        return passed, reasons


class DailyState:
    """당일 시작 자산 기록 (일일 손실 한도 계산용). live_state.json 에 저장."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.data = {}

    def day_start_equity(self, today_equity: float, today: str | None = None) -> float:
        d = today or date.today().isoformat()
        if self.data.get("date") != d:
            self.data = {"date": d, "start_equity": float(today_equity)}
            self.save()
        return float(self.data.get("start_equity", today_equity))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
