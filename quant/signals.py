"""오늘의 매매 신호 생성."""
from __future__ import annotations

from dataclasses import dataclass

ACTION_KO = {"BUY": "매수", "SELL": "매도", "HOLD": "보유", "NONE": "관망"}


@dataclass
class Signal:
    ticker: str
    action: str  # "BUY" | "SELL" | "HOLD" | "NONE"
    target_weight: float
    current_weight: float
    price: float
    reason: str


def _reason(action: str, tw: float, cw: float) -> str:
    base = f"목표 비중 {tw:.0%} vs 현재 {cw:.0%}"
    if action == "BUY":
        return f"{base} → 매수" + (" (신규 진입 구간입니다)" if cw <= 0 else " (비중을 늘립니다)")
    if action == "SELL":
        return f"{base} → 매도" + (" (보유 종료 구간입니다)" if tw <= 0 else " (비중을 줄입니다)")
    if action == "HOLD":
        return f"{base} → 현재 비중을 유지합니다"
    return f"{base} → 투자하지 않는 구간입니다"


def today_signals(strategy, prices, current_weights: dict[str, float] | None = None,
                  threshold: float = 0.02) -> list[Signal]:
    """전략의 마지막 행 목표 비중과 현재 비중의 차이로 액션을 결정한다."""
    current_weights = current_weights or {}
    w = strategy.generate_weights(prices)
    last = w.iloc[-1] if len(w) else None
    out: list[Signal] = []
    for t, df in prices.items():
        tw = 0.0
        if last is not None and t in last.index and last[t] == last[t]:
            tw = float(last[t])
        cw = float(current_weights.get(t, 0.0))
        diff = tw - cw
        if diff > threshold:
            action = "BUY"
        elif diff < -threshold:
            action = "SELL"
        elif tw > 0 or cw > 0:
            action = "HOLD"
        else:
            action = "NONE"
        px = float(df["close"].iloc[-1]) if len(df) else float("nan")
        out.append(Signal(t, action, tw, cw, px, _reason(action, tw, cw)))
    return out


def format_signals_korean(signals: list[Signal]) -> str:
    if not signals:
        return "신호가 없습니다."
    lines = []
    for s in signals:
        lines.append(
            f"[{ACTION_KO.get(s.action, s.action)}] {s.ticker}  종가 {s.price:,.2f}  "
            f"목표 {s.target_weight:.0%} / 현재 {s.current_weight:.0%}\n    {s.reason}"
        )
    return "\n".join(lines)
