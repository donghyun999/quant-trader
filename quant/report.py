"""월간 성과 리포트: 페이퍼 계좌 history 와 실전 로그(live_log)를 월별로 집계한다."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .live import read_logs
from .metrics import compute_metrics
from .paper import PaperAccount


def monthly_table(history: list[dict]) -> pd.DataFrame:
    """[{date,total,cash}] → 월별 수익률/월말자산/월중 최대낙폭."""
    if not history:
        return pd.DataFrame(columns=["month", "end_equity", "return", "max_drawdown"])
    h = pd.DataFrame(history)
    h["date"] = pd.to_datetime(h["date"])
    s = h.set_index("date")["total"].sort_index()
    rows = []
    prev = None
    for period, g in s.groupby(s.index.to_period("M")):
        start = prev if prev is not None else g.iloc[0]
        path = pd.concat([pd.Series([start]), g.reset_index(drop=True)])  # 전월 말 자산을 고점 후보에 포함
        dd = (path / path.cummax() - 1).min()
        rows.append({"month": str(period), "end_equity": float(g.iloc[-1]),
                     "return": float(g.iloc[-1] / start - 1) if start else float("nan"),
                     "max_drawdown": float(dd)})
        prev = g.iloc[-1]
    return pd.DataFrame(rows)


def orders_by_month(orders: list[dict]) -> pd.DataFrame:
    if not orders:
        return pd.DataFrame(columns=["month", "orders", "buy_value", "sell_value", "fees"])
    o = pd.DataFrame(orders)
    o["month"] = pd.to_datetime(o["date"]).dt.to_period("M").astype(str)
    o["buy_value"] = o["value"].where(o["side"] == "BUY", 0.0)
    o["sell_value"] = o["value"].where(o["side"] == "SELL", 0.0)
    g = o.groupby("month")
    return pd.DataFrame({
        "orders": g.size(),
        "buy_value": g["buy_value"].sum(),
        "sell_value": g["sell_value"].sum(),
        "fees": g["fee"].sum(),
    }).reset_index()


def live_log_summary(log_dir: Path, limit: int = 500) -> pd.DataFrame:
    logs = read_logs(log_dir, limit)
    if not logs:
        return pd.DataFrame(columns=["month", "runs", "executed_runs", "orders_ok", "orders_failed", "rejected", "errors"])
    rows = []
    for r in logs:
        res = r.get("results", [])
        rows.append({"month": r["time"][:7], "runs": 1, "executed_runs": 0 if r["dry_run"] else 1,
                     "orders_ok": sum(1 for x in res if x.get("ok")), "orders_failed": sum(1 for x in res if not x.get("ok")),
                     "rejected": len(r.get("rejected", [])), "errors": 1 if r.get("error") else 0})
    return pd.DataFrame(rows).groupby("month").sum().reset_index()


def build_report(account: PaperAccount, log_dir: Path | None = None) -> dict:
    out = {"monthly": monthly_table(account.history), "orders": orders_by_month(account.orders)}
    if account.history:
        s = pd.Series({pd.Timestamp(h["date"]): h["total"] for h in account.history}).sort_index()
        if len(s) > 2:
            out["metrics"] = compute_metrics(s)
    if log_dir is not None:
        out["live"] = live_log_summary(log_dir)
    return out


def format_report_korean(rep: dict) -> str:
    lines = ["=== 월간 성과 리포트 ==="]
    m = rep.get("monthly")
    if m is None or m.empty:
        lines.append("기록된 자산 이력이 없습니다. paper run 또는 live run 을 실행하면 쌓입니다.")
    else:
        lines.append(f"{'월':<9}{'월말 자산':>16}{'수익률':>9}{'월중 낙폭':>10}")
        for r in m.itertuples(index=False):
            lines.append(f"{r.month:<9}{r.end_equity:>16,.0f}{r._2:>9.1%}{r.max_drawdown:>10.1%}")
    met = rep.get("metrics")
    if met:
        lines.append(f"누적 수익률 {met['total_return']:.1%} / 최대 낙폭 {met['max_drawdown']:.1%} / 샤프 {met['sharpe']:.2f}")
    o = rep.get("orders")
    if o is not None and not o.empty:
        lines.append("")
        lines.append(f"{'월':<9}{'주문 수':>7}{'매수 금액':>16}{'매도 금액':>16}{'수수료':>12}")
        for r in o.itertuples(index=False):
            lines.append(f"{r.month:<9}{r.orders:>7}{r.buy_value:>16,.0f}{r.sell_value:>16,.0f}{r.fees:>12,.0f}")
    lv = rep.get("live")
    if lv is not None and not lv.empty:
        lines.append("")
        lines.append("실전 실행 로그: " + "; ".join(
            f"{r.month} 실행 {r.runs}회(전송 {r.executed_runs}) 주문성공 {r.orders_ok} 실패 {r.orders_failed} 거부 {r.rejected} 오류 {r.errors}"
            for r in lv.itertuples(index=False)))
    return "\n".join(lines)
