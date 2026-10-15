"""성과 지표 계산과 쉬운 한국어 설명.

연율화는 252 거래일 기준. 무위험수익률(rf)=0 으로 샤프 지수 계산.
max_drawdown 은 음수(예: -0.18 = -18%)로 표시한다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252
NAN = float("nan")


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _max_dd_duration_days(equity: pd.Series) -> int:
    """고점 대비 하락 상태가 가장 오래 지속된 기간 (달력일). 회복 못 했으면 마지막 날까지."""
    if len(equity) < 2:
        return 0
    peak = equity.cummax()
    under = (equity < peak).to_numpy()
    dates = equity.index
    longest = 0
    start = None
    for i, u in enumerate(under):
        if u and start is None:
            start = dates[i - 1] if i > 0 else dates[i]  # 직전 고점일
        if not u and start is not None:
            longest = max(longest, (dates[i] - start).days)
            start = None
    if start is not None:
        longest = max(longest, (dates[-1] - start).days)
    return int(longest)


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return NAN
    n = len(equity) - 1
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return -1.0
    return float(ratio ** (TRADING_DAYS / n) - 1.0)


def _closed_trade_returns(trades: pd.DataFrame | None) -> tuple[list[float], list[float]]:
    """매도(청산)마다 (수익률, 손익) 계산. 평균단가(매수 수수료 포함) 기준."""
    rets: list[float] = []
    pnls: list[float] = []
    if trades is None or len(trades) == 0:
        return rets, pnls
    shares: dict[str, float] = {}
    basis: dict[str, float] = {}
    for row in trades.itertuples(index=False):
        tk = row.ticker
        if row.side == "BUY":
            shares[tk] = shares.get(tk, 0.0) + row.qty
            basis[tk] = basis.get(tk, 0.0) + row.value + row.fee
        elif row.side == "SELL":
            sh = shares.get(tk, 0.0)
            if sh <= 0 or row.qty <= 0:
                continue
            avg = basis[tk] / sh
            cost = avg * row.qty
            proceeds = row.value - row.fee
            pnls.append(proceeds - cost)
            rets.append(proceeds / cost - 1.0 if cost > 0 else 0.0)
            shares[tk] = sh - row.qty
            basis[tk] = basis[tk] - cost
    return rets, pnls


def _exposure_from_trades(equity: pd.Series, trades: pd.DataFrame | None) -> float:
    if trades is None or len(trades) == 0 or len(equity) == 0:
        return 0.0
    signed = trades["qty"].where(trades["side"] == "BUY", -trades["qty"]).astype(float)
    pos = (
        pd.DataFrame({"date": pd.to_datetime(trades["date"]), "ticker": trades["ticker"], "q": signed})
        .pivot_table(index="date", columns="ticker", values="q", aggfunc="sum")
        .cumsum()
        .reindex(equity.index)
        .ffill()
        .fillna(0.0)
    )
    return float((pos.gt(0).any(axis=1)).mean())


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
) -> dict:
    """자산 곡선과 거래 내역으로 성과 지표를 계산한다.

    num_trades: 전체 체결 건수(매수+매도). win_rate/profit_factor/avg_trade_return: 매도(청산) 기준.
    exposure: 주식을 보유한 날의 비율 (0~1).
    """
    equity = equity.dropna().astype(float)
    m: dict = {}
    if len(equity) == 0:
        raise ValueError("자산 곡선이 비어 있습니다.")
    rets = equity.pct_change().dropna()

    m["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] > 0 else NAN
    m["cagr"] = _cagr(equity)
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else NAN
    m["annual_vol"] = vol * math.sqrt(TRADING_DAYS) if vol == vol else NAN
    mean = float(rets.mean()) if len(rets) else NAN
    m["sharpe"] = mean / vol * math.sqrt(TRADING_DAYS) if vol and vol == vol and vol > 0 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0.0) ** 2))) if len(rets) else 0.0
    m["sortino"] = mean / downside * math.sqrt(TRADING_DAYS) if downside > 0 else 0.0
    dd = _drawdown(equity)
    m["max_drawdown"] = float(dd.min())
    m["max_drawdown_duration_days"] = _max_dd_duration_days(equity)
    m["calmar"] = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 and m["cagr"] == m["cagr"] else 0.0

    trade_rets, pnls = _closed_trade_returns(trades)
    m["num_trades"] = int(len(trades)) if trades is not None else 0
    if trade_rets:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        m["win_rate"] = len(wins) / len(pnls)
        gross_loss = -sum(losses)
        m["profit_factor"] = (sum(wins) / gross_loss) if gross_loss > 0 else (float("inf") if wins else NAN)
        m["avg_trade_return"] = float(np.mean(trade_rets))
    else:
        m["win_rate"] = NAN
        m["profit_factor"] = NAN
        m["avg_trade_return"] = NAN

    if weights is not None and len(weights):
        m["exposure"] = float((weights.reindex(equity.index).fillna(0.0).sum(axis=1) > 1e-9).mean())
    else:
        m["exposure"] = _exposure_from_trades(equity, trades)

    # 월별 수익률 (첫 달은 시작 자산 대비)
    month_end = equity.groupby(equity.index.to_period("M")).last().dropna()  # resample 대신 (numpy 2.5 경고 회피)
    monthly = pd.concat([pd.Series([equity.iloc[0]]), month_end.reset_index(drop=True)]).pct_change().dropna()
    m["best_month"] = float(monthly.max()) if len(monthly) else NAN
    m["worst_month"] = float(monthly.min()) if len(monthly) else NAN
    m["final_equity"] = float(equity.iloc[-1])

    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.dropna().astype(float)
        m["benchmark_total_return"] = float(b.iloc[-1] / b.iloc[0] - 1.0)
        m["benchmark_cagr"] = _cagr(b)
        m["benchmark_max_drawdown"] = float(_drawdown(b).min())
    return m


def _pct(x: float, digits: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def _ok(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def describe_metrics_korean(metrics: dict) -> list[str]:
    """지표를 초보자도 이해할 수 있는 쉬운 문장 리스트로 바꾼다."""
    m = metrics
    out: list[str] = []
    tr, cagr, mdd = m.get("total_return"), m.get("cagr"), m.get("max_drawdown")
    if _ok(tr) and _ok(cagr):
        out.append(
            f"전체 기간 동안 {_pct(tr, sign=True)} 수익이 났고, 1년 평균으로 환산하면 연 {_pct(cagr)} 입니다."
        )
    if _ok(mdd):
        dur = m.get("max_drawdown_duration_days", 0)
        out.append(
            f"최악의 경우 고점 대비 {_pct(mdd)}까지 떨어진 적이 있습니다"
            f" (하락 후 회복까지 가장 길게 걸린 기간은 약 {int(dur)}일)."
            " 이 정도 하락을 견딜 수 있는지 꼭 생각해 보세요."
        )
    sharpe = m.get("sharpe")
    if _ok(sharpe):
        if sharpe >= 1.0:
            grade = "위험 대비 수익이 좋은 편입니다"
        elif sharpe >= 0.5:
            grade = "위험 대비 수익이 보통 수준입니다"
        elif sharpe > 0:
            grade = "감수한 위험에 비해 수익이 작은 편입니다"
        else:
            grade = "위험을 감수했지만 수익을 내지 못했습니다"
        out.append(f"샤프 지수는 {sharpe:.2f}로, {grade} (1 이상이면 우수, 0.5 미만이면 아쉬움).")
    vol = m.get("annual_vol")
    if _ok(vol):
        out.append(f"1년 동안 자산이 평균적으로 위아래 {_pct(vol)} 정도 출렁이는 수준입니다(연 변동성).")
    n = m.get("num_trades", 0)
    wr = m.get("win_rate")
    if n:
        s = f"총 {int(n)}번 사고팔았습니다"
        if _ok(wr):
            s += f". 매도한 거래 중 이익을 본 비율(승률)은 {_pct(wr, 0)} 입니다"
        avg = m.get("avg_trade_return")
        if _ok(avg):
            s += f", 거래당 평균 수익률은 {_pct(avg, sign=True)} 입니다"
        out.append(s + ".")
    else:
        out.append("기간 중 매매가 한 번도 없었습니다.")
    exp = m.get("exposure")
    if _ok(exp):
        out.append(f"전체 기간 중 {_pct(exp, 0)}의 날에 주식을 보유했고, 나머지는 현금으로 쉬었습니다.")
    bm, wm = m.get("best_month"), m.get("worst_month")
    if _ok(bm) and _ok(wm):
        out.append(f"가장 좋았던 달은 {_pct(bm, sign=True)}, 가장 나빴던 달은 {_pct(wm, sign=True)} 였습니다.")
    btr = m.get("benchmark_total_return")
    if _ok(btr) and _ok(tr):
        diff = tr - btr
        if diff >= 0:
            s = f"같은 종목을 그냥 사서 보유했을 때({_pct(btr, sign=True)})보다 {_pct(diff)}p 더 벌었습니다."
        else:
            s = f"같은 종목을 그냥 사서 보유했을 때({_pct(btr, sign=True)})보다 {_pct(-diff)}p 덜 벌었습니다."
        bmdd = m.get("benchmark_max_drawdown")
        if _ok(bmdd) and _ok(mdd):
            if mdd > bmdd:
                s += f" 대신 최대 하락폭은 단순 보유({_pct(bmdd)})보다 작았습니다."
            else:
                s += f" 최대 하락폭은 단순 보유({_pct(bmdd)})보다 컸습니다."
        out.append(s)
    out.append("※ 과거 성과가 미래 수익을 보장하지 않으며, 이 결과는 투자 조언이 아닙니다.")
    return out
