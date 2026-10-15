"""일별 루프 백테스트 엔진.

규칙
- 전략의 목표 비중(당일 종가까지의 정보)은 shift(1) 되어 **다음 거래일**에 체결된다.
- 체결가: execution="next_open" 이면 다음날 시가, "next_close" 이면 다음날 종가.
- 정수 주식 수, 수수료(fee_pct)와 슬리피지(slippage_pct) 반영, 현금 부족 시 가능한 만큼만 매수.
- 리밸런스 주기: daily=매일, weekly=주 첫 거래일, monthly=월 첫 거래일에만 목표 비중 변경을 반영.
  목표 비중이 직전 반영값과 달라진 종목만 매매한다 (가격 변동에 따른 비중 drift 는 그대로 둠).
- 손절(stop_loss_pct): 평균 진입가 대비 일중 저가가 손절가 이하이면 손절가(갭 하락 시 시가)에 전량 매도.
  손절된 종목은 전략의 목표 비중이 한 번 0 이 될 때까지 재진입하지 않는다.
- 벤치마크: 같은 종목 균등 비중 Buy&Hold 를 같은 엔진/같은 초기자금/같은 시작일로 계산.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .strategies import STRATEGIES, Strategy
from .strategies.base import common_index

TRADE_COLUMNS = ["date", "ticker", "side", "qty", "price", "value", "fee"]


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000_000   # 원 또는 달러, 단위 무관
    fee_pct: float = 0.0015            # 편도 수수료+세금 비율
    slippage_pct: float = 0.0005
    rebalance: str = "daily"           # "daily" | "weekly" | "monthly"
    stop_loss_pct: float | None = None  # 예: 0.1 = 진입가 대비 -10% 손절
    execution: str = "next_open"       # "next_open" | "next_close"


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    config: BacktestConfig
    strategy_name: str
    params: dict = field(default_factory=dict)


def _rebalance_mask(idx: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """해당 날짜가 리밸런스(비중 변경 반영) 가능일이면 True. 첫날은 항상 True."""
    n = len(idx)
    if freq == "daily":
        return np.ones(n, dtype=bool)
    if freq == "weekly":
        iso = idx.isocalendar()
        key = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy()).astype(np.int64)
    elif freq == "monthly":
        key = (idx.year * 100 + idx.month).to_numpy()
    else:
        raise ValueError(f"rebalance 는 daily/weekly/monthly 중 하나여야 합니다: {freq}")
    mask = np.ones(n, dtype=bool)
    mask[1:] = key[1:] != key[:-1]
    return mask


def _resolve_strategy(strategy: Any) -> Strategy:
    if isinstance(strategy, Strategy):
        return strategy
    if isinstance(strategy, str):
        return STRATEGIES[strategy]()
    if isinstance(strategy, type) and issubclass(strategy, Strategy):
        return strategy()
    raise TypeError("strategy 는 Strategy 인스턴스/클래스/이름이어야 합니다.")


def _clean_target(w: pd.DataFrame, idx: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """목표 비중을 공통 인덱스/종목에 맞추고 0~1, 행 합 <= 1 로 정리."""
    w = w.reindex(index=idx, columns=tickers).astype(float).fillna(0.0).clip(0.0, 1.0)
    s = w.sum(axis=1)
    over = s > 1.0 + 1e-9
    if over.any():
        w.loc[over] = w.loc[over].div(s[over], axis=0)
    return w


def _simulate(
    target: pd.DataFrame,
    px: dict[str, np.ndarray],
    sim_idx: pd.DatetimeIndex,
    tickers: list[str],
    cfg: BacktestConfig,
    use_stop: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """target: sim_idx 에 맞춰 이미 shift(1) 된 목표 비중 (해당 날짜에 체결할 비중)."""
    T, N = len(sim_idx), len(tickers)
    tgt = target.to_numpy(dtype=float)
    o, h, l, c = px["open"], px["high"], px["low"], px["close"]
    exec_on_open = cfg.execution == "next_open"
    exec_px = o if exec_on_open else c
    fee, slip = float(cfg.fee_pct), float(cfg.slippage_pct)
    sl = cfg.stop_loss_pct if (use_stop and cfg.stop_loss_pct) else None
    rebal = _rebalance_mask(sim_idx, cfg.rebalance)

    cash = float(cfg.initial_cash)
    qty = np.zeros(N, dtype=np.int64)
    entry = np.zeros(N)                # 평균 진입 체결가 (손절 기준)
    last_target = np.zeros(N)          # 마지막으로 반영한 목표 비중
    stopped = np.zeros(N, dtype=bool)  # 손절 후 재진입 대기
    last_close = np.full(N, np.nan)
    equity = np.zeros(T)
    hold_val = np.zeros((T, N))
    trades: list[dict] = []

    def record(t: int, i: int, side: str, q: int, price: float) -> float:
        value = q * price
        f = value * fee
        trades.append({"date": sim_idx[t], "ticker": tickers[i], "side": side,
                       "qty": int(q), "price": float(price), "value": float(value), "fee": float(f)})
        return f

    def stop_check(t: int) -> None:
        nonlocal cash
        if sl is None:
            return
        for i in range(N):
            if qty[i] <= 0 or entry[i] <= 0:
                continue
            lo = l[t, i] if not np.isnan(l[t, i]) else c[t, i]
            if np.isnan(lo):
                continue
            stop_px = entry[i] * (1.0 - sl)
            if lo <= stop_px:
                op = o[t, i]
                ref = op if (not np.isnan(op) and op < stop_px) else stop_px  # 갭 하락이면 시가
                price = ref * (1.0 - slip)
                q = int(qty[i])
                f = record(t, i, "SELL", q, price)
                cash += q * price - f
                qty[i] = 0
                entry[i] = 0.0
                stopped[i] = True
                last_target[i] = 0.0

    for t in range(T):
        valid_c = ~np.isnan(c[t])
        last_close[valid_c] = c[t, valid_c]
        raw = tgt[t]
        stopped[raw <= 0] = False
        eff = np.where(stopped, 0.0, raw)

        if not exec_on_open:
            stop_check(t)  # 장중 손절이 종가 체결보다 먼저

        if rebal[t]:
            p = exec_px[t]
            tradable = ~np.isnan(p) & (p > 0)
            changed = tradable & (np.abs(eff - last_target) > 1e-9)
            if changed.any():
                mark = np.where(tradable, p, last_close)
                mark = np.nan_to_num(mark, nan=0.0)
                E = cash + float(np.sum(qty * mark))
                desired = qty.copy()
                for i in np.flatnonzero(changed):
                    unit_cost = p[i] * (1.0 + slip) * (1.0 + fee)
                    desired[i] = int(np.floor(eff[i] * E / unit_cost + 1e-9))
                # 매도 먼저 (현금 확보)
                for i in np.flatnonzero(changed & (desired < qty)):
                    q = int(qty[i] - desired[i])
                    price = p[i] * (1.0 - slip)
                    f = record(t, i, "SELL", q, price)
                    cash += q * price - f
                    qty[i] -= q
                    if qty[i] == 0:
                        entry[i] = 0.0
                # 매수 (현금 부족 시 가능한 만큼)
                for i in np.flatnonzero(changed & (desired > qty)):
                    price = p[i] * (1.0 + slip)
                    q = int(desired[i] - qty[i])
                    q = min(q, int(np.floor(cash / (price * (1.0 + fee)) + 1e-9)))
                    if q <= 0:
                        continue
                    f = record(t, i, "BUY", q, price)
                    cash -= q * price + f
                    entry[i] = (entry[i] * qty[i] + price * q) / (qty[i] + q)
                    qty[i] += q
                last_target[changed] = eff[changed]

        if exec_on_open:
            stop_check(t)  # 시가 체결 후 장중 손절

        vals = qty * np.nan_to_num(last_close, nan=0.0)
        hold_val[t] = vals
        equity[t] = cash + vals.sum()

    eq = pd.Series(equity, index=sim_idx, name="equity")
    with np.errstate(divide="ignore", invalid="ignore"):
        wts = np.where(equity[:, None] > 0, hold_val / equity[:, None], 0.0)
    weights = pd.DataFrame(wts, index=sim_idx, columns=tickers)
    tr = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    if not tr.empty:
        tr["qty"] = tr["qty"].astype(np.int64)
    return eq, weights, tr


def run_backtest(
    strategy: Any,
    prices: dict[str, pd.DataFrame],
    config: BacktestConfig | None = None,
    start: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """전략 백테스트.

    start: 지정하면 그 날짜부터 시뮬레이션한다. 그 이전 데이터는 지표 계산(워밍업)에만 쓰인다
    (walk-forward 검증용).
    """
    cfg = config or BacktestConfig()
    if cfg.execution not in ("next_open", "next_close"):
        raise ValueError("execution 은 next_open 또는 next_close 여야 합니다.")
    strat = _resolve_strategy(strategy)
    if not prices:
        raise ValueError("가격 데이터가 비어 있습니다.")
    tickers = list(prices)
    idx = common_index(prices)
    if len(idx) < 2:
        raise ValueError("공통 기간 데이터가 너무 짧습니다 (최소 2거래일).")
    aligned = {t: prices[t].reindex(idx) for t in tickers}

    # 목표 비중 -> 다음 거래일 체결 (lookahead 방지)
    target = _clean_target(strat.generate_weights(aligned), idx, tickers).shift(1).fillna(0.0)
    bench_target = pd.DataFrame(1.0 / len(tickers), index=idx, columns=tickers).shift(1).fillna(0.0)

    if start is not None:
        sim_mask = idx >= pd.Timestamp(start)
        if not sim_mask.any():
            raise ValueError("시작일 이후 데이터가 없습니다.")
    else:
        sim_mask = np.ones(len(idx), dtype=bool)
    sim_idx = idx[sim_mask]

    px = {
        k: np.column_stack([aligned[t][k].to_numpy(dtype=float) for t in tickers])[sim_mask]
        for k in ("open", "high", "low", "close")
    }
    # 시가가 없으면 종가로 대체
    px["open"] = np.where(np.isnan(px["open"]), px["close"], px["open"])

    equity, weights, trades = _simulate(target.loc[sim_idx], px, sim_idx, tickers, cfg, use_stop=True)
    # 벤치마크는 리밸런스 주기와 무관하게 첫 체결일에 사서 계속 보유
    bench_cfg = BacktestConfig(**{**cfg.__dict__, "rebalance": "daily", "stop_loss_pct": None})
    bench, _, _ = _simulate(bench_target.loc[sim_idx], px, sim_idx, tickers, bench_cfg, use_stop=False)
    bench.name = "benchmark"

    metrics = compute_metrics(equity, trades, benchmark=bench, weights=weights)
    return BacktestResult(
        equity=equity,
        benchmark_equity=bench,
        weights=weights,
        trades=trades,
        metrics=metrics,
        config=cfg,
        strategy_name=strat.name,
        params=dict(strat.p),
    )
