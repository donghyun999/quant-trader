from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import indicators as ind
from quant.backtest import BacktestConfig, BacktestResult, run_backtest
from quant.data import _clean
from quant.metrics import compute_metrics, describe_metrics_korean
from quant.optimize import grid_search, walk_forward
from quant.strategies import STRATEGIES, get_strategy
from quant.strategies.sma_cross import SmaCross

METRIC_KEYS = [
    "total_return", "cagr", "annual_vol", "sharpe", "sortino", "max_drawdown",
    "max_drawdown_duration_days", "calmar", "win_rate", "num_trades", "profit_factor",
    "avg_trade_return", "exposure", "best_month", "worst_month", "final_equity",
    "benchmark_total_return", "benchmark_cagr", "benchmark_max_drawdown",
]
TRADE_COLS = ["date", "ticker", "side", "qty", "price", "value", "fee"]


# ---------------------------------------------------------------- 지표
def test_indicators_sanity(make_fake_prices):
    df = make_fake_prices(n_days=300, tickers=("AAA",))["AAA"]
    c = df["close"]
    s = ind.sma(c, 10)
    assert s.iloc[:9].isna().all()
    assert s.iloc[9] == pytest.approx(c.iloc[:10].mean())
    e = ind.ema(c, 10)
    assert e.iloc[9:].notna().all()
    r = ind.rsi(c, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    # 계속 오르기만 하면 RSI=100
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    a = ind.atr(df["high"], df["low"], c, 14).dropna()
    assert (a > 0).all()
    mid, upper, lower = ind.bollinger(c, 20, 2)
    ok = mid.notna()
    assert (upper[ok] >= mid[ok]).all() and (lower[ok] <= mid[ok]).all()
    m = ind.momentum(c, 5)
    assert m.iloc[5] == pytest.approx(c.iloc[5] / c.iloc[0] - 1)


def test_clean_handles_yfinance_multiindex():
    idx = pd.date_range("2024-01-01", periods=3, tz="America/New_York")
    cols = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], ["SPY"]], names=["Price", "Ticker"]
    )
    raw = pd.DataFrame(np.arange(15, dtype=float).reshape(3, 5), index=idx, columns=cols)
    df = _clean(raw, "SPY")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is None


# ---------------------------------------------------------------- 전략
@pytest.mark.parametrize("name", list(STRATEGIES))
@pytest.mark.parametrize("tickers", [("AAA",), ("AAA", "BBB", "CCC")])
def test_strategy_weights_valid(make_fake_prices, name, tickers):
    prices = make_fake_prices(n_days=500, tickers=tickers, seed=1)
    strat = get_strategy(name)
    assert strat.label and strat.description
    for p in strat.params:
        assert p.help
    w = strat.generate_weights(prices)
    assert list(w.columns) == list(tickers)
    assert len(w) == 500
    assert not w.isna().any().any()
    assert (w.to_numpy() >= 0).all() and (w.to_numpy() <= 1 + 1e-12).all()
    assert (w.sum(axis=1) <= 1 + 1e-9).all()


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategy_weights_causal(make_fake_prices, name):
    """잘린 데이터로 계산한 비중 == 전체 데이터로 계산한 비중의 앞부분 (미래 정보 미사용)."""
    prices = make_fake_prices(n_days=400, tickers=("AAA", "BBB"), seed=2)
    strat = get_strategy(name)
    full = strat.generate_weights(prices)
    cut = full.index[300]
    part = strat.generate_weights({t: df.loc[:cut] for t, df in prices.items()})
    pd.testing.assert_frame_equal(part, full.loc[:cut])


def test_strategies_actually_trade(make_fake_prices):
    prices = make_fake_prices(n_days=600, tickers=("AAA", "BBB"), seed=3)
    for name in ["sma_cross", "rsi_reversion", "bollinger", "dual_momentum"]:
        w = get_strategy(name).generate_weights(prices)
        assert w.to_numpy().max() > 0, name


def test_invalid_params():
    with pytest.raises(ValueError):
        get_strategy("sma_cross", fast=50, slow=20)
    with pytest.raises(ValueError):
        get_strategy("sma_cross", nope=1)
    assert get_strategy("sma_cross", fast=10.0).p["fast"] == 10


def test_dual_momentum_constant_within_month(make_fake_prices):
    prices = make_fake_prices(n_days=500, tickers=("AAA", "BBB", "CCC"), seed=4)
    w = get_strategy("dual_momentum", lookback=60, top_n=1).generate_weights(prices)
    # 비중은 월말(그 달의 마지막 거래일)에만 바뀐다
    changes = (w.diff().abs().sum(axis=1) > 0).to_numpy()
    periods = w.index.to_period("M")
    for i in np.flatnonzero(changes):
        assert i == len(w) - 1 or periods[i] != periods[i + 1]
    assert changes.any()


# ---------------------------------------------------------------- 백테스트
@pytest.mark.parametrize("execution", ["next_open", "next_close"])
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_backtest_no_lookahead(make_fake_prices, name, execution):
    prices = make_fake_prices(n_days=400, tickers=("AAA", "BBB"), seed=5)
    cfg = BacktestConfig(execution=execution, stop_loss_pct=0.08)
    r1 = run_backtest(get_strategy(name), prices, cfg)
    last = r1.equity.index[-1]
    bumped = {}
    for t, df in prices.items():
        df = df.copy()
        df.loc[last, "close"] *= 1.5
        df.loc[last, "high"] = max(df.loc[last, "high"], df.loc[last, "close"])
        bumped[t] = df
    r2 = run_backtest(get_strategy(name), bumped, cfg)
    t1 = r1.trades[r1.trades["date"] < last].reset_index(drop=True)
    t2 = r2.trades[r2.trades["date"] < last].reset_index(drop=True)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_series_equal(r1.equity.iloc[:-1], r2.equity.iloc[:-1])


def test_backtest_result_shape(make_fake_prices):
    prices = make_fake_prices(n_days=400, tickers=("AAA",), seed=6)
    res = run_backtest(get_strategy("sma_cross", fast=10, slow=30), prices)
    assert isinstance(res, BacktestResult)
    assert list(res.trades.columns) == TRADE_COLS
    assert len(res.trades) > 0
    assert set(res.trades["side"]) <= {"BUY", "SELL"}
    assert (res.trades["qty"] > 0).all()
    assert res.trades["qty"].dtype.kind == "i"
    assert len(res.equity) == len(res.benchmark_equity) == 400
    assert res.equity.iloc[0] == pytest.approx(res.config.initial_cash)
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    assert res.strategy_name == "sma_cross" and res.params == {"fast": 10, "slow": 30}
    # 첫 거래는 최초 신호 다음날 이후
    w = get_strategy("sma_cross", fast=10, slow=30).generate_weights(prices)["AAA"]
    first_sig = w[w > 0].index[0]
    assert res.trades["date"].iloc[0] > first_sig


def test_buy_hold_equals_benchmark(make_fake_prices):
    prices = make_fake_prices(n_days=500, tickers=("AAA", "BBB", "CCC"), seed=7)
    res = run_backtest(get_strategy("buy_hold"), prices)
    np.testing.assert_allclose(res.equity.to_numpy(), res.benchmark_equity.to_numpy(), rtol=1e-9)
    assert (res.trades["side"] == "BUY").all() and len(res.trades) == 3


def test_fees_reduce_equity(make_fake_prices):
    prices = make_fake_prices(n_days=400, tickers=("AAA",), seed=8)
    s = get_strategy("sma_cross", fast=5, slow=20)
    free = run_backtest(s, prices, BacktestConfig(fee_pct=0, slippage_pct=0))
    costly = run_backtest(s, prices, BacktestConfig(fee_pct=0.01, slippage_pct=0.005))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert (free.trades["fee"] == 0).all()


def test_rebalance_frequency(make_fake_prices):
    prices = make_fake_prices(n_days=500, tickers=("AAA", "BBB"), seed=9)
    s = get_strategy("sma_cross", fast=5, slow=20)
    daily = run_backtest(s, prices, BacktestConfig(rebalance="daily"))
    monthly = run_backtest(s, prices, BacktestConfig(rebalance="monthly"))
    weekly = run_backtest(s, prices, BacktestConfig(rebalance="weekly"))
    md = pd.DatetimeIndex(monthly.trades["date"])
    idx = monthly.equity.index
    first_of_month = idx[np.r_[True, idx.month[1:] != idx.month[:-1]]]
    assert set(md) <= set(first_of_month)
    assert len(monthly.trades) < len(weekly.trades) <= len(daily.trades)
    wd = pd.DatetimeIndex(weekly.trades["date"])
    iso = idx.isocalendar()
    key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    first_of_week = idx[np.r_[True, key[1:] != key[:-1]]]
    assert set(wd) <= set(first_of_week)


def test_stop_loss_triggers(make_fake_prices):
    prices = make_fake_prices(n_days=400, tickers=("AAA",), seed=10)
    df = prices["AAA"].copy()
    # 200일째부터 폭락시킨다
    crash = df.index[200:]
    factor = np.linspace(1.0, 0.5, len(crash))
    for col in ["open", "high", "low", "close"]:
        df.loc[crash, col] = df.loc[crash, col] * factor
    prices = {"AAA": df}
    res = run_backtest(get_strategy("buy_hold"), prices, BacktestConfig(stop_loss_pct=0.05))
    sells = res.trades[res.trades["side"] == "SELL"]
    assert len(sells) == 1  # 손절 후 재진입 없음 (목표 비중이 계속 양수)
    entry = res.trades.iloc[0]["price"]
    assert sells.iloc[0]["price"] <= entry * 0.95 + 1e-9


def test_multi_ticker_misaligned_dates(make_fake_prices):
    prices = make_fake_prices(n_days=300, tickers=("AAA", "BBB"), seed=11)
    prices["BBB"] = prices["BBB"].iloc[50:]
    res = run_backtest(get_strategy("sma_cross", fast=5, slow=20), prices)
    assert len(res.equity) == 250


# ---------------------------------------------------------------- 지표
def test_metrics_keys(make_fake_prices):
    prices = make_fake_prices(n_days=400, tickers=("AAA", "BBB"), seed=12)
    res = run_backtest(get_strategy("rsi_reversion"), prices)
    for k in METRIC_KEYS:
        assert k in res.metrics, k
    assert res.metrics["max_drawdown"] <= 0
    assert 0 <= res.metrics["exposure"] <= 1
    lines = describe_metrics_korean(res.metrics)
    assert isinstance(lines, list) and all(isinstance(s, str) for s in lines) and len(lines) >= 3


def test_metrics_known_values():
    idx = pd.bdate_range("2021-01-01", periods=253)
    eq = pd.Series(100.0 * 1.10 ** (np.arange(253) / 252), index=idx)
    m = compute_metrics(eq, pd.DataFrame(columns=TRADE_COLS))
    assert m["total_return"] == pytest.approx(0.10, rel=1e-6)
    assert m["cagr"] == pytest.approx(0.10, rel=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert m["num_trades"] == 0
    assert "benchmark_total_return" not in m


# ---------------------------------------------------------------- 최적화
def test_grid_search_sorted(make_fake_prices):
    prices = make_fake_prices(n_days=500, tickers=("AAA", "BBB"), seed=13)
    grid = {"fast": [5, 10, 20], "slow": [10, 30, 60]}
    df = grid_search(SmaCross, prices, grid, BacktestConfig(), metric="sharpe")
    assert len(df) == 7  # fast < slow 인 조합만 유효
    assert {"fast", "slow", "sharpe", "total_return"} <= set(df.columns)
    vals = df["sharpe"].to_numpy()
    assert (np.diff(vals) <= 1e-12).all()


def test_walk_forward(make_fake_prices):
    prices = make_fake_prices(n_days=600, tickers=("AAA", "BBB"), seed=14)
    grid = {"fast": [5, 10], "slow": [20, 40]}
    out = walk_forward(SmaCross, prices, grid, BacktestConfig(), n_splits=3)
    assert set(out) >= {"folds", "oos_equity", "oos_metrics", "warning"}
    assert len(out["folds"]) == 3
    for f in out["folds"]:
        assert f["train_end"] < f["test_start"] <= f["test_end"]
        assert set(f["best_params"]) == {"fast", "slow"}
        assert all(isinstance(v, int) for v in f["best_params"].values())
        assert "train_metric" in f and "test_metric" in f
    eq = out["oos_equity"]
    assert eq.index.is_monotonic_increasing and not eq.index.has_duplicates
    assert eq.iloc[0] == pytest.approx(10_000_000, rel=0.01)
    assert isinstance(out["warning"], str) and out["warning"]
    assert "sharpe" in out["oos_metrics"]
