"""실전 실행기 재현(replay) 회귀 테스트: 합성 데이터로 60일 전진 실행 → 백테스트와 일치해야 한다."""
from __future__ import annotations

import pandas as pd

from quant.backtest import BacktestConfig, run_backtest
from quant.broker import PaperBroker
from quant.config import LiveConfig
from quant.live import LiveTrader
from quant.paper import PaperAccount
from quant.risk import RiskLimits
from quant.strategies import get_strategy


def test_live_replay_matches_backtest(make_fake_prices, tmp_path, monkeypatch):
    monkeypatch.setattr(LiveConfig, "root", property(lambda self: tmp_path))
    prices = make_fake_prices(n_days=420, tickers=("AAA", "BBB", "CCC"), seed=3)
    idx = prices["AAA"].index
    start = idx[-60]
    fee, thr = 0.0005, 0.02
    acc = PaperAccount(str(tmp_path / "p.json"), 10_000_000)
    cfg = LiveConfig(broker="paper", tickers=list(prices), strategy="ensemble",
                     log_dir=str(tmp_path / "log"), fee_pct=fee, threshold=thr)
    cfg.risk = RiskLimits(max_position_weight=1.0)
    eq = {}
    for d in idx[idx >= start]:
        hist = {t: df.loc[:d] for t, df in prices.items()}
        LiveTrader(cfg, broker=PaperBroker(acc, fee_pct=fee), prices=hist).run(execute=True)
        eq[d] = acc.value({t: float(df.loc[d, "close"]) for t, df in prices.items()})["total"]
    live_eq = pd.Series(eq)
    bt = run_backtest(get_strategy("ensemble"), prices,
                      BacktestConfig(fee_pct=fee, slippage_pct=0.0, execution="next_close",
                                     min_trade_weight=thr), start=start)
    gap = live_eq.iloc[-1] / bt.equity.iloc[-1] - 1
    corr = live_eq.pct_change().dropna().corr(bt.equity.pct_change().dropna())
    assert abs(gap) < 0.02, gap
    assert corr > 0.95, corr
