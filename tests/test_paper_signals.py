import numpy as np
import pandas as pd

from quant.paper import PaperAccount
from quant.signals import Signal, today_signals, format_signals_korean


def make_prices(n=50, tickers=("AAA", "BBB")):
    idx = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        c = pd.Series(100.0 + np.arange(n) * (i + 1), index=idx)
        out[t] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000}, index=idx)
    return out


class FixedStrategy:
    def __init__(self, w):
        self.w = w

    def generate_weights(self, prices):
        idx = next(iter(prices.values())).index
        return pd.DataFrame({t: [self.w.get(t, 0.0)] * len(idx) for t in prices}, index=idx)


def test_buy_sell_and_pnl(tmp_path):
    acc = PaperAccount(str(tmp_path / "a.json"), 1_000_000)
    acc.buy("AAA", 100, 1000, fee_pct=0.001)
    assert acc.cash == 1_000_000 - 100_000 - 100
    acc.sell("AAA", 50, 1100, fee_pct=0.001)
    assert acc.positions["AAA"]["qty"] == 50
    assert abs(acc.realized_pnl - (50 * 100 - 55)) < 1e-6
    v = acc.value({"AAA": 1100})
    assert v["positions"]["AAA"]["weight"] > 0
    assert abs(v["total"] - (acc.cash + 55_000)) < 1e-6


def test_rebalance_sells_first_and_persists(tmp_path):
    p = str(tmp_path / "a.json")
    acc = PaperAccount(p, 1_000_000)
    o1 = acc.rebalance_to({"AAA": 0.5, "BBB": 0.5}, {"AAA": 100, "BBB": 200}, 0.0)
    assert all(o["side"] == "BUY" for o in o1) and len(o1) == 2
    o2 = acc.rebalance_to({"AAA": 0.0, "BBB": 1.0}, {"AAA": 100, "BBB": 200}, 0.0)
    assert [o["side"] for o in o2] == ["SELL", "BUY"]
    assert "AAA" not in acc.positions
    acc.snapshot({"BBB": 200}, date="2024-01-01")
    acc.snapshot({"BBB": 210}, date="2024-01-01")
    assert len(acc.history) == 1
    acc.save()
    acc2 = PaperAccount(p)
    assert acc2.positions == acc.positions and acc2.cash == acc.cash
    assert isinstance(acc2.positions["BBB"]["qty"], int)


def test_dust_skipped(tmp_path):
    acc = PaperAccount(str(tmp_path / "a.json"), 1_000_000)
    acc.rebalance_to({"AAA": 0.5}, {"AAA": 100}, 0.0)
    qty = acc.positions["AAA"]["qty"]
    # 목표 0.505 -> 차이 0.5% < 1% -> 거래 없음
    assert acc.rebalance_to({"AAA": 0.505}, {"AAA": 100}, 0.0) == []
    assert acc.positions["AAA"]["qty"] == qty


def test_signals():
    prices = make_prices()
    sigs = today_signals(FixedStrategy({"AAA": 0.5}), prices)
    d = {s.ticker: s for s in sigs}
    assert d["AAA"].action == "BUY" and d["BBB"].action == "NONE"
    sigs = today_signals(FixedStrategy({"AAA": 0.0}), prices, {"AAA": 0.5})
    assert sigs[0].action == "SELL"
    sigs = today_signals(FixedStrategy({"AAA": 0.5}), prices, {"AAA": 0.5})
    assert sigs[0].action == "HOLD"
    assert isinstance(sigs[0], Signal)
    assert "AAA" in format_signals_korean(sigs)
