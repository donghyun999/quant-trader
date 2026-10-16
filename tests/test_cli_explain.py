"""cli explain / signal / strategies 스모크 (합성 데이터, 네트워크 없음)."""
from __future__ import annotations

import cli
import quant.data as qd
from quant.data import fake_prices


def _patch_prices(monkeypatch):
    monkeypatch.setattr(qd, "get_prices", lambda tickers, start=None, end=None, period="5y", force=False: fake_prices(tickers, 700))


def test_explain_ensemble(monkeypatch, capsys):
    _patch_prices(monkeypatch)
    cli.main(["explain", "--tickers", "AAA,BBB,CCC", "--strategy", "ensemble"])
    out = capsys.readouterr().out
    assert "시장 추세 필터" in out and "구성원별" in out and "tranched(base=trend_invvol)" in out


def test_signal_default_strategy_is_ensemble(monkeypatch, capsys):
    _patch_prices(monkeypatch)
    cli.main(["signal", "--tickers", "AAA,BBB"])
    out = capsys.readouterr().out
    assert "전략 앙상블" in out


def test_strategies_lists_all(capsys):
    from quant.strategies import STRATEGIES
    cli.main(["strategies"])
    out = capsys.readouterr().out
    assert all(name in out for name in STRATEGIES)
