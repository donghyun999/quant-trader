"""Streamlit 대시보드 스모크 테스트 (AppTest, 합성 데이터, 네트워크 없음)."""
from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(autouse=True)
def _fake_data(monkeypatch):
    monkeypatch.setenv("QUANT_TRADER_FAKE_DATA", "1")


def _run_backtest_for(strategy: str):
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    [s for s in at.sidebar.selectbox if s.label == "전략"][0].select(strategy).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.button[[b.label for b in at.button].index("백테스트 실행")].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


@pytest.mark.parametrize("strategy", ["sma_cross", "dual_momentum", "ensemble", "vol_target", "tranched"])
def test_backtest_tab_renders(strategy):
    at = _run_backtest_for(strategy)
    assert any(m.label == "총 수익률" for m in at.metric)


def test_signal_tab_renders():
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.button[[b.label for b in at.button].index("신호 계산")].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
