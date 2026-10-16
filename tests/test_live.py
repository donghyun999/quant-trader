"""실전 매매 계층 테스트: 브로커 인터페이스, 리스크 가드, 실행기, KIS 어댑터(가짜 HTTP)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.broker import PaperBroker, make_broker
from quant.broker.base import Balance, Position
from quant.broker.kis import ENDPOINTS, KISBroker, split_ticker
from quant.config import LiveConfig, load_config
from quant.live import LiveTrader, plan_orders, read_logs
from quant.paper import PaperAccount
from quant.risk import DailyState, PlannedOrder, RiskLimits
from quant.signals import Signal


def make_prices(n=300, tickers=("AAA", "BBB")):
    idx = pd.bdate_range("2024-01-01", periods=n)
    out = {}
    for i, t in enumerate(tickers):
        c = pd.Series(100.0 + np.arange(n) * (i + 1), index=idx)
        out[t] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000}, index=idx)
    return out


# ---------- 브로커 ----------
def test_paper_broker_roundtrip(tmp_path):
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    b = PaperBroker(acc, fee_pct=0.001, prices={"AAA": 100.0})
    r = b.submit_order("AAA", "BUY", 10)
    assert r.ok and r.order_id.startswith("paper")
    bal = b.get_balance()
    assert bal.positions["AAA"].qty == 10
    assert abs(bal.total - (1_000_000 - 1)) < 1e-6  # 수수료 1 만 빠짐
    r2 = b.submit_order("AAA", "SELL", 999)
    assert not r2.ok and "부족" in r2.message
    assert make_broker("paper", account=acc).name == "paper"


def test_balance_weights():
    b = Balance(cash=500.0, positions={"X": Position("X", 5, 90.0, 100.0)})
    assert b.total == 1000.0
    assert b.weights() == {"X": 0.5}


# ---------- 리스크 ----------
def _bal(cash=1_000_000, **pos):
    return Balance(cash=cash, positions={t: Position(t, q, px, px) for t, (q, px) in pos.items()})


def test_risk_kill_switch(tmp_path):
    (tmp_path / "STOP_TRADING").write_text("stop")
    passed, why = RiskLimits().check_orders([PlannedOrder("A", "BUY", 1, 100)], _bal(), None, tmp_path)
    assert passed == [] and "킬 스위치" in why[0]


def test_risk_position_weight_and_order_value(tmp_path):
    lim = RiskLimits(max_position_weight=0.3, max_order_value=200_000)
    orders = [PlannedOrder("A", "BUY", 3500, 100.0),  # 350k > 30% and > 200k
              PlannedOrder("B", "BUY", 1000, 100.0)]   # 100k ok
    passed, why = lim.check_orders(orders, _bal(), None, tmp_path)
    assert [o.ticker for o in passed] == ["B"]
    assert any("A:" in w for w in why)


def test_risk_sell_more_than_held_and_allowed(tmp_path):
    lim = RiskLimits(allowed_tickers=["A"])
    bal = _bal(cash=0, A=(10, 100.0))
    orders = [PlannedOrder("A", "SELL", 20, 100.0), PlannedOrder("B", "BUY", 1, 100.0)]
    passed, why = lim.check_orders(orders, bal, None, tmp_path)
    assert passed == [] and len(why) == 2


def test_risk_daily_loss_blocks_buy_not_sell(tmp_path):
    lim = RiskLimits(max_daily_loss_pct=0.05)
    bal = _bal(cash=0, A=(10, 90.0))  # total 900, start 1000 → -10%
    orders = [PlannedOrder("A", "SELL", 5, 90.0), PlannedOrder("A", "BUY", 1, 90.0)]
    passed, why = lim.check_orders(orders, bal, 1000.0, tmp_path)
    assert [o.side for o in passed] == ["SELL"]
    assert any("손실" in w for w in why)


def test_risk_max_orders(tmp_path):
    lim = RiskLimits(max_orders_per_run=1, max_position_weight=1.0)
    orders = [PlannedOrder("A", "BUY", 1, 1.0), PlannedOrder("B", "BUY", 1, 1.0)]
    passed, why = lim.check_orders(orders, _bal(), None, tmp_path)
    assert passed == [] and "주문 수" in why[-1]


def test_daily_state(tmp_path):
    st = DailyState(tmp_path / "s.json")
    assert st.day_start_equity(1000.0, "2026-01-01") == 1000.0
    assert st.day_start_equity(900.0, "2026-01-01") == 1000.0   # 같은 날은 유지
    assert st.day_start_equity(900.0, "2026-01-02") == 900.0    # 새 날은 갱신
    assert DailyState(tmp_path / "s.json").data["start_equity"] == 900.0


# ---------- 주문 계획 ----------
def test_plan_orders_sell_then_buy():
    bal = _bal(cash=0, A=(100, 100.0))  # total 10,000 전부 A
    sigs = [Signal("A", "SELL", 0.0, 1.0, 100.0, ""), Signal("B", "BUY", 0.5, 0.0, 50.0, "")]
    plan = plan_orders(sigs, bal, {"A": 100.0, "B": 50.0}, fee_pct=0.0)
    assert [(o.side, o.ticker, o.qty) for o in plan] == [("SELL", "A", 100), ("BUY", "B", 100)]


def test_plan_orders_threshold_skips_small_diff():
    bal = _bal(cash=0, A=(50, 100.0))  # A 100%
    sigs = [Signal("A", "HOLD", 0.99, 1.0, 100.0, "")]
    assert plan_orders(sigs, bal, {"A": 100.0}, 0.0) == []


def test_plan_orders_respects_cash():
    bal = _bal(cash=1000.0)
    sigs = [Signal("A", "BUY", 1.0, 0.0, 300.0, "")]
    plan = plan_orders(sigs, bal, {"A": 300.0}, fee_pct=0.0)
    assert plan[0].qty == 3


# ---------- 실행기 ----------
def _cfg(tmp_path, **kw):
    cfg = LiveConfig(broker="paper", tickers=["AAA", "BBB"], strategy="buy_hold", log_dir=str(tmp_path / "log"))
    cfg.risk = RiskLimits(max_position_weight=1.0)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_live_dry_run_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("quant.config.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(LiveConfig, "root", property(lambda self: tmp_path))
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    broker = PaperBroker(acc)
    tr = LiveTrader(_cfg(tmp_path), broker=broker, prices=make_prices())
    rep = tr.run(execute=False)
    assert rep.error == "" and rep.dry_run
    assert len(rep.planned) == 2 and rep.results == []
    assert acc.positions == {}                     # 아무것도 안 샀다
    logs = read_logs(tmp_path / "log")
    assert len(logs) == 1 and logs[0]["dry_run"] is True


def test_live_execute_on_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(LiveConfig, "root", property(lambda self: tmp_path))
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    tr = LiveTrader(_cfg(tmp_path), broker=PaperBroker(acc), prices=make_prices())
    rep = tr.run(execute=True)
    assert rep.error == "" and all(r["ok"] for r in rep.results) and len(rep.results) == 2
    assert set(acc.positions) == {"AAA", "BBB"}
    assert rep.balance_after["total"] < 1_000_000  # 수수료
    # 두 번째 실행은 이미 목표 비중이라 주문 없음
    rep2 = tr.run(execute=True)
    assert rep2.planned == [] and rep2.results == []


def test_live_kill_switch_blocks_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(LiveConfig, "root", property(lambda self: tmp_path))
    (tmp_path / "STOP_TRADING").write_text("")
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    rep = LiveTrader(_cfg(tmp_path), broker=PaperBroker(acc), prices=make_prices()).run(execute=True)
    assert rep.results == [] and "킬 스위치" in rep.rejected[0]
    assert acc.positions == {}


def test_live_error_is_captured_and_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(LiveConfig, "root", property(lambda self: tmp_path))

    class Broken(PaperBroker):
        def get_balance(self, tickers=None):
            raise RuntimeError("boom")

    rep = LiveTrader(_cfg(tmp_path), broker=Broken(PaperAccount(str(tmp_path / "p.json"))),
                     prices=make_prices()).run(execute=True)
    assert "boom" in rep.error and read_logs(tmp_path / "log")


# ---------- 설정 ----------
def test_load_config(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    p.write_text("""
[live]
broker = "kis"
tickers = ["005930.KS", " 069500.ks "]
strategy = "sma_cross"
[live.params]
fast = 10
[kis]
app_key = "k"
mode = "prod"
[risk]
max_position_weight = 0.3
allowed_tickers = ["005930.KS"]
""", encoding="utf-8")
    monkeypatch.setenv("KIS_APP_SECRET", "from-env")
    cfg = load_config(p)
    assert cfg.broker == "kis" and cfg.tickers == ["005930.KS", "069500.KS"]
    assert cfg.params == {"fast": 10}
    assert cfg.kis["app_key"] == "k" and cfg.kis["app_secret"] == "from-env" and cfg.kis["mode"] == "prod"
    assert cfg.risk.max_position_weight == 0.3 and cfg.risk.allowed_tickers == ["005930.KS"]
    assert load_config(tmp_path / "none.toml").broker == "paper"


# ---------- KIS 어댑터 (가짜 HTTP) ----------
class FakeResp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self._d


class FakeSession:
    """호출 기록 + 경로별 응답."""

    def __init__(self):
        self.calls: list[dict] = []
        self.responses: dict[str, object] = {}

    def post(self, url, json=None, timeout=None, **kw):
        self.calls.append({"method": "POST", "url": url, "body": json, "headers": kw.get("headers")})
        return FakeResp({"access_token": "tok", "expires_in": 86400})

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "body": json, "headers": headers})
        key = url.split("/uapi/")[-1]
        r = self.responses.get(key)
        if callable(r):
            return r(method, params, json)
        return FakeResp(r or {"rt_cd": "0", "output": {}})


@pytest.fixture
def kis(tmp_path):
    s = FakeSession()
    b = KISBroker("key", "secret", "12345678-01", mode="vts", token_cache=tmp_path / "tok.json", session=s)
    b._throttle = lambda: None  # 테스트에서 sleep 제거
    return b, s


def test_split_ticker():
    assert split_ticker("005930.KS") == ("KR", "005930")
    assert split_ticker("035720.kq") == ("KR", "035720")
    assert split_ticker("spy") == ("US", "SPY")


def test_kis_init_validation(tmp_path):
    with pytest.raises(ValueError):
        KISBroker("k", "s", "123", mode="vts", session=FakeSession())
    with pytest.raises(ValueError):
        KISBroker("k", "s", "12345678-01", mode="live", session=FakeSession())
    assert KISBroker("k", "s", "12345678-01", mode="prod", session=FakeSession()).real_money is True
    assert KISBroker("k", "s", "12345678-01", mode="vts", session=FakeSession()).real_money is False


def test_kis_token_cached(kis, tmp_path):
    b, s = kis
    assert b.token() == "tok"
    assert b.token() == "tok"
    assert sum(1 for c in s.calls if c["method"] == "POST") == 1
    assert (tmp_path / "tok.json").exists()
    # 새 인스턴스는 파일 캐시를 재사용
    s2 = FakeSession()
    b2 = KISBroker("key", "secret", "12345678-01", mode="vts", token_cache=tmp_path / "tok.json", session=s2)
    assert b2.token() == "tok" and s2.calls == []


def test_kis_price_and_order_kr(kis):
    b, s = kis
    s.responses["domestic-stock/v1/quotations/inquire-price"] = {"rt_cd": "0", "output": {"stck_prpr": "71,500"}}
    assert b.get_price("005930.KS") == 71500.0
    s.responses["domestic-stock/v1/trading/order-cash"] = {"rt_cd": "0", "msg1": "정상", "output": {"ODNO": "0001"}}
    r = b.submit_order("005930.KS", "BUY", 3)
    assert r.ok and r.order_id == "0001" and r.broker == "kis:vts"
    call = s.calls[-1]
    assert call["headers"]["tr_id"] == ENDPOINTS["kr_order"]["tr"][("vts", "BUY")]
    assert call["body"]["PDNO"] == "005930" and call["body"]["ORD_DVSN"] == "01" and call["body"]["ORD_QTY"] == "3"
    r2 = b.submit_order("005930.KS", "SELL", 2, price=70000)
    assert s.calls[-1]["body"]["ORD_DVSN"] == "00" and s.calls[-1]["body"]["ORD_UNPR"] == "70000"
    assert s.calls[-1]["headers"]["tr_id"] == ENDPOINTS["kr_order"]["tr"][("vts", "SELL")]


def test_kis_order_error_returns_not_ok(kis):
    b, s = kis
    s.responses["domestic-stock/v1/trading/order-cash"] = {"rt_cd": "1", "msg_cd": "APBK0013", "msg1": "주문가능금액 초과"}
    r = b.submit_order("005930.KS", "BUY", 1)
    assert not r.ok and "주문가능금액" in r.message


def test_kis_balance_kr(kis):
    b, s = kis
    s.responses["domestic-stock/v1/trading/inquire-balance"] = {
        "rt_cd": "0",
        "output1": [{"pdno": "005930", "hldg_qty": "10", "pchs_avg_pric": "70000", "prpr": "71500"},
                    {"pdno": "000660", "hldg_qty": "0", "pchs_avg_pric": "0", "prpr": "1"}],
        "output2": [{"dnca_tot_amt": "1500000"}],
    }
    bal = b.get_balance(["005930.KS"])
    assert bal.cash == 1_500_000 and list(bal.positions) == ["005930.KS"]
    assert bal.positions["005930.KS"].qty == 10 and bal.total == 1_500_000 + 715_000


def test_kis_us_order_is_limit_with_slippage(kis):
    b, s = kis
    s.responses["overseas-price/v1/quotations/price"] = {"rt_cd": "0", "output": {"last": "100.00"}}
    s.responses["overseas-stock/v1/trading/order"] = {"rt_cd": "0", "output": {"ODNO": "9"}}
    r = b.submit_order("SPY", "BUY", 2)
    body = s.calls[-1]["body"]
    assert r.ok and body["OVRS_EXCG_CD"] == "AMEX" and body["OVRS_ORD_UNPR"] == "100.30" and body["ORD_DVSN"] == "00"
    assert s.calls[-1]["headers"]["tr_id"] == ENDPOINTS["us_order"]["tr"][("vts", "BUY")]
    r2 = b.submit_order("ZZZZ", "BUY", 1, price=10.0)
    assert not r2.ok and "거래소 코드" in r2.message


def test_live_notes_stale_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(LiveConfig, "root", property(lambda self: tmp_path))
    prices = make_prices()
    prices["AAA"] = prices["AAA"].iloc[:-1]   # AAA 만 하루 늦음
    acc = PaperAccount(str(tmp_path / "p.json"), 1_000_000)
    rep = LiveTrader(_cfg(tmp_path), broker=PaperBroker(acc), prices=prices).run(execute=False)
    assert rep.error == ""
    assert any("AAA" in n and "오래된" in n for n in rep.notes)


def test_tranche_calendar():
    from quant.live import tranche_calendar
    idx = pd.bdate_range("2024-01-02", "2024-01-31")   # 1/31 은 1월 마지막 거래일 → 트랜치 1
    hits, nxt = tranche_calendar(idx)
    assert hits == [1] and nxt is not None and nxt > idx[-1]
    hits2, nxt2 = tranche_calendar(pd.bdate_range("2024-01-02", "2024-01-17"))
    assert hits2 == [] and nxt2 is not None
