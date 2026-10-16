from quant.config import LiveConfig, validate_config
from quant.risk import RiskLimits


def test_doctor_clean_paper():
    cfg = LiveConfig(broker="paper", tickers=["SPY", "QQQ"], strategy="ensemble")
    assert validate_config(cfg) == []


def test_doctor_flags_mixed_markets_and_kis_keys():
    cfg = LiveConfig(broker="kis", tickers=["SPY", "005930.KS"], strategy="ensemble", kis={"mode": "prod", "account_no": "123"})
    cfg.risk = RiskLimits(max_order_value=0)
    issues = validate_config(cfg)
    text = " ".join(issues)
    assert "섞여" in text and "app_key" in text and "8자리" in text and "max_order_value" in text


def test_doctor_bad_strategy_params():
    cfg = LiveConfig(strategy="sma_cross", params={"fast": 100, "slow": 20})
    assert any("파라미터" in i for i in validate_config(cfg))
    cfg2 = LiveConfig(strategy="nope")
    assert any("알 수 없는 전략" in i for i in validate_config(cfg2))
