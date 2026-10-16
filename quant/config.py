"""실전 매매 설정 (config.toml) 로더.

config.toml 은 증권사 키를 담으므로 git 에 올리지 않는다 (.gitignore). config.example.toml 을 복사해 쓴다.
환경변수 KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT_NO 가 있으면 파일 값보다 우선한다.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .risk import RiskLimits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"


@dataclass
class LiveConfig:
    broker: str = "paper"              # "paper" | "kis"
    tickers: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "TLT", "GLD"])
    strategy: str = "ensemble"
    params: dict[str, Any] = field(default_factory=dict)
    period: str = "2y"
    fee_pct: float = 0.0015
    threshold: float = 0.02            # 목표/현재 비중 차이가 이보다 작으면 주문 안 함
    order_type: str = "market"         # "market" | "limit"
    kis: dict[str, Any] = field(default_factory=dict)
    risk: RiskLimits = field(default_factory=RiskLimits)
    log_dir: str = "live_log"

    @property
    def root(self) -> Path:
        return PROJECT_ROOT


def load_config(path: str | Path | None = None) -> LiveConfig:
    p = Path(path) if path else DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if p.exists():
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    live = raw.get("live", {})
    cfg = LiveConfig(
        broker=str(live.get("broker", "paper")),
        tickers=[str(t).strip().upper() for t in live.get("tickers", LiveConfig().tickers) if str(t).strip()],
        strategy=str(live.get("strategy", "ensemble")),
        params=dict(live.get("params", {})),
        period=str(live.get("period", "2y")),
        fee_pct=float(live.get("fee_pct", 0.0015)),
        threshold=float(live.get("threshold", 0.02)),
        order_type=str(live.get("order_type", "market")),
        kis=dict(raw.get("kis", {})),
        log_dir=str(live.get("log_dir", "live_log")),
    )
    r = raw.get("risk", {})
    cfg.risk = RiskLimits(
        max_position_weight=float(r.get("max_position_weight", 0.5)),
        max_order_value=float(r.get("max_order_value", 0.0)),
        max_orders_per_run=int(r.get("max_orders_per_run", 20)),
        max_daily_loss_pct=float(r.get("max_daily_loss_pct", 0.05)),
        allowed_tickers=[str(t).upper() for t in r.get("allowed_tickers", [])],
        kill_switch_file=str(r.get("kill_switch_file", "STOP_TRADING")),
        min_order_value=float(r.get("min_order_value", 0.0)),
    )
    # 환경변수 우선
    for env, key in (("KIS_APP_KEY", "app_key"), ("KIS_APP_SECRET", "app_secret"),
                     ("KIS_ACCOUNT_NO", "account_no"), ("KIS_MODE", "mode")):
        if os.environ.get(env):
            cfg.kis[key] = os.environ[env]
    return cfg


def validate_config(cfg: LiveConfig) -> list[str]:
    """실전 설정의 흔한 실수를 점검해 한국어 문제 목록을 돌려준다 (비어 있으면 통과)."""
    from .strategies import STRATEGIES

    issues: list[str] = []
    if cfg.broker not in ("paper", "kis"):
        issues.append(f"broker 는 paper 또는 kis 여야 합니다: {cfg.broker}")
    if not cfg.tickers:
        issues.append("tickers 가 비어 있습니다.")
    kr = [t for t in cfg.tickers if t.endswith((".KS", ".KQ"))]
    if kr and len(kr) != len(cfg.tickers):
        issues.append("국내와 해외 종목이 섞여 있습니다. 통화가 달라 현금 계산이 틀어지므로 설정을 나누세요.")
    if cfg.strategy not in STRATEGIES:
        issues.append(f"알 수 없는 전략: {cfg.strategy} (가능: {', '.join(STRATEGIES)})")
    else:
        try:
            STRATEGIES[cfg.strategy](**cfg.params)
        except Exception as e:  # noqa: BLE001
            issues.append(f"전략 파라미터 오류: {e}")
    if cfg.strategy in ("dual_momentum", "momentum_ensemble", "rsi_reversion", "bollinger"):
        issues.append(f"[권고] {cfg.strategy} 단독은 평가 세트에서 낙폭이 벤치마크보다 큰 경우가 많았습니다. ensemble 을 권합니다.")
    if cfg.broker == "kis":
        k = cfg.kis
        for key in ("app_key", "app_secret", "account_no"):
            if not k.get(key):
                issues.append(f"[kis] {key} 가 비어 있습니다 (환경변수 KIS_{key.upper()} 도 가능).")
        acct = str(k.get("account_no", "")).replace("-", "")
        if acct and len(acct) != 10:
            issues.append("[kis] account_no 는 8자리-2자리(예: 12345678-01) 형식이어야 합니다.")
        if k.get("mode", "vts") not in ("vts", "prod"):
            issues.append("[kis] mode 는 vts 또는 prod 여야 합니다.")
        if k.get("mode") == "prod" and cfg.risk.max_order_value <= 0:
            issues.append("[권고] 실전(prod)인데 max_order_value 가 0(무제한)입니다. 1회 주문 한도를 두세요.")
    r = cfg.risk
    if not 0 < r.max_position_weight <= 1:
        issues.append("risk.max_position_weight 는 0~1 사이여야 합니다.")
    if not 0 < r.max_daily_loss_pct < 1:
        issues.append("risk.max_daily_loss_pct 는 0~1 사이여야 합니다.")
    if r.allowed_tickers and (set(cfg.tickers) - set(r.allowed_tickers)):
        issues.append("tickers 중 risk.allowed_tickers 에 없는 종목이 있어 주문이 거부됩니다.")
    if not 0 <= cfg.threshold < 0.5:
        issues.append("threshold 는 0~0.5 사이가 정상입니다.")
    return issues
