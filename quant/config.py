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
    strategy: str = "dual_momentum"
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
        strategy=str(live.get("strategy", "dual_momentum")),
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
