"""실전 실행기(LiveTrader) 재현 검증.

과거 N 거래일을 하루씩 전진하며 실전 실행기를 가상 브로커로 실제 실행하고, 그 자산 곡선을
같은 조건의 백테스트(next_close, 슬리피지 0)와 비교한다. 두 결과가 거의 같아야 실행기에
버그(비중 계산, 주문 수량, 수수료, 필터)가 없다고 볼 수 있다.

실행: python -X utf8 research/replay_live.py [--tickers SPY,QQQ,TLT,GLD] [--strategy ensemble] [--days 250]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.broker import PaperBroker  # noqa: E402
from quant.config import LiveConfig  # noqa: E402
from quant.data import get_prices  # noqa: E402
from quant.live import LiveTrader  # noqa: E402
from quant.paper import PaperAccount  # noqa: E402
from quant.risk import RiskLimits  # noqa: E402
from quant.strategies import get_strategy  # noqa: E402

warnings.filterwarnings("ignore")


def replay(tickers: list[str], strategy: str, days: int, fee: float, threshold: float) -> dict:
    prices = get_prices(tickers, period="3y")
    idx = next(iter(prices.values())).index
    for df in prices.values():
        idx = idx.intersection(df.index)
    start = idx[-days]
    tmp = Path(tempfile.mkdtemp())
    acc = PaperAccount(str(tmp / "paper.json"), 10_000_000)
    cfg = LiveConfig(broker="paper", tickers=tickers, strategy=strategy, log_dir=str(tmp / "log"),
                     fee_pct=fee, threshold=threshold)
    cfg.risk = RiskLimits(max_position_weight=1.0)
    LiveConfig.root = property(lambda self: tmp)  # 킬 스위치 등 파일 위치를 임시 폴더로

    equity = {}
    for d in idx[idx >= start]:
        hist = {t: df.loc[:d] for t, df in prices.items()}
        LiveTrader(cfg, broker=PaperBroker(acc, fee_pct=fee), prices=hist).run(execute=True)
        px = {t: float(df.loc[d, "close"]) for t, df in prices.items()}
        equity[d] = acc.value(px)["total"]
    live_eq = pd.Series(equity)

    bt = run_backtest(get_strategy(strategy), prices,
                      BacktestConfig(fee_pct=fee, slippage_pct=0.0, execution="next_close",
                                     min_trade_weight=threshold), start=start)
    r1, r2 = live_eq.pct_change().dropna(), bt.equity.pct_change().dropna()
    return {
        "live_final": float(live_eq.iloc[-1]), "bt_final": float(bt.equity.iloc[-1]),
        "final_gap_pct": float(live_eq.iloc[-1] / bt.equity.iloc[-1] - 1),
        "daily_corr": float(r1.corr(r2)),
        "live_orders": len(acc.orders), "bt_trades": len(bt.trades),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,QQQ,TLT,GLD")
    ap.add_argument("--strategy", default="ensemble")
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--threshold", type=float, default=0.02)
    a = ap.parse_args(argv)
    r = replay([t.strip().upper() for t in a.tickers.split(",")], a.strategy, a.days, a.fee, a.threshold)
    print(f"실행기 최종 {r['live_final']:,.0f} / 백테스트 최종 {r['bt_final']:,.0f} / 차이 {r['final_gap_pct']:+.2%}")
    print(f"일별 수익률 상관 {r['daily_corr']:.3f} / 실행기 주문 {r['live_orders']} / 백테스트 거래 {r['bt_trades']}")
    ok = abs(r["final_gap_pct"]) < 0.02 and r["daily_corr"] > 0.98
    print("판정:", "일치 (실행기 정상)" if ok else "불일치 — 실행기 또는 백테스트 로직 점검 필요")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
