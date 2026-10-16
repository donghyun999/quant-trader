"""전략 평가 세트(테스트셋).

여러 종목 묶음 × 여러 기간 × 워크포워드로 모든 전략을 돌려 '강건성 점수'를 매긴다.
단일 백테스트 숫자가 아니라 "처음 보는 구간(OOS)에서도 벤치마크를 이기는가"를 본다.

실행:  python -X utf8 research/evaluate.py [--quick] [--strategies a,b] [--out research/results]
출력:  research/results/leaderboard.csv, research/results/runs_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.data import get_prices  # noqa: E402
from quant.optimize import walk_forward  # noqa: E402
from quant.strategies import STRATEGIES  # noqa: E402

warnings.filterwarnings("ignore")

# ---------- 평가 세트 정의 ----------
UNIVERSES: dict[str, list[str]] = {
    "us_etf4": ["SPY", "QQQ", "TLT", "GLD"],
    "us_global7": ["SPY", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ"],
    "kr_large4": ["005930.KS", "000660.KS", "035420.KS", "005380.KS"],
    "kr_etf3": ["069500.KS", "229200.KS", "114260.KS"],
    "kr_top8": ["005930.KS", "000660.KS", "035420.KS", "005380.KS", "000270.KS", "068270.KS", "005490.KS", "105560.KS"],
    "kr_mixed5": ["069500.KS", "229200.KS", "114260.KS", "148070.KS", "132030.KS"],  # 코스피200, 코스닥150, 국고채3년/10년, 금
    "spy_only": ["SPY"],
    "samsung_only": ["005930.KS"],
}
# 홀드아웃: 개발 중 한 번도 보지 않은 묶음. 과적합(유니버스 선택 편향) 점검용. --holdout 으로만 평가.
HOLDOUT_UNIVERSES: dict[str, list[str]] = {
    "us_sectors5": ["XLK", "XLF", "XLE", "XLV", "XLU"],
    "us_mixed6": ["VTI", "VEA", "VWO", "BND", "IAU", "VNQ"],
    "kr_theme4": ["091160.KS", "102780.KS", "143850.KS", "148070.KS"],  # 반도체, 삼성그룹, S&P500, 국고채10년
    "aapl_only": ["AAPL"],
}
PERIODS = ["3y", "5y", "10y"]
WF_PERIOD = "10y"
WF_SPLITS = 4

# 워크포워드용 작은 파라미터 격자 (기본값 주변). 전략별로 없으면 {} = 기본값만.
WF_GRIDS: dict[str, dict[str, list]] = {
    "sma_cross": {"fast": [10, 20, 50], "slow": [60, 120, 200]},
    "rsi_reversion": {"period": [7, 14], "oversold": [25, 30, 35]},
    "bollinger": {"period": [20, 40], "n_std": [1.5, 2.0, 2.5]},
    "dual_momentum": {"lookback": [63, 126, 252], "top_n": [1, 2]},
    "momentum_ensemble": {"top_n": [1, 2, 3], "skip_recent": [0, 21]},
    "trend_invvol": {"trend_window": [100, 200], "max_weight": [0.5, 1.0]},
    "vol_target": {"target_vol": [0.10, 0.15, 0.20]},
    "tranched": {"n_tranches": [2, 4]},
    "dd_control": {"dd_trigger": [0.08, 0.12, 0.2]},
    "regime_guard": {"soft_exposure": [0.3, 0.5, 0.7]},
    "ensemble": {},
    "buy_hold": {},
}

# 위기 구간: (이름, 시작, 끝). 전략 vs 벤치마크 수익률 차이를 본다.
CRISES = [
    ("covid_2020", "2020-02-19", "2020-03-23"),
    ("bear_2022", "2022-01-03", "2022-10-12"),
    ("kr_2024_08", "2024-07-11", "2024-08-05"),
]


def crisis_excess(equity: pd.Series, bench: pd.Series) -> dict[str, float]:
    out = {}
    for name, a, b in CRISES:
        e = equity.loc[a:b]
        bb = bench.loc[a:b]
        if len(e) < 5 or len(bb) < 5:
            out[name] = float("nan")
            continue
        out[name] = float((e.iloc[-1] / e.iloc[0] - 1) - (bb.iloc[-1] / bb.iloc[0] - 1))
    return out

CFG = BacktestConfig(fee_pct=0.0015, slippage_pct=0.0005, min_trade_weight=0.02)  # min_trade = live threshold 와 동일

# 변형: 라벨 -> (전략 이름, 고정 파라미터). --variants 로 포함. 워크포워드는 고정 파라미터만 사용(격자 없음).
VARIANTS: dict[str, tuple[str, dict]] = {
    "ens_tv+bh": ("ensemble", {"members": "trend_invvol,buy_hold"}),
    "ens_tv+me": ("ensemble", {"members": "trend_invvol,momentum_ensemble"}),
    "ens_all4": ("ensemble", {"members": "trend_invvol,momentum_ensemble,dual_momentum,buy_hold"}),
    "ens_tv+me+dm": ("ensemble", {"members": "trend_invvol,momentum_ensemble,dual_momentum"}),
    "vt15_tv": ("vol_target", {"base": "trend_invvol", "target_vol": 0.15}),
    "vt15_bh": ("vol_target", {"base": "buy_hold", "target_vol": 0.15}),
    "tr_tv": ("tranched", {"base": "trend_invvol"}),
    "tr_me": ("tranched", {"base": "momentum_ensemble"}),
    "tr_dm": ("tranched", {"base": "dual_momentum"}),
    "ens_untranched": ("ensemble", {"members": "trend_invvol,momentum_ensemble,buy_hold"}),
    "me_fixed": ("momentum_ensemble", {"top_n": 2}),
    "me_skip21": ("momentum_ensemble", {"skip_recent": 21}),
    "me_top3": ("momentum_ensemble", {"top_n": 3}),
    "me_skip21_top3": ("momentum_ensemble", {"skip_recent": 21, "top_n": 3}),
    "ens_me_skip21": ("ensemble", {"members": "tranched(base=trend_invvol),tranched(base=momentum_ensemble(skip_recent=21)),buy_hold"}),
    "ens_me_top3": ("ensemble", {"members": "tranched(base=trend_invvol),tranched(base=momentum_ensemble(top_n=3)),buy_hold"}),
    "ens_me_skip21_top3": ("ensemble", {"members": "tranched(base=trend_invvol),tranched(base=momentum_ensemble(skip_recent=21|top_n=3)),buy_hold"}),
    "ens2_tv+bh": ("ensemble", {"members": "tranched(base=trend_invvol),buy_hold"}),
    "ens2_rg+bh": ("ensemble", {"members": "tranched(base=regime_guard),buy_hold"}),
    "ens3_tv+rg+bh": ("ensemble", {"members": "tranched(base=trend_invvol),tranched(base=regime_guard),buy_hold"}),
    "ens3_tv+me+bh": ("ensemble", {}),   # 현 기본 구성 (비교 기준)
    "tv_mkt": ("trend_invvol", {"market_filter": 1}),
    "rg_fixed": ("regime_guard", {"soft_exposure": 0.5}),    # 격자 없이 기본값 고정 (trend_invvol 과 공정 비교용)
    "rg_stress": ("regime_guard", {"use_stress": 1}),
    "tv_fixed": ("trend_invvol", {"trend_window": 200}),      # 격자 없이 기본값 고정
    "rg_persist5": ("regime_guard", {"use_stress": 1, "stress_days": 5}),
    "rg_re50": ("regime_guard", {"reentry_window": 50}),
    "rg_re100": ("regime_guard", {"reentry_window": 100}),
    "rg_hard_re50": ("regime_guard", {"soft_exposure": 0.0, "reentry_window": 50}),
    "rg_hard_re100": ("regime_guard", {"soft_exposure": 0.0, "reentry_window": 100}),
    "rg_hard_fixed": ("regime_guard", {"soft_exposure": 0.0}),
    "rg_ramp_sym5": ("regime_guard", {"ramp": 1, "band": 0.05}),
    "rg_ramp_sym10": ("regime_guard", {"ramp": 1, "band": 0.10}),
    "rg_ramp_down5": ("regime_guard", {"ramp": 2, "band": 0.05}),
    "rg_ramp_down10": ("regime_guard", {"ramp": 2, "band": 0.10}),
    "rg_monthly": ("regime_guard", {"daily_exposure": 0}),
    "rg_hard": ("regime_guard", {"soft_exposure": 0.0}),
    "tr_rg": ("tranched", {"base": "regime_guard"}),
    "ens_rg": ("ensemble", {"members": "tranched(base=regime_guard),tranched(base=momentum_ensemble),buy_hold"}),
    "tv_proxy": ("trend_invvol", {"cash_proxy": 1}),
    "tr_tv_proxy": ("tranched", {"base": "trend_invvol(cash_proxy=1)"}),
    "ens_proxy": ("ensemble", {"members": "tranched(base=trend_invvol(cash_proxy=1)),tranched(base=momentum_ensemble),buy_hold"}),
    "dd10_ens": ("dd_control", {"base": "ensemble", "dd_trigger": 0.10}),
    "dd15_ens": ("dd_control", {"base": "ensemble", "dd_trigger": 0.15}),
    "dd10_bh": ("dd_control", {"base": "buy_hold", "dd_trigger": 0.10}),
    "ens_w211": ("ensemble", {"weights": "2,1,1"}),
    "ens_w121": ("ensemble", {"weights": "1,2,1"}),
    "ens_w112": ("ensemble", {"weights": "1,1,2"}),
    "ens_w110": ("ensemble", {"weights": "1,1,0"}),
    "tv_p3": ("trend_invvol", {"max_positions": 3}),
    "tv_p4": ("trend_invvol", {"max_positions": 4}),
    "ens_p4": ("ensemble", {"members": "tranched(base=trend_invvol(max_positions=4)),tranched(base=momentum_ensemble),buy_hold"}),
    "me_mkt": ("momentum_ensemble", {"market_filter": 1}),
    "ens_me_mkt": ("ensemble", {"members": "tranched(base=trend_invvol),tranched(base=momentum_ensemble(market_filter=1)),buy_hold"}),
    "tv_b02": ("trend_invvol", {"band": 0.02}),
    "tv_b05": ("trend_invvol", {"band": 0.05}),
    "ens_b03": ("ensemble", {"members": "tranched(base=trend_invvol(band=0.03)),tranched(base=momentum_ensemble),buy_hold"}),
    "tv_w100": ("trend_invvol", {"trend_window": 100}),
    "tv_w150": ("trend_invvol", {"trend_window": 150}),
    "tv_w250": ("trend_invvol", {"trend_window": 250}),
    "ens_w150": ("ensemble", {"members": "tranched(base=trend_invvol(trend_window=150)),tranched(base=momentum_ensemble),buy_hold"}),
    "tr_tv_mkt": ("tranched", {"base": "trend_invvol(market_filter=1)"}),
    "ens_mkt": ("ensemble", {"members": "tranched(base=trend_invvol(market_filter=1)),tranched(base=momentum_ensemble),buy_hold"}),
    "tr_tv_n2": ("tranched", {"base": "trend_invvol", "n_tranches": 2, "spacing": 10}),
    "tr_tv_n4s3": ("tranched", {"base": "trend_invvol", "n_tranches": 4, "spacing": 3}),
    "tr_tv_n4s7": ("tranched", {"base": "trend_invvol", "n_tranches": 4, "spacing": 7}),
    "tr_tv_n8s2": ("tranched", {"base": "trend_invvol", "n_tranches": 8, "spacing": 2}),
    "tr_me_n2": ("tranched", {"base": "momentum_ensemble", "n_tranches": 2, "spacing": 10}),
    "tr_me_n8s2": ("tranched", {"base": "momentum_ensemble", "n_tranches": 8, "spacing": 2}),
    "vt15_ens": ("vol_target", {"base": "ensemble", "target_vol": 0.15}),
    "vt12_ens": ("vol_target", {"base": "ensemble", "target_vol": 0.12}),
    "ens_tr_tv+bh": ("ensemble", {"members": "tranched(base=trend_invvol),buy_hold"}),
}


# 시장별 편도 비용(수수료+세금). 한국 개별주/ETF 0.15% (매도세 포함 평균), 미국 ETF 0.05%. 섞인 묶음은 높은 쪽.
FEE_BY_MARKET = {"KR": 0.0015, "US": 0.0005}


def fee_for(tickers: list[str], override: float | None) -> float:
    if override is not None:
        return override
    kr = any(t.endswith((".KS", ".KQ")) for t in tickers)
    return FEE_BY_MARKET["KR"] if kr else FEE_BY_MARKET["US"]


def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def evaluate_strategy(name: str, data: dict[str, dict[str, pd.DataFrame]], quick: bool,
                      cfg: BacktestConfig = CFG, fixed: dict | None = None,
                      grid: dict | None = None, cls=None) -> list[dict]:
    """한 전략(또는 변형)을 모든 유니버스/기간에 대해 평가. 행 단위 결과 반환.

    name: 결과 라벨. cls: 전략 클래스(None 이면 STRATEGIES[name]). fixed: 고정 파라미터.
    grid: 워크포워드 격자(None 이면 WF_GRIDS[name], 고정 파라미터는 단일값 격자로 합쳐짐).
    """
    cls = cls or STRATEGIES[name]
    fixed = fixed or {}
    if grid is None:
        grid = WF_GRIDS.get(name, {})
    grid = {**grid, **{k: [v] for k, v in fixed.items()}}
    rows: list[dict] = []
    for uni, prices10 in data.items():
        base_kw = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
        CFG_ = BacktestConfig(**{**base_kw, "fee_pct": fee_for(list(prices10), getattr(cfg, "_fee_override", None))})
        if cls.multi_asset and len(prices10) == 1 and name == "dual_momentum":
            pass  # 단일 종목 듀얼모멘텀 = 절대 모멘텀만. 평가에 포함.
        # 전체 기간 백테스트 (기본 파라미터)
        for period in (["5y"] if quick else PERIODS):
            prices = _slice(prices10, period)
            if not prices:
                continue
            try:
                res = run_backtest(cls(**fixed), prices, CFG_)
            except Exception as e:  # noqa: BLE001
                rows.append({"strategy": name, "universe": uni, "kind": f"bt_{period}", "error": str(e)})
                continue
            m = res.metrics
            rows.append({
                "strategy": name, "universe": uni, "kind": f"bt_{period}",
                "total_return": _f(m["total_return"]), "cagr": _f(m["cagr"]), "sharpe": _f(m["sharpe"]),
                "mdd": _f(m["max_drawdown"]), "num_trades": int(m["num_trades"]),
                "bench_return": _f(m.get("benchmark_total_return")), "bench_mdd": _f(m.get("benchmark_max_drawdown")),
                "excess_return": _f(m["total_return"]) - _f(m.get("benchmark_total_return")),
                "mdd_improvement": _f(m["max_drawdown"]) - _f(m.get("benchmark_max_drawdown")),  # 양수면 낙폭 작음
                **({f"crisis_{k}": v for k, v in crisis_excess(res.equity, res.benchmark_equity).items()} if period == "10y" else {}),
            })
        # 워크포워드 (OOS)
        if quick and uni not in ("us_etf4", "kr_large4"):
            continue
        try:
            wf = walk_forward(cls, prices10, grid, CFG_, n_splits=WF_SPLITS)
            om = wf["oos_metrics"]
            folds = wf["folds"]
            rows.append({
                "strategy": name, "universe": uni, "kind": "wf_oos",
                "total_return": _f(om["total_return"]), "cagr": _f(om["cagr"]), "sharpe": _f(om["sharpe"]),
                "mdd": _f(om["max_drawdown"]), "num_trades": int(om["num_trades"]),
                "bench_return": _f(om.get("benchmark_total_return")), "bench_mdd": _f(om.get("benchmark_max_drawdown")),
                "excess_return": _f(om["total_return"]) - _f(om.get("benchmark_total_return")),
                "mdd_improvement": _f(om["max_drawdown"]) - _f(om.get("benchmark_max_drawdown")),
                "is_sharpe_avg": float(np.nanmean([f["train_metric"] for f in folds])),
                "oos_fold_sharpes": [round(_f(f["test_metric"]), 2) for f in folds],
                "n_param_sets": len({tuple(sorted(f["best_params"].items())) for f in folds}),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"strategy": name, "universe": uni, "kind": "wf_oos", "error": str(e)})
    return rows


def _slice(prices10: dict[str, pd.DataFrame], period: str) -> dict[str, pd.DataFrame]:
    years = int(period.rstrip("y"))
    end = max(df.index[-1] for df in prices10.values())
    start = end - pd.DateOffset(years=years)
    out = {t: df.loc[df.index >= start] for t, df in prices10.items()}
    return {t: d for t, d in out.items() if len(d) > 60}


def score_table(rows: list[dict]) -> pd.DataFrame:
    """전략별 강건성 점수.

    robust_score = 0.5*평균 OOS 샤프 + 0.3*(OOS에서 벤치마크 이긴 비율) + 0.2*(OOS 낙폭 개선 비율)
                   + 0.2*평균 전체기간 샤프(5y)  - 0.3*(파라미터 불안정: 구간별 최적값이 전부 다른 비율)
                   + 1.0*위기구간 평균 초과수익 (코로나 2020, 약세장 2022, 2024-08 급락)
    """
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        return df
    out = []
    for s, g in df.groupby("strategy"):
        oos = g[g.kind == "wf_oos"]
        bt5 = g[g.kind == "bt_5y"]
        bt10 = g[g.kind == "bt_10y"]
        unstable = float((oos["n_param_sets"] >= WF_SPLITS).mean()) if len(oos) else float("nan")
        ccols = [c for c in bt10.columns if c.startswith("crisis_")]
        crisis = float(np.nanmean(bt10[ccols].to_numpy())) if len(bt10) and ccols else float("nan")
        row = {
            "strategy": s,
            "oos_sharpe_mean": float(oos["sharpe"].mean()),
            "oos_sharpe_min": float(oos["sharpe"].min()),
            "oos_beat_bench_frac": float((oos["excess_return"] > 0).mean()),
            "oos_mdd_better_frac": float((oos["mdd_improvement"] > 0).mean()),
            "oos_excess_return_mean": float(oos["excess_return"].mean()),
            "bt5_sharpe_mean": float(bt5["sharpe"].mean()),
            "bt5_beat_bench_frac": float((bt5["excess_return"] > 0).mean()),
            "bt10_sharpe_mean": float(bt10["sharpe"].mean()) if len(bt10) else float("nan"),
            "bt10_cagr_mean": float(bt10["cagr"].mean()) if len(bt10) else float("nan"),
            "mdd_worst": float(g["mdd"].min()),
            "param_unstable_frac": unstable,
            "crisis_excess_mean": crisis,
            "n_evals": int(len(g)),
        }
        row["robust_score"] = (
            0.5 * np.nan_to_num(row["oos_sharpe_mean"])
            + 0.3 * np.nan_to_num(row["oos_beat_bench_frac"])
            + 0.2 * np.nan_to_num(row["oos_mdd_better_frac"])
            + 0.2 * np.nan_to_num(row["bt5_sharpe_mean"])
            - 0.3 * np.nan_to_num(unstable)
            + 1.0 * np.nan_to_num(crisis)   # 위기 구간 초과수익(예: +0.10 = 벤치마크보다 10%p 덜 잃음)
        )
        out.append(row)
    return pd.DataFrame(out).sort_values("robust_score", ascending=False).reset_index(drop=True)


def main(argv=None) -> int:
    global WF_SPLITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5y + 핵심 유니버스 2개만 워크포워드")
    ap.add_argument("--strategies", default="", help="쉼표 구분. 기본 전체")
    ap.add_argument("--out", default=str(ROOT / "research" / "results"))
    ap.add_argument("--variants", action="store_true", help="VARIANTS(고정 파라미터 변형)도 포함")
    ap.add_argument("--only-variants", action="store_true", help="VARIANTS 만 평가")
    ap.add_argument("--variant-names", default="", help="쉼표 구분 변형 라벨 (기본 전부)")
    ap.add_argument("--fee", type=float, default=None, help="편도 수수료 강제 (기본: 한국 0.0015 / 미국 0.0005 자동)")
    ap.add_argument("--tag", default="", help="결과 파일 접미사 (예: fee2x)")
    ap.add_argument("--cash-rate", type=float, default=0.0, help="현금 연 이자율 (예: 0.03)")
    ap.add_argument("--min-trade", type=float, default=0.02, help="백테스트 최소 거래 비중 차이 (live threshold 와 동일, 예: 0.02)")
    ap.add_argument("--splits", type=int, default=WF_SPLITS, help="워크포워드 구간 수")
    ap.add_argument("--universes", default="", help="쉼표 구분 유니버스 이름 (기본 전부)")
    ap.add_argument("--holdout", action="store_true", help="개발에 쓰지 않은 홀드아웃 유니버스만 평가")
    ap.add_argument("--baseline", default="", help="저장된 리더보드 CSV 와 비교해 robust_score 하락을 경고 (예: research/results/leaderboard.csv)")
    ap.add_argument("--save-baseline", default="", help="이번 결과를 기준선 CSV 로 저장")
    ap.add_argument("--tolerance", type=float, default=0.03, help="기준선 대비 허용 하락폭")
    args = ap.parse_args(argv)
    names = [s for s in args.strategies.split(",") if s] or list(STRATEGIES)
    if args.only_variants:
        names = []
    cfg = BacktestConfig(fee_pct=args.fee if args.fee is not None else CFG.fee_pct,
                         slippage_pct=CFG.slippage_pct, min_trade_weight=args.min_trade, cash_rate=args.cash_rate)
    cfg._fee_override = args.fee  # None 이면 시장별 자동
    WF_SPLITS = args.splits
    pool = HOLDOUT_UNIVERSES if args.holdout else UNIVERSES
    unis = {u: t for u, t in pool.items() if not args.universes or u in args.universes.split(",")}
    if args.holdout and not args.tag:
        args.tag = "holdout"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("데이터 로드...")
    data = {u: get_prices(tk, period=WF_PERIOD) for u, tk in unis.items()}
    rows: list[dict] = []
    jobs = [(n, None, None) for n in names]
    if args.variants or args.only_variants:
        wanted = [v for v in args.variant_names.split(",") if v] or list(VARIANTS)
        jobs += [(label, STRATEGIES[sn], fixed) for label, (sn, fixed) in VARIANTS.items() if label in wanted]
    for n, vcls, fixed in jobs:
        t1 = time.time()
        r = evaluate_strategy(n, data, args.quick, cfg=cfg, fixed=fixed, grid=({} if vcls else None), cls=vcls)
        rows.extend(r)
        errs = [x for x in r if "error" in x]
        print(f"  {n:<24} {len(r):>3} evals  {time.time() - t1:5.1f}s" + (f"  오류 {len(errs)}: {errs[0]['error'][:60]}" if errs else ""))
    table = score_table(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"runs_{stamp}.json").write_text(json.dumps(rows, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    tag = args.tag or ("quick" if args.quick else "")
    suffix = f"_{tag}" if tag else ""
    lb_path = out_dir / f"leaderboard{suffix}.csv"
    table.to_csv(lb_path, index=False, encoding="utf-8-sig")
    # 상세(유니버스별) 표
    pd.DataFrame([r for r in rows if "error" not in r]).to_csv(out_dir / f"detail_latest{suffix}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    cols = ["strategy", "robust_score", "oos_sharpe_mean", "oos_sharpe_min", "oos_beat_bench_frac",
            "oos_mdd_better_frac", "bt5_sharpe_mean", "bt10_cagr_mean", "mdd_worst", "param_unstable_frac", "crisis_excess_mean"]
    print("\n=== 리더보드 (robust_score 내림차순) ===")
    print(table[cols].round(3).to_string(index=False))
    print(f"\n총 {len(rows)} evals, {time.time() - t0:.0f}s. 저장: {lb_path}")
    rc = 0
    if args.baseline:
        rc = compare_baseline(table, Path(args.baseline), args.tolerance)
    if args.save_baseline:
        table.to_csv(args.save_baseline, index=False, encoding="utf-8-sig")
        print(f"기준선 저장: {args.save_baseline}")
    return rc


def compare_baseline(table: pd.DataFrame, path: Path, tol: float) -> int:
    """기준선 대비 robust_score 가 tol 이상 떨어진 전략이 있으면 경고하고 1 반환 (회귀 가드)."""
    if not path.exists():
        print(f"기준선 파일이 없습니다: {path}")
        return 0
    base = pd.read_csv(path).set_index("strategy")["robust_score"]
    cur = table.set_index("strategy")["robust_score"]
    worse = [f"{n}: {b:.3f} → {cur[n]:.3f}" for n, b in base.items() if n in cur.index and cur[n] < b - tol]
    if worse:
        print("\n[회귀 경고] 기준선보다 robust_score 가 떨어진 전략:")
        for w in worse:
            print("  -", w)
        return 1
    print(f"\n회귀 없음 (허용 하락폭 {tol}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
