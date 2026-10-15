"""파라미터 탐색: grid_search 와 walk-forward 검증."""
from __future__ import annotations

import itertools
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .metrics import compute_metrics
from .strategies import STRATEGIES, Strategy
from .strategies.base import common_index

RESULT_METRICS = [
    "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
    "calmar", "win_rate", "num_trades", "exposure", "final_equity",
]


def _resolve_cls(strategy_cls: Any) -> type[Strategy]:
    if isinstance(strategy_cls, str):
        return STRATEGIES[strategy_cls]
    if isinstance(strategy_cls, Strategy):
        return type(strategy_cls)
    return strategy_cls


def _combos(param_grid: dict[str, list]) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys))]


def _metric_value(metrics: dict, metric: str) -> float:
    v = metrics.get(metric, float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def grid_search(
    strategy_cls: Any,
    prices: dict[str, pd.DataFrame],
    param_grid: dict[str, list],
    config: BacktestConfig | None = None,
    metric: str = "sharpe",
    start: str | pd.Timestamp | None = None,
    progress: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """모든 파라미터 조합을 백테스트해 metric 내림차순으로 정렬한 표를 반환한다.

    잘못된 조합(예: fast >= slow)은 건너뛴다. 컬럼: 파라미터들 + 주요 지표.
    """
    cls = _resolve_cls(strategy_cls)
    cfg = config or BacktestConfig()
    combos = _combos(param_grid)
    rows: list[dict] = []
    for k, params in enumerate(combos):
        try:
            strat = cls(**params)
        except ValueError:
            continue
        res = run_backtest(strat, prices, cfg, start=start)
        row = dict(strat.p)
        row.update({key: res.metrics.get(key, float("nan")) for key in RESULT_METRICS})
        if metric not in row:
            row[metric] = res.metrics.get(metric, float("nan"))
        rows.append(row)
        if progress is not None:
            progress((k + 1) / len(combos))
    cols = [p.name for p in cls.params] + [m for m in RESULT_METRICS]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    sort_key = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.assign(_k=sort_key).sort_values("_k", ascending=False, na_position="last")
    return df.drop(columns="_k").reset_index(drop=True)


def walk_forward(
    strategy_cls: Any,
    prices: dict[str, pd.DataFrame],
    param_grid: dict[str, list],
    config: BacktestConfig | None = None,
    n_splits: int = 4,
    train_ratio: float = 0.7,
    metric: str = "sharpe",
) -> dict:
    """전체 기간을 n_splits 개 구간으로 나누고, 각 구간 앞부분(train)에서 최적 파라미터를 찾은 뒤
    뒷부분(test, 처음 보는 데이터)에서 성과를 측정한다.

    각 구간의 학습/검증 백테스트는 그 시점 이전 데이터만 사용한다 (이전 데이터는 지표 워밍업용).
    """
    cls = _resolve_cls(strategy_cls)
    cfg = config or BacktestConfig()
    if n_splits < 1:
        raise ValueError("n_splits 는 1 이상이어야 합니다.")
    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError("train_ratio 는 0.1~0.95 사이여야 합니다.")
    idx = common_index(prices)
    fold_len = len(idx) // n_splits
    if fold_len < 40:
        raise ValueError("데이터가 너무 짧아 구간을 나눌 수 없습니다. 기간을 늘리거나 구간 수를 줄이세요.")

    def upto(end: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {t: df.loc[df.index <= end] for t, df in prices.items()}

    folds: list[dict] = []
    oos_parts: list[pd.Series] = []
    bench_parts: list[pd.Series] = []
    trade_parts: list[pd.DataFrame] = []
    for k in range(n_splits):
        lo = k * fold_len
        hi = (k + 1) * fold_len if k < n_splits - 1 else len(idx)
        n_train = max(int((hi - lo) * train_ratio), 2)
        train_start, train_end = idx[lo], idx[lo + n_train - 1]
        test_start, test_end = idx[lo + n_train], idx[hi - 1]

        table = grid_search(cls, upto(train_end), param_grid, cfg, metric=metric, start=train_start)
        if table.empty:
            raise ValueError("유효한 파라미터 조합이 없습니다.")
        best = table.iloc[0]
        raw = {p.name: _py(best[p.name]) for p in cls.params if p.name in table.columns}
        best_params = dict(cls(**raw).p)  # 기본값 타입(int/float)으로 정규화
        train_metric = _metric_value(best.to_dict(), metric)

        test_res = run_backtest(cls(**best_params), upto(test_end), cfg, start=test_start)
        test_metric = _metric_value(test_res.metrics, metric)
        folds.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "best_params": best_params,
            "train_metric": train_metric, "test_metric": test_metric,
        })
        oos_parts.append(test_res.equity)
        bench_parts.append(test_res.benchmark_equity)
        trade_parts.append(test_res.trades)

    init = float(cfg.initial_cash)
    oos_equity = _chain(oos_parts, init).rename("oos_equity")
    oos_bench = _chain(bench_parts, init).rename("benchmark")
    nonempty = [t for t in trade_parts if len(t)]
    trades = pd.concat(nonempty, ignore_index=True) if nonempty else trade_parts[0].iloc[0:0]
    oos_metrics = compute_metrics(oos_equity, trades, benchmark=oos_bench)

    return {
        "folds": folds,
        "oos_equity": oos_equity,
        "oos_metrics": oos_metrics,
        "warning": _overfit_warning(folds, oos_metrics, metric),
    }


def _py(v: Any) -> Any:
    """numpy 스칼라를 파이썬 기본형으로."""
    return v.item() if isinstance(v, np.generic) else v


def _chain(parts: list[pd.Series], init: float) -> pd.Series:
    """구간별 자산곡선(각각 init 에서 시작)을 이어 붙여 하나의 연속 곡선으로 만든다."""
    out: list[pd.Series] = []
    level = init
    for s in parts:
        rel = s / init * level
        out.append(rel)
        level = float(rel.iloc[-1])
    res = pd.concat(out)
    return res[~res.index.duplicated(keep="first")]


def _overfit_warning(folds: list[dict], oos_metrics: dict, metric: str) -> str:
    """학습 구간(in-sample)과 검증 구간(out-of-sample) 샤프 지수를 비교한 한국어 경고."""
    is_vals = [f["train_metric"] for f in folds if math.isfinite(f["train_metric"])]
    is_avg = float(np.mean(is_vals)) if is_vals else float("nan")
    if metric == "sharpe":
        oos = oos_metrics.get("sharpe", float("nan"))
    else:
        vals = [f["test_metric"] for f in folds if math.isfinite(f["test_metric"])]
        oos = float(np.mean(vals)) if vals else float("nan")
    name = "샤프 지수" if metric == "sharpe" else metric
    if not (math.isfinite(is_avg) and math.isfinite(oos)):
        return "검증 결과를 계산할 수 없습니다. 기간을 늘려 다시 시도하세요."
    base = f"학습 구간 평균 {name} {is_avg:.2f} → 처음 보는 검증 구간 {name} {oos:.2f}. "
    params_set = {tuple(sorted(f["best_params"].items())) for f in folds}
    unstable = len(params_set) == len(folds) and len(folds) >= 3
    if is_avg > 0 and (oos < 0 or oos < 0.5 * is_avg):
        msg = base + (
            "[주의] 과최적화(오버피팅) 의심: 과거 데이터에 맞춘 파라미터가 새로운 기간에서는 훨씬 나빴습니다. "
            "백테스트 최고 성과를 그대로 믿지 말고, 기본값처럼 단순한 설정을 쓰는 것이 안전합니다."
        )
    elif oos < 0:
        msg = base + "학습/검증 구간 모두 성과가 좋지 않습니다. 이 종목에는 이 전략이 맞지 않을 수 있습니다."
    else:
        msg = base + "검증 구간에서도 성과가 크게 무너지지 않아 과최적화 위험이 상대적으로 낮아 보입니다."
    if unstable:
        msg += " 다만 구간마다 최적 파라미터가 모두 달라 결과가 불안정할 수 있습니다."
    return msg
