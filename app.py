"""quant-trader Streamlit 대시보드 (한국어). 실행: streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from quant.data import get_prices  # noqa: E402
from quant.strategies import STRATEGIES, get_strategy  # noqa: E402
from quant.backtest import BacktestConfig, run_backtest  # noqa: E402
from quant.metrics import describe_metrics_korean  # noqa: E402
from quant.optimize import grid_search, walk_forward  # noqa: E402
from quant.signals import today_signals, ACTION_KO  # noqa: E402
from quant.paper import PaperAccount  # noqa: E402

DISCLAIMER = "투자 조언이 아닙니다. 과거 성과는 미래 수익을 보장하지 않으며, 모든 투자 판단과 손실의 책임은 본인에게 있습니다."

PRESETS = {
    "미국 ETF": "SPY,QQQ,TLT,GLD",
    "한국 대형주": "005930.KS,000660.KS,035420.KS,005380.KS",
    "한국 ETF": "069500.KS,229200.KS,114260.KS",
}
REBAL = {"매일": "daily", "매주": "weekly", "매월": "monthly"}

METRIC_LABELS = {
    "total_return": "총 수익률", "cagr": "연평균 수익률(CAGR)", "annual_vol": "연 변동성",
    "sharpe": "샤프지수", "sortino": "소르티노지수", "max_drawdown": "최대 낙폭(MDD)",
    "max_drawdown_duration_days": "최대 낙폭 지속일", "calmar": "칼마지수", "win_rate": "승률",
    "num_trades": "거래 횟수", "profit_factor": "손익비", "avg_trade_return": "거래당 평균 수익률",
    "exposure": "시장 노출 비율", "best_month": "최고의 달", "worst_month": "최악의 달",
    "final_equity": "최종 자산",
}
PCT_KEYS = {"total_return", "cagr", "annual_vol", "max_drawdown", "win_rate", "avg_trade_return",
            "exposure", "best_month", "worst_month"}

st.set_page_config(page_title="퀀트 트레이더", page_icon="📈", layout="wide")


# ---------- 공통 헬퍼 ----------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple, period: str):
    return get_prices(list(tickers), period=period)


def fmt(k: str, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if k in PCT_KEYS:
        return f"{v:.1%}"
    if k == "final_equity":
        return f"{v:,.0f}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def build_strategy(name: str, params: dict):
    return get_strategy(name, **params)


def get_data(tickers: list[str], period: str):
    if not tickers:
        st.error("종목을 한 개 이상 입력해 주세요.")
        return None
    try:
        with st.spinner("가격 데이터를 불러오는 중..."):
            return fetch_prices(tuple(tickers), period)
    except Exception as e:  # noqa: BLE001
        st.error(f"가격 데이터를 불러오지 못했습니다: {e}")
        return None


def last_prices(prices) -> dict:
    return {t: float(df["close"].iloc[-1]) for t, df in prices.items()}


def line_fig(series_dict: dict, title: str, log=False, pct=False):
    fig = go.Figure()
    for name, s in series_dict.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines"))
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.1), hovermode="x unified")
    if log:
        fig.update_yaxes(type="log")
    if pct:
        fig.update_yaxes(tickformat=".0%")
    return fig


# ---------- 사이드바 ----------
if "tickers" not in st.session_state:
    st.session_state["tickers"] = PRESETS["미국 ETF"]


def _set_preset(v: str):
    st.session_state["tickers"] = v


with st.sidebar:
    st.title("📈 퀀트 트레이더")
    st.text_input("종목 (쉼표로 구분)", key="tickers",
                  help="미국: SPY, QQQ / 한국: 005930.KS (코스피), 035720.KQ (코스닥)")
    pc = st.columns(3)
    for col, (label, v) in zip(pc, PRESETS.items()):
        col.button(label, on_click=_set_preset, args=(v,), width="stretch")
    period = st.selectbox("기간", ["1y", "3y", "5y", "10y"], index=2)

    names = list(STRATEGIES.keys())
    sname = st.selectbox("전략", names, format_func=lambda n: STRATEGIES[n].label)
    scls = STRATEGIES[sname]
    st.caption(scls.description)

    params: dict = {}
    for p in scls.params:
        key = f"param_{sname}_{p.name}"
        is_int = isinstance(p.default, int) and not isinstance(p.default, bool)
        if isinstance(p.default, bool):
            params[p.name] = st.checkbox(p.name, value=p.default, key=key, help=p.help or None)
        elif is_int:
            params[p.name] = int(st.number_input(
                p.name, value=int(p.default),
                min_value=int(p.min) if p.min is not None else None,
                max_value=int(p.max) if p.max is not None else None,
                step=int(p.step) if p.step else 1, key=key, help=p.help or None))
        elif isinstance(p.default, float):
            params[p.name] = float(st.number_input(
                p.name, value=float(p.default),
                min_value=float(p.min) if p.min is not None else None,
                max_value=float(p.max) if p.max is not None else None,
                step=float(p.step) if p.step else 0.01, format="%.4f", key=key, help=p.help or None))
        else:
            params[p.name] = st.text_input(p.name, value=str(p.default), key=key, help=p.help or None)

    fee = st.number_input("수수료+세금 (편도, %)", 0.0, 5.0, 0.15, 0.01, format="%.3f") / 100
    use_sl = st.checkbox("손절 사용", value=False)
    stop_loss = st.slider("손절 비율 (%)", 1, 50, 10) / 100 if use_sl else None
    rebal_label = st.selectbox("리밸런스 주기", list(REBAL.keys()))
    rebalance = REBAL[rebal_label]

    st.divider()
    st.caption("⚠️ " + DISCLAIMER)

tickers = [t.strip().upper() for t in st.session_state["tickers"].split(",") if t.strip()]
config = BacktestConfig(fee_pct=fee, stop_loss_pct=stop_loss, rebalance=rebalance)

st.title("퀀트 트레이더 대시보드")
tabs = st.tabs(["① 백테스트", "② 파라미터 탐색", "③ 오늘의 신호", "④ 페이퍼 계좌", "⑤ 도움말"])

# ---------- ① 백테스트 ----------
with tabs[0]:
    st.subheader("백테스트: 과거에 이 전략을 썼다면?")
    log_scale = st.checkbox("로그 스케일로 보기", value=False)
    if st.button("백테스트 실행", type="primary"):
        prices = get_data(tickers, period)
        if prices:
            try:
                with st.spinner("백테스트 계산 중..."):
                    st.session_state["bt"] = run_backtest(build_strategy(sname, params), prices, config)
            except Exception as e:  # noqa: BLE001
                st.error(f"백테스트 중 오류가 발생했습니다: {e}")
    res = st.session_state.get("bt")
    if res is not None:
        st.plotly_chart(line_fig({"전략": res.equity, "벤치마크(균등 보유)": res.benchmark_equity},
                                 "자산 곡선", log=log_scale), width="stretch")
        dd = res.equity / res.equity.cummax() - 1
        bdd = res.benchmark_equity / res.benchmark_equity.cummax() - 1
        st.plotly_chart(line_fig({"전략": dd, "벤치마크": bdd}, "낙폭(고점 대비 하락률)", pct=True),
                        width="stretch")
        m = res.metrics
        keys = ["total_return", "cagr", "sharpe", "max_drawdown", "annual_vol", "win_rate",
                "num_trades", "profit_factor", "exposure", "final_equity"]
        keys = [k for k in keys if k in m]
        for i in range(0, len(keys), 5):
            cols = st.columns(5)
            for c, k in zip(cols, keys[i:i + 5]):
                c.metric(METRIC_LABELS.get(k, k), fmt(k, m[k]))
        if "benchmark_total_return" in m:
            st.caption(f"벤치마크 총 수익률 {fmt('total_return', m['benchmark_total_return'])}, "
                       f"CAGR {fmt('cagr', m.get('benchmark_cagr'))}, "
                       f"MDD {fmt('max_drawdown', m.get('benchmark_max_drawdown'))}")
        st.markdown("#### 쉬운 말 해석")
        st.markdown("\n".join(f"- {s}" for s in describe_metrics_korean(m)))
        st.markdown("#### 거래 내역")
        if len(res.trades):
            st.dataframe(res.trades, width="stretch")
        else:
            st.info("거래가 없었습니다.")
        eq = res.equity.rename("equity").to_frame()
        eq["benchmark"] = res.benchmark_equity
        st.download_button("자산 곡선 CSV 다운로드", eq.to_csv().encode("utf-8-sig"),
                           file_name=f"equity_{sname}.csv", mime="text/csv")
    else:
        st.info("왼쪽에서 종목과 전략을 고른 뒤 '백테스트 실행'을 눌러 보세요.")

# ---------- ② 파라미터 탐색 ----------
with tabs[1]:
    st.subheader("파라미터 탐색")
    st.warning("⚠️ 과거 데이터에 가장 잘 맞는 값을 고르면 '과최적화'가 되어, 미래에는 오히려 성과가 나쁠 수 있습니다. "
               "아래 워크포워드 결과(처음 보는 구간에서의 성과)를 꼭 함께 확인하세요.")
    plist = scls.params
    if not plist:
        st.info("이 전략은 조정할 파라미터가 없습니다.")
    else:
        pnames = [p.name for p in plist]
        chosen = st.multiselect("탐색할 파라미터 (1~2개)", pnames, default=pnames[:1], max_selections=2)
        pmeta = {p.name: p for p in plist}
        grid: dict[str, list] = {}
        for n in chosen:
            p = pmeta[n]
            is_int = isinstance(p.default, int) and not isinstance(p.default, bool)
            c1, c2, c3 = st.columns(3)
            dmin = p.min if p.min is not None else p.default / 2
            dmax = p.max if p.max is not None else p.default * 2
            dstep = p.step if p.step else (1 if is_int else (dmax - dmin) / 5 or 0.1)
            if is_int:
                lo = c1.number_input(f"{n} 최소", value=int(dmin), step=int(dstep), key=f"lo_{n}")
                hi = c2.number_input(f"{n} 최대", value=int(dmax), step=int(dstep), key=f"hi_{n}")
                stp = c3.number_input(f"{n} 간격", value=int(max(dstep, 1)), min_value=1, key=f"st_{n}")
                vals = list(range(int(lo), int(hi) + 1, int(stp)))
            else:
                lo = c1.number_input(f"{n} 최소", value=float(dmin), format="%.4f", key=f"lo_{n}")
                hi = c2.number_input(f"{n} 최대", value=float(dmax), format="%.4f", key=f"hi_{n}")
                stp = c3.number_input(f"{n} 간격", value=float(dstep), min_value=1e-6, format="%.4f", key=f"st_{n}")
                vals = [round(float(v), 6) for v in np.arange(lo, hi + stp / 2, stp)]
            grid[n] = vals[:30]
            if len(vals) > 30:
                st.caption(f"{n}: 값이 너무 많아 처음 30개만 사용합니다.")
        total_combo = int(np.prod([len(v) for v in grid.values()])) if grid else 0
        metric = st.selectbox("평가 기준", ["sharpe", "cagr", "calmar", "total_return"],
                              format_func=lambda k: METRIC_LABELS.get(k, k))
        st.caption(f"조합 수: {total_combo}")
        b1, b2 = st.columns(2)
        run_grid = b1.button("그리드 탐색 실행", disabled=not grid)
        run_wf = b2.button("워크포워드 검증 실행", disabled=not grid)
        if run_grid or run_wf:
            prices = get_data(tickers, period)
            if prices:
                try:
                    with st.spinner("계산 중... (조합이 많으면 시간이 걸립니다)"):
                        if run_grid:
                            st.session_state["grid"] = (grid, metric, grid_search(scls, prices, grid, config, metric=metric))
                        if run_wf:
                            st.session_state["wf"] = walk_forward(scls, prices, grid, config)
                except Exception as e:  # noqa: BLE001
                    st.error(f"탐색 중 오류가 발생했습니다: {e}")
        if "grid" in st.session_state:
            g_grid, g_metric, gdf = st.session_state["grid"]
            st.markdown("#### 그리드 탐색 결과")
            st.dataframe(gdf, width="stretch")
            gk = [k for k in g_grid if k in gdf.columns]
            if len(gk) == 2 and g_metric in gdf.columns:
                pv = gdf.pivot_table(index=gk[1], columns=gk[0], values=g_metric)
                fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale="RdYlGn",
                                           colorbar=dict(title=METRIC_LABELS.get(g_metric, g_metric))))
                fig.update_layout(title="파라미터 히트맵 (초록일수록 좋음)", xaxis_title=gk[0],
                                  yaxis_title=gk[1], height=420)
                st.plotly_chart(fig, width="stretch")
                st.caption("초록 영역이 넓고 고르게 퍼져 있으면 안정적입니다. 한 점만 튀면 우연일 가능성이 큽니다.")
        if "wf" in st.session_state:
            wf = st.session_state["wf"]
            st.markdown("#### 워크포워드 결과")
            if isinstance(wf, dict):
                warn = wf.get("warning") or wf.get("warnings") or wf.get("message")
                if warn:
                    st.error("⚠️ " + (warn if isinstance(warn, str) else " ".join(map(str, warn))))
                for k, v in wf.items():
                    if k in ("warning", "warnings", "message"):
                        continue
                    if isinstance(v, pd.DataFrame):
                        st.markdown(f"**{k}**")
                        st.dataframe(v, width="stretch")
                    elif isinstance(v, pd.Series) and len(v) > 1:
                        st.plotly_chart(line_fig({k: v}, "검증 구간(OOS) 자산 곡선"), width="stretch")
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        st.markdown(f"**{k}**")
                        st.dataframe(pd.DataFrame(v), width="stretch")
                    elif isinstance(v, (int, float, str)):
                        st.write(f"{k}: {v}")
                    elif isinstance(v, dict):
                        st.markdown(f"**{k}**")
                        st.json({str(a): (b if isinstance(b, (int, float, str)) else str(b)) for a, b in v.items()})
            else:
                st.write(wf)
            st.error("⚠️ 워크포워드에서도 성과가 크게 떨어지면 그 전략은 믿기 어렵습니다. 어떤 결과도 미래를 보장하지 않습니다.")

# ---------- ③ 오늘의 신호 ----------
ACTION_COLOR = {"BUY": "#d9f2d9", "SELL": "#f8d7da", "HOLD": "#fff3cd", "NONE": "#eeeeee"}
with tabs[2]:
    st.subheader("오늘의 신호")
    if st.button("신호 계산", type="primary"):
        prices = get_data(tickers, "2y" if period == "1y" else period)
        if prices:
            try:
                acc = PaperAccount()
                cur = {}
                if acc.exists():
                    cur = {t: d["weight"] for t, d in acc.value(last_prices(prices))["positions"].items()}
                st.session_state["sigs"] = today_signals(build_strategy(sname, params), prices, cur)
            except Exception as e:  # noqa: BLE001
                st.error(f"신호 계산 중 오류가 발생했습니다: {e}")
    sigs = st.session_state.get("sigs")
    if sigs:
        df = pd.DataFrame([{
            "종목": s.ticker, "신호": ACTION_KO.get(s.action, s.action), "_a": s.action,
            "목표 비중": s.target_weight, "현재 비중": s.current_weight, "종가": s.price, "이유": s.reason,
        } for s in sigs])
        styled = (df.drop(columns="_a").style
                  .apply(lambda r: [f"background-color: {ACTION_COLOR[df.loc[r.name, '_a']]}; color: #222"] * len(r), axis=1)
                  .format({"목표 비중": "{:.0%}", "현재 비중": "{:.0%}", "종가": "{:,.2f}"}))
        st.dataframe(styled, width="stretch")
        for s in sigs:
            icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "NONE": "⚪"}[s.action]
            st.markdown(f"{icon} **{s.ticker}** ({ACTION_KO.get(s.action)}): {s.reason}")
        st.caption("현재 비중은 페이퍼 계좌 기준입니다. 신호는 어제까지의 종가로 계산한 참고용 정보입니다.")
    else:
        st.info("'신호 계산'을 눌러 보세요.")

# ---------- ④ 페이퍼 계좌 ----------
with tabs[3]:
    st.subheader("페이퍼 계좌 (가상 투자)")
    st.caption("실제 돈이 오가지 않는 연습용 계좌입니다. 실제 주문은 전송되지 않습니다.")
    acc = PaperAccount()
    if not acc.exists():
        cash = st.number_input("초기 자금", 1_000_000, 10_000_000_000, 10_000_000, 1_000_000)
        if st.button("계좌 만들기", type="primary"):
            acc.reset(float(cash))
            st.rerun()
    else:
        prices_cur = {}
        px_data = None
        if tickers or acc.positions:
            try:
                px_data = fetch_prices(tuple(sorted(set(tickers) | set(acc.positions))), "1y")
                prices_cur = last_prices(px_data)
            except Exception as e:  # noqa: BLE001
                st.warning(f"현재가를 불러오지 못해 평균 매수가로 평가합니다: {e}")
        v = acc.value(prices_cur)
        c = st.columns(4)
        c[0].metric("총자산", f"{v['total']:,.0f}")
        c[1].metric("현금", f"{v['cash']:,.0f}")
        c[2].metric("총 손익", f"{v['pnl']:+,.0f}", f"{v['pnl_pct']:+.2%}")
        c[3].metric("실현 손익", f"{v['realized_pnl']:+,.0f}")
        if v["positions"]:
            pdf = pd.DataFrame([{
                "종목": t, "수량": d["qty"], "평균단가": d["avg_price"], "현재가": d["price"],
                "평가금액": d["value"], "비중": d["weight"], "손익": d["pnl"], "수익률": d["pnl_pct"],
            } for t, d in v["positions"].items()])
            st.dataframe(pdf.style.format({"평균단가": "{:,.2f}", "현재가": "{:,.2f}", "평가금액": "{:,.0f}",
                                           "비중": "{:.1%}", "손익": "{:+,.0f}", "수익률": "{:+.2%}"}),
                         width="stretch")
        else:
            st.info("보유 종목이 없습니다.")

        if st.button("신호대로 리밸런스", type="primary"):
            if not tickers:
                st.error("종목을 입력해 주세요.")
            else:
                try:
                    prices = get_data(tickers, "2y" if period == "1y" else period)
                    if prices:
                        px = last_prices(prices)
                        v2 = acc.value(px)
                        cur_w = {t: d["weight"] for t, d in v2["positions"].items()}
                        sg = today_signals(build_strategy(sname, params), prices, cur_w)
                        last_date = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
                        orders = acc.rebalance_to({s.ticker: s.target_weight for s in sg}, px, fee, date=last_date)
                        acc.snapshot({**prices_cur, **px}, date=last_date)
                        acc.save()
                        if orders:
                            st.success(f"{len(orders)}건 주문을 실행했습니다.")
                            st.dataframe(pd.DataFrame(orders), width="stretch")
                        else:
                            st.info("이미 목표 비중에 가까워 실행할 주문이 없습니다.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"리밸런스 중 오류가 발생했습니다: {e}")
        if acc.history:
            h = pd.DataFrame(acc.history)
            h["date"] = pd.to_datetime(h["date"])
            st.plotly_chart(line_fig({"총자산": h.set_index("date")["total"]}, "자산 추이"), width="stretch")
        if acc.orders:
            with st.expander("전체 주문 내역"):
                st.dataframe(pd.DataFrame(acc.orders[::-1]), width="stretch")

        with st.expander("계좌 초기화"):
            new_cash = st.number_input("새 초기 자금", 1_000_000, 10_000_000_000, int(acc.initial_cash), 1_000_000)
            ok = st.checkbox("모든 보유 종목과 기록이 삭제되는 것을 이해했습니다.")
            if st.button("초기화", disabled=not ok):
                acc.reset(float(new_cash))
                st.success("초기화했습니다.")
                st.rerun()

# ---------- ⑤ 도움말 ----------
with tabs[4]:
    st.subheader("도움말")
    st.markdown("### 전략 설명")
    for n, cls in STRATEGIES.items():
        st.markdown(f"- **{cls.label}** (`{n}`): {cls.description}")
    st.markdown("""
### 용어 설명
- **백테스트**: 과거 데이터로 "이 규칙대로 거래했다면 결과가 어땠을까"를 계산해 보는 것입니다. 연습 문제 풀이와 같아서 실제 시험(미래) 점수를 보장하지는 않습니다.
- **CAGR(연평균 수익률)**: 매년 같은 비율로 불어났다고 가정했을 때의 연 수익률입니다.
- **샤프지수**: 위험(등락 폭) 한 단위당 얻은 수익입니다. 보통 1 이상이면 괜찮고, 2 이상이면 매우 좋지만 의심도 해봐야 합니다.
- **MDD(최대 낙폭)**: 최고점에서 최저점까지 가장 크게 떨어진 비율입니다. -30%라면 한때 자산이 30% 줄었다는 뜻이며, 그 시기를 버틸 수 있어야 합니다.
- **과최적화**: 과거 데이터에만 지나치게 맞춘 설정입니다. 과거엔 훌륭하지만 미래엔 통하지 않습니다.
- **워크포워드**: 과거 구간에서 설정을 고른 뒤, 그 설정을 '처음 보는 다음 구간'에서 시험하는 방식입니다. 과최적화를 걸러내는 데 도움이 됩니다.
- **리밸런싱**: 목표 비중에 맞춰 보유 종목을 사고파는 조정 작업입니다.
- **손절**: 매수가보다 일정 비율 이상 떨어지면 더 큰 손실을 막기 위해 팔아 버리는 규칙입니다.

### 결과 읽는 법
1. 벤치마크(그냥 들고 있기)보다 **MDD가 얼마나 작은지**, 수익이 비슷한지 먼저 봅니다.
2. 거래 횟수가 너무 적으면(수 회) 우연일 수 있습니다.
3. 수수료를 올려도 성과가 유지되는지 확인합니다.
4. 파라미터를 조금 바꿔도 결과가 비슷해야 믿을 만합니다.
5. 페이퍼 계좌로 몇 주~몇 달 지켜본 뒤 판단하세요.
""")
    st.error("⚠️ " + DISCLAIMER)
