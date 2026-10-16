"""quant-trader CLI. 사용 예: python cli.py backtest --tickers SPY --strategy sma_cross"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DISCLAIMER = "※ 투자 조언이 아니며, 과거 성과가 미래 수익을 보장하지 않습니다."


def _tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def _parse_params(strategy_cls, items: list[str] | None) -> dict:
    """--param key=value 를 전략 Param 기본값 타입에 맞춰 변환."""
    meta = {p.name: p for p in getattr(strategy_cls, "params", [])}
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param 형식 오류: '{it}' (예: fast=20)")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in meta:
            raise SystemExit(f"'{k}'는 이 전략의 파라미터가 아닙니다. 가능: {', '.join(meta) or '없음'}")
        d = meta[k].default
        try:
            if isinstance(d, bool):
                out[k] = v.lower() in ("1", "true", "yes", "y")
            elif isinstance(d, int):
                out[k] = int(float(v))
            elif isinstance(d, float):
                out[k] = float(v)
            else:
                out[k] = v
        except ValueError:
            raise SystemExit(f"--param {k}의 값 '{v}'을(를) 숫자로 읽을 수 없습니다.")
    return out


def _load(args):
    from quant.data import get_prices
    from quant.strategies import STRATEGIES, get_strategy

    if args.strategy not in STRATEGIES:
        raise SystemExit(f"알 수 없는 전략: {args.strategy}. 가능: {', '.join(STRATEGIES)}")
    cls = STRATEGIES[args.strategy]
    params = _parse_params(cls, getattr(args, "param", None))
    strategy = get_strategy(args.strategy, **params)
    prices = get_prices(_tickers(args.tickers), period=getattr(args, "period", "2y"))
    return strategy, prices, params


def _fmt_metric(k, v):
    pct_keys = {"total_return", "cagr", "annual_vol", "max_drawdown", "win_rate", "avg_trade_return",
                "exposure", "best_month", "worst_month", "benchmark_total_return",
                "benchmark_cagr", "benchmark_max_drawdown"}
    if v is None:
        return "-"
    if k in pct_keys:
        return f"{v:.2%}"
    if k == "final_equity":
        return f"{v:,.0f}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


METRIC_LABELS = {
    "total_return": "총 수익률", "cagr": "연평균 수익률(CAGR)", "annual_vol": "연 변동성",
    "sharpe": "샤프지수", "sortino": "소르티노", "max_drawdown": "최대 낙폭(MDD)",
    "max_drawdown_duration_days": "최대 낙폭 지속일", "calmar": "칼마", "win_rate": "승률",
    "num_trades": "거래 횟수", "profit_factor": "손익비(PF)", "avg_trade_return": "거래당 평균 수익",
    "exposure": "시장 노출 비율", "best_month": "최고의 달", "worst_month": "최악의 달",
    "final_equity": "최종 자산", "benchmark_total_return": "벤치마크 총 수익률",
    "benchmark_cagr": "벤치마크 CAGR", "benchmark_max_drawdown": "벤치마크 MDD",
}


def cmd_backtest(args):
    from quant.backtest import BacktestConfig, run_backtest
    from quant.metrics import describe_metrics_korean

    strategy, prices, params = _load(args)
    cfg = BacktestConfig(fee_pct=args.fee, stop_loss_pct=args.stop_loss, rebalance=args.rebalance,
                         min_trade_weight=args.min_trade, cash_rate=args.cash_rate)
    res = run_backtest(strategy, prices, cfg)
    print(f"\n=== 백테스트: {strategy.label} ({args.strategy}) / {args.tickers} / {args.period} ===")
    if params:
        print("파라미터:", params)
    print(f"{'지표':<24}{'값':>16}")
    print("-" * 42)
    for k, v in res.metrics.items():
        print(f"{METRIC_LABELS.get(k, k):<{24 - sum(1 for c in METRIC_LABELS.get(k, k) if ord(c) > 127)}}{_fmt_metric(k, v):>16}")
    print("\n[쉬운 설명]")
    for line in describe_metrics_korean(res.metrics):
        print(" -", line)
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    name = f"equity_{args.strategy}_{args.tickers.replace(',', '-')}_{args.period}.csv"
    path = out_dir / name
    df = res.equity.rename("equity").to_frame()
    df["benchmark"] = res.benchmark_equity
    df.to_csv(path, encoding="utf-8-sig")
    print(f"\n자산 곡선 저장: {path}")
    print(DISCLAIMER)


def cmd_signal(args):
    from quant.signals import format_signals_korean, today_signals

    strategy, prices, _ = _load(args)
    print(f"\n=== 오늘의 신호: {strategy.label} ===")
    print(format_signals_korean(today_signals(strategy, prices)))
    print(DISCLAIMER)


def _last_prices(prices) -> dict:
    return {t: float(df["close"].iloc[-1]) for t, df in prices.items()}


def cmd_paper(args):
    from quant.paper import PaperAccount

    if args.paper_cmd == "init":
        acc = PaperAccount(args.file, args.cash)
        acc.reset(args.cash)
        print(f"페이퍼 계좌를 초기화했습니다. 초기 자금 {args.cash:,.0f} ({acc.path})")
        return
    acc = PaperAccount(args.file)
    if not acc.exists():
        raise SystemExit("계좌가 없습니다. 먼저 `python cli.py paper init --cash 10000000`을 실행하세요.")

    if args.paper_cmd == "status":
        from quant.data import get_prices

        tk = _tickers(args.tickers) if args.tickers else list(acc.positions)
        px = _last_prices(get_prices(tk, period="1y")) if tk else {}
        _print_status(acc.value(px))
        return

    if args.paper_cmd == "run":
        from quant.signals import format_signals_korean, today_signals

        strategy, prices, _ = _load(args)
        px = _last_prices(prices)
        v = acc.value(px)
        cur_w = {t: d["weight"] for t, d in v["positions"].items()}
        sigs = today_signals(strategy, prices, cur_w)
        print(format_signals_korean(sigs))
        targets = {s.ticker: s.target_weight for s in sigs}
        last_date = max(df.index[-1] for df in prices.values()).strftime("%Y-%m-%d")
        orders = acc.rebalance_to(targets, px, args.fee, date=last_date)
        acc.snapshot(px, date=last_date)
        acc.save()
        print("\n[실행된 주문]")
        if not orders:
            print(" 없음 (이미 목표 비중에 가깝습니다)")
        for o in orders:
            side = "매수" if o["side"] == "BUY" else "매도"
            print(f" {o['date']} {side} {o['ticker']} {o['qty']}주 @ {o['price']:,.2f} "
                  f"(금액 {o['value']:,.0f}, 수수료 {o['fee']:,.0f})")
        _print_status(acc.value(px))
        print(DISCLAIMER)


def _print_status(v: dict):
    print("\n[계좌 현황]")
    print(f" 총자산 {v['total']:,.0f} / 현금 {v['cash']:,.0f} / 평가금액 {v['positions_value']:,.0f}")
    print(f" 손익 {v['pnl']:+,.0f} ({v['pnl_pct']:+.2%}) / 실현손익 {v['realized_pnl']:+,.0f}")
    for t, d in v["positions"].items():
        print(f"  - {t}: {d['qty']}주, 평균 {d['avg_price']:,.2f}, 현재 {d['price']:,.2f}, "
              f"비중 {d['weight']:.1%}, 손익 {d['pnl_pct']:+.2%}")


def _live_apply_overrides(cfg, args):
    if args.tickers:
        cfg.tickers = _tickers(args.tickers)
    if args.strategy:
        cfg.strategy = args.strategy
    if getattr(args, "broker", None):
        cfg.broker = args.broker
    return cfg


def _live_kill_switch(cmd: str, kill) -> None:
    if cmd == "stop":
        kill.write_text("stopped by cli\n", encoding="utf-8")
        print(f"킬 스위치를 켰습니다. 이 파일이 있는 동안 어떤 주문도 나가지 않습니다: {kill}")
    elif kill.exists():
        kill.unlink()
        print("킬 스위치를 껐습니다. 다음 실행부터 주문이 다시 나갑니다.")
    else:
        print("킬 스위치가 꺼져 있습니다.")


def _live_status(cfg, kill, limit: int) -> None:
    from quant.config import PROJECT_ROOT
    from quant.live import read_logs

    print(f"브로커 {cfg.broker} / 전략 {cfg.strategy} / 종목 {','.join(cfg.tickers)} / 킬스위치 {'ON' if kill.exists() else 'off'}")
    for r in read_logs(PROJECT_ROOT / cfg.log_dir, limit):
        n_ok = sum(1 for x in r.get("results", []) if x.get("ok"))
        print(f" {r['time']} {r['broker']:<10} {'dry-run' if r['dry_run'] else 'EXECUTE'} "
              f"계획 {len(r.get('planned', []))} / 체결요청 {n_ok} / 거부 {len(r.get('rejected', []))}"
              f"{' / 오류: ' + r['error'] if r.get('error') else ''}")
        for n in r.get("notes", [])[:3]:
            print(f"    ※ {n}")


def _live_check(trader, cfg) -> None:
    trader.broker.connect()
    trader.current_prices(trader.load_prices())  # paper 브로커는 종가를 현재가로 쓴다
    bal = trader.broker.get_balance(cfg.tickers)
    print(f"연결 OK: {trader.broker.name} ({getattr(trader.broker, 'mode', '-')}) 실계좌={trader.broker.real_money}")
    print(f" 현금 {bal.cash:,.0f} {bal.currency} / 평가 {bal.positions_value:,.0f} / 총 {bal.total:,.0f}")
    for t, p in bal.positions.items():
        print(f"  - {t}: {p.qty}주 평균 {p.avg_price:,.2f} 현재 {p.price:,.2f}")
    for t in cfg.tickers:
        try:
            print(f"  현재가 {t}: {trader.broker.get_price(t):,.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"  현재가 {t}: 조회 실패 ({e})")


def _confirm_real_money(trader, cfg, args) -> None:
    if args.execute and trader.broker.real_money and not args.yes:
        print(f"⚠️ 실계좌({cfg.broker}:{cfg.kis.get('mode')})에 실제 주문을 보냅니다. 계속하려면 'yes' 를 입력하세요.")
        if input("> ").strip().lower() != "yes":
            raise SystemExit("취소했습니다.")


def _live_schedule(trader, at: str, execute: bool) -> None:
    import time as _time
    from datetime import datetime, timedelta

    hh, mm = (int(x) for x in at.split(":"))
    print(f"매일 {hh:02d}:{mm:02d} 에 실행합니다 (Ctrl+C 로 종료). 킬 스위치: python cli.py live stop")
    while True:
        now = datetime.now()
        nxt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        print(f" 다음 실행: {nxt:%Y-%m-%d %H:%M}")
        _time.sleep(max(1.0, (nxt - datetime.now()).total_seconds()))
        trader._prices = None  # 새 데이터로
        try:
            print(trader.run(execute=execute).summary_korean())
            print(DISCLAIMER)
        except Exception as e:  # noqa: BLE001
            print(f" 실행 오류: {e}")


def cmd_live(args):
    """실전 자동매매: doctor / check / run / status / stop / resume / schedule."""
    from quant.config import PROJECT_ROOT, load_config, validate_config
    from quant.live import LiveTrader

    cfg = _live_apply_overrides(load_config(args.config), args)
    kill = PROJECT_ROOT / cfg.risk.kill_switch_file
    if args.live_cmd == "doctor":
        issues = validate_config(cfg)
        src = args.config or "config.toml"
        print(f"설정 점검 ({src}): 브로커 {cfg.broker}, 전략 {cfg.strategy}, 종목 {len(cfg.tickers)}개")
        for i in issues:
            print(" -", i)
        print("문제 없음." if not issues else f"{len(issues)}건 확인 필요.")
        return
    if args.live_cmd in ("stop", "resume"):
        return _live_kill_switch(args.live_cmd, kill)
    if args.live_cmd == "status":
        return _live_status(cfg, kill, args.limit)
    if cfg.broker == "kis" and not (cfg.kis.get("app_key") and cfg.kis.get("app_secret")):
        raise SystemExit("KIS 키가 없습니다. config.toml 의 [kis] 또는 환경변수 KIS_APP_KEY/KIS_APP_SECRET/KIS_ACCOUNT_NO 를 설정하세요.")
    trader = LiveTrader(cfg)
    if args.live_cmd == "check":
        return _live_check(trader, cfg)
    _confirm_real_money(trader, cfg, args)
    if args.live_cmd == "run":
        print(trader.run(execute=bool(args.execute)).summary_korean())
        print(DISCLAIMER)
    elif args.live_cmd == "schedule":
        _live_schedule(trader, args.at, bool(args.execute))


def cmd_report(args):
    from quant.config import PROJECT_ROOT, load_config
    from quant.paper import PaperAccount
    from quant.report import build_report, format_report_korean

    acc = PaperAccount(args.file)
    cfg = load_config(args.config)
    rep = build_report(acc, PROJECT_ROOT / cfg.log_dir)
    print(format_report_korean(rep))
    if args.csv:
        out = PROJECT_ROOT / "outputs"
        out.mkdir(exist_ok=True)
        rep["monthly"].to_csv(out / "report_monthly.csv", index=False, encoding="utf-8-sig")
        rep["orders"].to_csv(out / "report_orders.csv", index=False, encoding="utf-8-sig")
        print(f"CSV 저장: {out / 'report_monthly.csv'}, {out / 'report_orders.csv'}")


def cmd_strategies(_args):
    from quant.strategies import STRATEGIES

    for name, cls in STRATEGIES.items():
        print(f"\n● {name} - {cls.label}")
        print(f"  {cls.description}")
        for p in cls.params:
            rng = f" (범위 {p.min}~{p.max})" if p.min is not None else ""
            print(f"    · {p.name} = {p.default}{rng}  {p.help}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cli.py", description="quant-trader: 백테스트 / 신호 / 페이퍼 트레이딩")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, period_default="5y"):
        p.add_argument("--tickers", required=True, help="쉼표 구분 종목. 예: SPY,QQQ 또는 005930.KS")
        p.add_argument("--strategy", default="sma_cross", help="전략 이름 (cli.py strategies 참고)")
        p.add_argument("--param", action="append", default=[], help="전략 파라미터 key=value (반복 가능)")
        p.add_argument("--period", default=period_default, help="1y/2y/3y/5y/10y 등")

    b = sub.add_parser("backtest", help="백테스트 실행")
    common(b)
    b.add_argument("--fee", type=float, default=0.0015, help="편도 수수료율 (기본 0.0015)")
    b.add_argument("--stop-loss", type=float, default=None, help="손절 비율 (예: 0.1)")
    b.add_argument("--rebalance", default="daily", choices=["daily", "weekly", "monthly"])
    b.add_argument("--min-trade", type=float, default=0.02, help="비중 차이가 이보다 작으면 매매 생략 (기본 0.02, 실전과 동일)")
    b.add_argument("--cash-rate", type=float, default=0.0, help="현금 연 이자율 (예: 0.03). 기본 0 (보수적)")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("signal", help="오늘의 신호")
    common(s, "2y")
    s.set_defaults(func=cmd_signal)

    pp = sub.add_parser("paper", help="페이퍼 트레이딩 계좌")
    pp.add_argument("--file", default="paper_account.json", help="계좌 파일 경로")
    ps = pp.add_subparsers(dest="paper_cmd", required=True)
    pi = ps.add_parser("init", help="계좌 초기화")
    pi.add_argument("--cash", type=float, default=10_000_000)
    pst = ps.add_parser("status", help="계좌 현황")
    pst.add_argument("--tickers", default="", help="평가에 쓸 종목 (생략 시 보유 종목)")
    pr = ps.add_parser("run", help="신호 계산 후 계좌 리밸런스")
    common(pr, "2y")
    pr.add_argument("--fee", type=float, default=0.0015)
    pp.set_defaults(func=cmd_paper)

    st = sub.add_parser("strategies", help="전략 목록")
    st.set_defaults(func=cmd_strategies)

    rp = sub.add_parser("report", help="월간 성과 리포트 (페이퍼 계좌 + 실전 로그)")
    rp.add_argument("--file", default="paper_account.json", help="계좌 파일 경로")
    rp.add_argument("--config", default=None)
    rp.add_argument("--csv", action="store_true", help="outputs/ 에 CSV 저장")
    rp.set_defaults(func=cmd_report)

    lv = sub.add_parser("live", help="실전 자동매매 (config.toml 기반, 기본 dry-run)")
    lv.add_argument("--config", default=None, help="설정 파일 (기본 config.toml)")
    lv.add_argument("--tickers", default="", help="설정 파일의 종목을 덮어씀")
    lv.add_argument("--strategy", default="", help="설정 파일의 전략을 덮어씀")
    lv.add_argument("--broker", default="", choices=["", "paper", "kis"], help="설정 파일의 브로커를 덮어씀")
    ls = lv.add_subparsers(dest="live_cmd", required=True)
    ls.add_parser("doctor", help="설정 파일 점검 (네트워크 없음)")
    ls.add_parser("check", help="브로커 연결/잔고/현재가 확인 (주문 없음)")
    lr = ls.add_parser("run", help="신호 계산 후 주문 (기본 dry-run)")
    lr.add_argument("--execute", action="store_true", help="실제로 주문을 전송한다")
    lr.add_argument("--yes", action="store_true", help="실계좌 확인 질문 생략 (스케줄러용)")
    lsch = ls.add_parser("schedule", help="매일 지정 시각에 run 반복")
    lsch.add_argument("--at", default="15:20", help="HH:MM (기본 15:20, 국내 장 마감 전)")
    lsch.add_argument("--execute", action="store_true")
    lsch.add_argument("--yes", action="store_true")
    lst = ls.add_parser("status", help="최근 실행 로그")
    lst.add_argument("--limit", type=int, default=10)
    ls.add_parser("stop", help="킬 스위치 ON (모든 주문 중단)")
    ls.add_parser("resume", help="킬 스위치 OFF")
    lv.set_defaults(func=cmd_live)
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, ImportError) as e:
        raise SystemExit(f"오류: {e}")


if __name__ == "__main__":
    main()
