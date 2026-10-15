# quant-trader 설계 계약 (모든 모듈은 이 계약을 따른다)

## 목표
퀀트를 모르는 사용자가 실제로 쓸 수 있는 프로그램.
- 무료 데이터(yfinance)로 미국/한국 주식·ETF 조회 (한국: `005930.KS`, `069500.KS` 형식)
- 검증된 단순 전략 몇 개를 백테스트 → 성과를 쉬운 말로 설명
- "오늘 무엇을 해야 하나" 신호를 매일 생성 (매수/매도/보유)
- 가상 계좌(페이퍼 트레이딩)로 실전처럼 추적. 실제 주문 전송은 하지 않음.
- Streamlit 대시보드(한국어) + CLI

## 패키지 구조
```
quant/
  __init__.py
  data.py          # get_prices(tickers, start, end) -> dict[ticker, DataFrame(OHLCV)], parquet 캐시
  indicators.py    # sma, ema, rsi, atr, bollinger, momentum (pd.Series in/out)
  strategies/
    __init__.py    # STRATEGIES 레지스트리: {name: StrategyClass}, get_strategy(name, **params)
    base.py        # Strategy 베이스 클래스
    sma_cross.py, rsi_reversion.py, bollinger.py, dual_momentum.py, buy_hold.py
  backtest.py      # run_backtest(...) -> BacktestResult
  metrics.py       # compute_metrics(equity: pd.Series, trades) -> dict
  optimize.py      # grid_search(...) + walk_forward(...)
  signals.py       # today_signal(strategy, prices) -> list[Signal]
  paper.py         # PaperAccount: json 저장, 입금/주문/평가
app.py             # Streamlit
cli.py             # argparse: backtest / signal / paper
tests/
```

## 핵심 타입 (quant/strategies/base.py)
```python
@dataclass
class Param:               # 파라미터 메타 (UI 자동생성용)
    name: str; default: Any; min: Any=None; max: Any=None; step: Any=None; help: str=""

class Strategy:
    name: str                      # 영문 id, 예: "sma_cross"
    label: str                     # 한국어 표시명
    description: str               # 쉬운 말 설명 (한국어, 2~3문장)
    params: list[Param]            # 클래스 속성
    multi_asset: bool = False      # True면 여러 종목을 한 번에 받아 비중 배분
    def __init__(self, **params)   # 미지정 시 default 사용. self.p: dict
    def generate_weights(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame
        # 반환: index=날짜, columns=ticker, 값=목표 비중(0.0~1.0, 롱온리). 행 합 <= 1.0
        # 반드시 당일 종가 정보까지만 사용 (lookahead 금지). 신호는 다음날 시가/종가에 체결됨 (backtest가 shift)
```

## 데이터 (quant/data.py)
- `get_prices(tickers: list[str], start: str|None=None, end: str|None=None, period: str="5y") -> dict[str, pd.DataFrame]`
- 각 DataFrame: columns = ["open","high","low","close","volume"] 소문자, index=DatetimeIndex(tz-naive, 정렬, 중복 제거), auto_adjust=True
- 캐시: `data_cache/{ticker}.parquet`, 24시간 이내면 재사용. `force=True`로 갱신.
- yfinance MultiIndex 컬럼 처리 필수. 실패 티커는 경고 후 제외, 전부 실패면 ValueError.
- `search_ticker_hint(text)`는 선택 (없어도 됨).

## 백테스트 (quant/backtest.py)
```python
@dataclass
class BacktestConfig:
    initial_cash: float = 10_000_000   # 원 또는 달러, 단위 무관
    fee_pct: float = 0.0015            # 편도 수수료+세금 비율 (한국 주식 기본 0.15%)
    slippage_pct: float = 0.0005
    rebalance: str = "daily"           # "daily" | "weekly" | "monthly"  (비중 변화 체크 주기)
    stop_loss_pct: float | None = None # 종목별 진입가 대비 손절 (예: 0.1 = -10%)
    execution: str = "next_open"       # "next_open" | "next_close"

@dataclass
class BacktestResult:
    equity: pd.Series                  # 일별 총자산
    benchmark_equity: pd.Series        # 동일 종목 균등 Buy&Hold 자산 (같은 초기자금)
    weights: pd.DataFrame              # 실제 보유 비중
    trades: pd.DataFrame               # columns: date, ticker, side(BUY/SELL), qty, price, value, fee
    metrics: dict                      # metrics.compute_metrics 결과
    config: BacktestConfig
    strategy_name: str
    params: dict

def run_backtest(strategy, prices, config=BacktestConfig()) -> BacktestResult
```
- 목표 비중은 **다음 거래일**에 체결 (shift(1)). lookahead 절대 금지.
- 정수 주식 수 단위 체결, 현금 부족 시 가능한 만큼만.
- 데이터 길이가 다르면 공통 기간(inner join) 사용. 한 종목에 결측이면 그 날 그 종목 비중 0.

## 지표 (quant/metrics.py) — compute_metrics 반환 키
`total_return, cagr, annual_vol, sharpe(rf=0), sortino, max_drawdown, max_drawdown_duration_days, calmar, win_rate, num_trades, profit_factor, avg_trade_return, exposure, best_month, worst_month, final_equity` + `benchmark_total_return, benchmark_cagr, benchmark_max_drawdown` (벤치마크 전달 시)
- 연율화는 252 거래일 기준.
- `describe_metrics_korean(metrics) -> list[str]` : 쉬운 말 문장 리스트 (예: "연평균 12.3% 수익, 최악의 경우 -18%까지 떨어진 적이 있습니다").

## 최적화 (quant/optimize.py)
- `grid_search(strategy_cls, prices, param_grid: dict[str, list], config, metric="sharpe") -> pd.DataFrame` (각 조합 결과, 정렬)
- `walk_forward(strategy_cls, prices, param_grid, config, n_splits=4, train_ratio=0.7) -> dict` : 구간별 in-sample 최적 파라미터 → out-of-sample 성과. 과최적화 경고 메시지 포함.

## 신호 (quant/signals.py)
```python
@dataclass
class Signal:
    ticker: str; action: str  # "BUY" | "SELL" | "HOLD" | "NONE"
    target_weight: float; current_weight: float; price: float; reason: str  # 한국어 이유
def today_signals(strategy, prices, current_weights: dict[str,float]|None=None, threshold=0.02) -> list[Signal]
```
- 마지막 날 목표 비중과 현재 비중 차이로 액션 결정.

## 페이퍼 계좌 (quant/paper.py)
```python
class PaperAccount:
    def __init__(self, path="paper_account.json", initial_cash=10_000_000)
    cash, positions: dict[ticker, {"qty": int, "avg_price": float}], history: list[dict]
    def load()/save()
    def deposit(amount)
    def buy(ticker, qty, price, fee_pct=0.0015)/sell(...)
    def rebalance_to(target_weights: dict[str,float], prices: dict[str,float], fee_pct) -> list[dict]  # 실행된 주문 목록
    def value(prices: dict[str,float]) -> dict  # total, cash, positions_value, pnl, pnl_pct
    def snapshot(prices) -> None  # history에 일별 자산 기록
```

## UI (app.py, Streamlit, 한국어)
사이드바: 종목 입력(쉼표 구분, 프리셋 버튼: 미국 ETF / 한국 대형주 / 한국 ETF), 기간, 전략 선택, 파라미터(Param 메타로 자동 생성), 수수료, 손절.
탭: ① 백테스트(자산곡선 vs 벤치마크 plotly, 지표 카드, 쉬운말 설명, 거래내역) ② 파라미터 탐색(grid+walk-forward, 과최적화 경고) ③ 오늘의 신호 ④ 페이퍼 계좌(현황, 신호대로 리밸런스 버튼, 자산 추이) ⑤ 전략 설명/도움말(초보용 용어 설명)

## 공통 규칙
- Python 3.12, pandas 2.x, numpy 2.x. 타입힌트. 외부 의존: yfinance, pandas, numpy, streamlit, plotly, pyarrow.
- 모든 사용자 노출 문자열은 한국어. 코드 주석은 한국어 또는 영어.
- 테스트: pytest, 네트워크 없이 합성 데이터로 (tests/conftest.py에 make_fake_prices 픽스처).
- 디스클레이머: 투자 조언 아님, 과거 성과가 미래를 보장하지 않음. README와 UI에 표기.
