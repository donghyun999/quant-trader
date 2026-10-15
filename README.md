# quant-trader (퀀트 트레이더)

퀀트를 몰라도 쓸 수 있는 **주식·ETF 백테스트 + 오늘의 신호 + 가상 투자(페이퍼 트레이딩)** 프로그램입니다.
무료 데이터(yfinance)를 사용하고, **실제 주문은 전송하지 않습니다.**

## 설치

Python 3.12 권장.

```
pip install -r requirements.txt
```

## 대시보드 실행

```
streamlit run app.py
```
Windows에서는 `run.bat`을 더블클릭해도 됩니다. 브라우저에 한국어 화면이 열립니다.

탭 구성: 백테스트 / 파라미터 탐색 / 오늘의 신호 / 페이퍼 계좌 / 도움말

## CLI 사용 예

```
python cli.py strategies
python cli.py backtest --tickers SPY,QQQ --strategy sma_cross --period 5y --param fast=20 --param slow=60 --fee 0.0015 --stop-loss 0.1
python cli.py signal --tickers 005930.KS,000660.KS --strategy dual_momentum
python cli.py paper init --cash 10000000
python cli.py paper run --tickers SPY,QQQ,TLT,GLD --strategy dual_momentum
python cli.py paper status --tickers SPY,QQQ,TLT,GLD
```
백테스트의 자산 곡선은 `outputs/` 폴더에 CSV로 저장됩니다.
종목 표기: 미국은 `SPY`, 한국 코스피는 `005930.KS`, 코스닥은 `.KQ`.

## 전략 (쉬운 설명)

- **이동평균 교차 (sma_cross)**: 단기 평균 가격(예: 20일)이 장기 평균(예: 60일)을 넘으면 "오르는 추세"로 보고 사고, 아래로 내려가면 팝니다.
- **RSI 되돌림 (rsi_reversion)**: 너무 많이 떨어졌을 때(과매도) 사서 평균으로 돌아오면 파는 방식입니다.
- **볼린저 밴드 (bollinger)**: 가격이 평소 변동 범위의 아래쪽을 벗어나면 사고, 중앙선으로 돌아오면 파는 방식입니다.
- **듀얼 모멘텀 (dual_momentum)**: 최근 가장 많이 오른 종목에 투자하고, 모두 부진하면 현금으로 대기합니다.
- **단순 보유 (buy_hold)**: 사서 그냥 들고 있습니다. 다른 전략이 이보다 나은지 비교하는 기준입니다.

정확한 목록과 설정값은 `python cli.py strategies`로 확인하세요.

## 결과 읽는 법

- **총 수익률 / CAGR**: 전체 기간 수익 / 연평균 수익.
- **샤프지수**: 위험 대비 수익. 1 이상이면 괜찮은 편.
- **MDD(최대 낙폭)**: 고점 대비 가장 크게 떨어진 비율. 내가 견딜 수 있는 수준인지 확인하세요.
- **벤치마크**: 같은 종목을 균등하게 사서 들고만 있었을 때의 결과. 전략이 이보다 나아야 의미가 있습니다.
- **과최적화 주의**: 파라미터 탐색에서 가장 좋은 값만 고르면 과거에만 잘 맞습니다. 워크포워드 결과를 꼭 확인하세요.

## 한계와 면책

- 이 프로그램은 **투자 조언이 아닙니다.** 과거 성과가 미래 수익을 보장하지 않습니다.
- 무료 데이터에는 오류·누락이 있을 수 있고, 실제 체결가·슬리피지·세금은 다를 수 있습니다.
- 실제 매매 기능은 없으며, 모든 투자 판단과 책임은 본인에게 있습니다.

## 프로젝트 구조

```
app.py              Streamlit 대시보드
cli.py              명령줄 도구
run.bat             Windows 실행 파일
quant/
  data.py           가격 데이터 (yfinance, parquet 캐시)
  indicators.py     지표(SMA, RSI 등)
  strategies/       전략 모음
  backtest.py       백테스트 엔진
  metrics.py        성과 지표와 쉬운 말 설명
  optimize.py       파라미터 탐색, 워크포워드
  signals.py        오늘의 신호
  paper.py          페이퍼 계좌 (paper_account.json)
tests/              테스트 (pytest)
outputs/            백테스트 결과 CSV
```
