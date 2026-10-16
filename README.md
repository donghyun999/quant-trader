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
- **추세 필터 + 변동성 역가중 (trend_invvol)**: 종목 묶음 전체가 200일 평균 위에 있을 때만 투자하고, 변동성이 낮은 종목에 더 많이 배분합니다. 큰 하락을 피하는 데 초점.
- **모멘텀 앙상블 (momentum_ensemble)**: 3·6·12개월 수익률 평균으로 순위를 매겨 상위 종목을 삽니다. 묶음 전체가 200일 평균 아래면 현금으로 피합니다.
- **변동성 목표 오버레이 (vol_target)**: 다른 전략 위에 얹어 포트폴리오 변동성이 목표(연 15%)를 넘으면 비중을 줄입니다.
- **분할 리밸런스 (tranched)**: 월간 전략을 4조각으로 나눠 월말, +5일, +10일, +15일에 각각 리밸런스해 '날짜 운'을 없앱니다.
- **낙폭 제어 오버레이 (dd_control)**: 기반 전략의 자산이 고점 대비 일정 비율(기본 10%) 넘게 빠져 있는 동안 비중을 절반으로 줄입니다. 수익을 약간 내주고 최악 낙폭을 더 줄이는 방어형 옵션.
- **전략 앙상블 (ensemble)**: 여러 전략의 목표 비중을 평균합니다. 기본 구성은 분할 리밸런스한 추세필터 + 모멘텀 앙상블 + 단순 보유이며, 평가 세트(`research/`)에서 가장 강건했던 방식입니다.

전략 평가 세트와 개발 기록은 `research/LOG.md`, 실행은 `python -X utf8 research/evaluate.py` 입니다.

정확한 목록과 설정값은 `python cli.py strategies`로 확인하세요.

## 전략의 성격 (평가 결과 요약)

10년 × 8개 종목 묶음 평가(`research/LOG.md`)에서 모든 타이밍·리스크 전략은 **상승장에서 단순 보유보다 덜 벌었고**, 2020년 급락·2022년 약세장 같은 위기 구간에서만 앞섰습니다.
즉 이 프로그램의 전략은 수익 극대화가 아니라 **낙폭 관리** 도구입니다. 예를 들어 기본 `ensemble` 은 10년 연평균 11.7%(단순 보유 16.4%)에 최악 낙폭 -45%(단순 보유 -54%), `vol_target(base=ensemble)` 은 8.7%에 -35% 입니다.
수익 몇 %p 를 내주고 최악의 하락을 줄이는 거래이며, 그 거래가 본인에게 맞는지가 선택의 핵심입니다.

개발에 한 번도 쓰지 않은 홀드아웃 묶음 4개(미국 섹터·광범위 ETF, 한국 테마 ETF, 단일 종목)에서도 `ensemble` 이 1위였고 4개 모두에서 낙폭을 줄였습니다(`python -X utf8 research/evaluate.py --holdout`). 전략 기본값은 개발 묶음과 홀드아웃 **둘 다** 개선될 때만 바꿉니다.

## 결과 읽는 법

- **총 수익률 / CAGR**: 전체 기간 수익 / 연평균 수익.
- **샤프지수**: 위험 대비 수익. 1 이상이면 괜찮은 편.
- **MDD(최대 낙폭)**: 고점 대비 가장 크게 떨어진 비율. 내가 견딜 수 있는 수준인지 확인하세요.
- **벤치마크**: 같은 종목을 균등하게 사서 들고만 있었을 때의 결과. 전략이 이보다 나아야 의미가 있습니다.
- **과최적화 주의**: 파라미터 탐색에서 가장 좋은 값만 고르면 과거에만 잘 맞습니다. 워크포워드 결과를 꼭 확인하세요.

## 실전 자동매매 (한국투자증권 KIS API)

페이퍼 계좌로 충분히 검증한 뒤에만 쓰세요. 구조는 `브로커 인터페이스 → 리스크 가드 → 실행기` 입니다.

```
quant/broker/base.py   Broker 인터페이스 (잔고, 현재가, 주문)
quant/broker/paper.py  가상 브로커 (실제 돈 없음)
quant/broker/kis.py    한국투자증권 KIS Developers (국내주식 + 미국주식, 모의/실전)
quant/risk.py          리스크 한도 (종목 비중, 1회 금액, 일일 손실, 킬 스위치)
quant/live.py          실행기: 신호 → 주문 계획 → 리스크 가드 → 전송 → live_log/*.jsonl
quant/config.py        config.toml 로더
```

### 설정

단계별 체크리스트는 [docs/KIS_SETUP.md](docs/KIS_SETUP.md) 를 보세요. 요약:

1. https://apiportal.koreainvestment.com 에서 앱 키/시크릿을 발급받고 **모의투자** 계좌를 먼저 엽니다.
2. `config.example.toml` 을 `config.toml` 로 복사해 `[kis]` 키와 계좌번호, `mode = "vts"` 를 채웁니다. (키는 환경변수 `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO` 로도 줄 수 있습니다.)
3. `[risk]` 한도를 본인 기준으로 정합니다. 특히 `max_order_value` 와 `max_position_weight`.

### 사용 순서 (반드시 이 순서로)

```
python cli.py live check                 # 연결/잔고/현재가 확인. 주문 없음
python cli.py live run                   # dry-run: 어떤 주문이 나갈지 계산만 함
python cli.py live run --execute         # 모의투자(vts)에 실제 전송
python cli.py live status                # 실행 로그
python cli.py live stop / resume         # 킬 스위치 ON/OFF (STOP_TRADING 파일)
python cli.py live schedule --at 15:20 --execute --yes   # 매일 자동 실행 (평일)
```

모의투자에서 몇 주 이상 문제 없이 돌아간 뒤에 `mode = "prod"` 로 바꾸세요. 실계좌에 `--execute` 를 쓰면 확인 질문이 뜹니다(`--yes` 로 생략, 스케줄러용).
Windows 작업 스케줄러에 `run_live.bat --execute --yes` 를 등록해도 됩니다.

### 안전장치

- 기본은 항상 dry-run. `--execute` 없이는 어떤 주문도 나가지 않습니다.
- 프로젝트 루트에 `STOP_TRADING` 파일이 있으면 모든 주문이 거부됩니다.
- 리스크 가드: 한 종목 최대 비중, 1회 주문 금액, 1회 실행 최대 주문 수, 당일 손실 한도 초과 시 신규 매수 금지, 허용 종목 목록.
- 모든 실행(계획, 거부 사유, 전송 결과, 전후 잔고)이 `live_log/날짜.jsonl` 에 남습니다.

### 알아둘 점

- KIS API 의 TR ID 와 엔드포인트는 `quant/broker/kis.py` 의 `ENDPOINTS` 한 곳에 있습니다. KIS 가 스펙을 바꾸면 거기만 고치면 됩니다. 이 저장소의 KIS 연동은 **가짜 HTTP 응답으로만 테스트**됐으므로 모의투자에서 반드시 직접 확인하세요.
- 국내 종목과 미국 종목은 통화가 달라 한 설정에 섞지 마세요. 전략별로 `config` 를 나눠 `--config` 로 지정하면 됩니다.
- 미국 주문은 지정가(현재가 ±0.3%)로 나갑니다. 미국 종목 거래소 코드가 기본 목록에 없으면 `[kis.us_exchange]` 에 추가하세요.
- 신호는 전날 종가 기준이므로 장중 아무 때나 실행해도 같은 결과입니다. 국내는 15:20 전후, 미국은 23:30 이후(서머타임 22:30) 실행을 권합니다.

## 한계와 면책

- 이 프로그램은 **투자 조언이 아닙니다.** 과거 성과가 미래 수익을 보장하지 않습니다.
- 무료 데이터에는 오류·누락이 있을 수 있고, 실제 체결가·슬리피지·세금은 다를 수 있습니다.
- 실전 매매 기능은 본인이 직접 설정하고 `--execute` 를 켜야만 동작하며, 그로 인한 모든 손실과 책임은 본인에게 있습니다.

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
  broker/           브로커 인터페이스 (paper, kis)
  risk.py           실전 리스크 가드
  live.py           실전 실행기
  config.py         config.toml 로더
config.example.toml 실전 설정 예시 (config.toml 로 복사)
run_live.bat        실전 1회 실행 (기본 dry-run)
tests/              테스트 (pytest)
outputs/            백테스트 결과 CSV
```
