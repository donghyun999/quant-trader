# research/ — 전략 평가와 개발 기록

| 파일 | 용도 |
|---|---|
| `evaluate.py` | 평가 세트. 8개 개발 묶음(+4개 홀드아웃) × 3·5·10년 × 10년 워크포워드로 전 전략을 돌려 `robust_score` 리더보드 생성 |
| `replay_live.py` | 실전 실행기를 과거 N일 하루씩 재현해 백테스트와 일치하는지 검증(실데이터, 수동) |
| `LOG.md` | 회차별 실험·결과·배운 것. 전략 기본값이 왜 그렇게 정해졌는지의 근거 |
| `results/leaderboard.csv` | 최신 전체 리더보드 (대시보드 ⑤ 도움말 탭에 표시) |
| `results/leaderboard_*.csv` | 변형·민감도 실험 결과(태그별) |
| `results/baseline_quick.csv`, `baseline_full.csv` | 회귀 가드 기준선 |
| `results/detail_latest*.csv` | 유니버스별 상세 |

## 자주 쓰는 명령

```
python -X utf8 research/evaluate.py                       # 전체 (약 2분)
python -X utf8 research/evaluate.py --quick --baseline research/results/baseline_quick.csv   # 회귀 확인 (약 35초)
python -X utf8 research/evaluate.py --holdout             # 개발에 안 쓴 묶음으로 검증
python -X utf8 research/evaluate.py --variants --variant-names ens_w211,tv_mkt --tag mytest   # 고정 파라미터 변형
python -X utf8 research/evaluate.py --cash-rate 0.03 --tag cash3                              # 현금 이자 가정
python -X utf8 research/replay_live.py --days 120         # 실행기 재현 검증
```

## 채택 규칙

1. 전략 기본값 변경은 개발 묶음 리더보드 **와** 홀드아웃 둘 다 개선될 때만.
2. `robust_score` 0.03 미만 차이는 노이즈로 간주한다.
3. 거래 횟수가 늘어나는 변경은 `--fee 0.003` 으로도 확인한다.
4. 변경 후 `pytest` 와 `--quick --baseline` 을 돌린다.
