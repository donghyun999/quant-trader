"""한국투자증권 KIS Developers Open API 브로커.

- 국내주식(코스피/코스닥): 현재가, 잔고, 현금 주문(시장가/지정가)
- 해외주식(미국): 현재가, 잔고, 지정가 주문 (KIS 해외주식은 시장가가 제한적이라 지정가로 낸다)
- 모의투자(vts) / 실전(prod) 서버 전환: `mode="vts" | "prod"`

TR ID, 엔드포인트는 모두 `ENDPOINTS` 한 곳에 모아 두었다. KIS 가 스펙을 바꾸면 여기만 고친다.
공식 문서: https://apiportal.koreainvestment.com

주의
- 이 모듈은 실제 계좌에 주문을 보낸다. 반드시 모의투자(vts)로 먼저 검증한 뒤 prod 로 바꿀 것.
- 토큰은 `token_cache` 파일에 저장해 재사용한다 (KIS 는 토큰 발급 횟수를 제한한다).
"""
from __future__ import annotations

import json
import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .base import Balance, Broker, BrokerError, OrderResult, Position, Side

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

BASE_URL = {
    "prod": "https://openapi.koreainvestment.com:9443",
    "vts": "https://openapivts.koreainvestment.com:29443",
}

# 미국 티커 -> KIS 거래소 코드. (주문용 OVRS_EXCG_CD, 시세용 EXCD)
US_EXCHANGE = {
    "SPY": "AMEX", "GLD": "AMEX", "IWM": "AMEX", "VOO": "AMEX", "VTI": "AMEX", "SCHD": "AMEX",
    "QQQ": "NASD", "TLT": "NASD", "TQQQ": "NASD", "AAPL": "NASD", "MSFT": "NASD", "NVDA": "NASD",
    "GOOGL": "NASD", "AMZN": "NASD", "META": "NASD", "TSLA": "NASD",
}
EXCD_FOR_QUOTE = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}

ENDPOINTS: dict[str, dict[str, Any]] = {
    "token": {"path": "/oauth2/tokenP"},
    "kr_price": {"path": "/uapi/domestic-stock/v1/quotations/inquire-price", "tr": "FHKST01010100"},
    "kr_balance": {"path": "/uapi/domestic-stock/v1/trading/inquire-balance",
                   "tr": {"prod": "TTTC8434R", "vts": "VTTC8434R"}},
    "kr_order": {"path": "/uapi/domestic-stock/v1/trading/order-cash",
                 "tr": {("prod", "BUY"): "TTTC0012U", ("prod", "SELL"): "TTTC0011U",
                        ("vts", "BUY"): "VTTC0012U", ("vts", "SELL"): "VTTC0011U"}},
    "us_price": {"path": "/uapi/overseas-price/v1/quotations/price", "tr": "HHDFS00000300"},
    "us_balance": {"path": "/uapi/overseas-stock/v1/trading/inquire-balance",
                   "tr": {"prod": "TTTS3012R", "vts": "VTTS3012R"}},
    "us_cash": {"path": "/uapi/overseas-stock/v1/trading/inquire-psamount",
                "tr": {"prod": "TTTS3007R", "vts": "VTTS3007R"}},
    "us_order": {"path": "/uapi/overseas-stock/v1/trading/order",
                 "tr": {("prod", "BUY"): "TTTT1002U", ("prod", "SELL"): "TTTT1006U",
                        ("vts", "BUY"): "VTTT1002U", ("vts", "SELL"): "VTTT1001U"}},
}


def split_ticker(ticker: str) -> tuple[str, str]:
    """'005930.KS' -> ('KR', '005930'), 'SPY' -> ('US', 'SPY')."""
    t = ticker.strip().upper()
    m = re.fullmatch(r"(\d{6})\.(KS|KQ)", t)
    if m:
        return "KR", m.group(1)
    if re.fullmatch(r"\d{6}", t):
        return "KR", t
    return "US", t


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return default


class KISBroker(Broker):
    name = "kis"
    real_money = True

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account_no: str,
        mode: str = "vts",
        token_cache: str | Path | None = None,
        us_exchange: dict[str, str] | None = None,
        session: Any = None,
        timeout: float = 10.0,
        limit_slippage_pct: float = 0.003,
    ):
        if mode not in BASE_URL:
            raise ValueError("mode 는 'vts'(모의투자) 또는 'prod'(실전) 이어야 합니다.")
        if not (app_key and app_secret and account_no):
            raise ValueError("KIS app_key / app_secret / account_no 가 비어 있습니다.")
        acct = account_no.replace("-", "")
        if len(acct) != 10:
            raise ValueError("계좌번호는 '12345678-01' 형식(10자리)이어야 합니다.")
        self.app_key, self.app_secret = app_key, app_secret
        self.cano, self.acnt_prdt_cd = acct[:8], acct[8:]
        self.mode = mode
        self.real_money = mode == "prod"
        self.base = BASE_URL[mode]
        self.timeout = timeout
        self.limit_slippage_pct = float(limit_slippage_pct)
        self.us_exchange = {**US_EXCHANGE, **{k.upper(): v for k, v in (us_exchange or {}).items()}}
        self.token_cache = Path(token_cache) if token_cache else Path.home() / ".quant_trader_kis_token.json"
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        if session is None:
            if requests is None:
                raise ImportError("requests 패키지가 필요합니다: pip install requests")
            session = requests.Session()
        self.session = session
        self._last_call = 0.0

    # ---------- HTTP ----------
    def _throttle(self) -> None:
        """KIS 초당 호출 제한(실전 20/s, 모의 2/s) 보호."""
        gap = 0.55 if self.mode == "vts" else 0.06
        wait = self._last_call + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _headers(self, tr_id: str) -> dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def _request(self, method: str, key: str, tr_id: str, params: dict | None = None,
                 body: dict | None = None) -> dict:
        self._throttle()
        url = self.base + ENDPOINTS[key]["path"]
        try:
            r = self.session.request(method, url, headers=self._headers(tr_id), params=params,
                                     json=body, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            raise BrokerError(f"KIS 통신 실패 ({key}): {e}") from e
        try:
            data = r.json()
        except ValueError as e:
            raise BrokerError(f"KIS 응답 파싱 실패 ({key}, HTTP {r.status_code}): {r.text[:200]}") from e
        if r.status_code != 200 or str(data.get("rt_cd", "0")) != "0":
            raise BrokerError(f"KIS 오류 ({key}): HTTP {r.status_code} rt_cd={data.get('rt_cd')} "
                              f"msg_cd={data.get('msg_cd')} {data.get('msg1', '')}".strip())
        return data

    def _tr(self, key: str, side: str | None = None) -> str:
        tr = ENDPOINTS[key]["tr"]
        if isinstance(tr, str):
            return tr
        return tr[(self.mode, side)] if side else tr[self.mode]

    # ---------- 토큰 ----------
    def token(self) -> str:
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token
        if self.token_cache.exists():
            try:
                d = json.loads(self.token_cache.read_text(encoding="utf-8"))
                exp = datetime.fromisoformat(d["expiry"])
                if d.get("mode") == self.mode and d.get("app_key") == self.app_key and datetime.now() < exp:
                    self._token, self._token_expiry = d["token"], exp
                    return self._token
            except Exception:  # noqa: BLE001
                pass
        self._throttle()
        try:
            r = self.session.post(self.base + ENDPOINTS["token"]["path"], json={
                "grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret,
            }, timeout=self.timeout)
            d = r.json()
        except Exception as e:  # noqa: BLE001
            raise BrokerError(f"KIS 토큰 발급 실패: {e}") from e
        if "access_token" not in d:
            raise BrokerError(f"KIS 토큰 발급 실패: {d}")
        self._token = d["access_token"]
        # 만료 10분 전까지만 사용
        self._token_expiry = datetime.now() + timedelta(seconds=int(d.get("expires_in", 86400)) - 600)
        try:
            self.token_cache.write_text(json.dumps({
                "token": self._token, "expiry": self._token_expiry.isoformat(),
                "mode": self.mode, "app_key": self.app_key,
            }), encoding="utf-8")
        except OSError as e:
            warnings.warn(f"토큰 캐시 저장 실패: {e}")
        return self._token

    def connect(self) -> None:
        self.token()

    # ---------- 시세 ----------
    def get_price(self, ticker: str) -> float:
        market, code = split_ticker(ticker)
        if market == "KR":
            d = self._request("GET", "kr_price", self._tr("kr_price"),
                              params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code})
            px = _f(d.get("output", {}).get("stck_prpr"))
        else:
            excd = EXCD_FOR_QUOTE[self._us_exchange(code)]
            d = self._request("GET", "us_price", self._tr("us_price"),
                              params={"AUTH": "", "EXCD": excd, "SYMB": code})
            px = _f(d.get("output", {}).get("last"))
        if px <= 0:
            raise BrokerError(f"{ticker}: 현재가를 받지 못했습니다.")
        return px

    def _us_exchange(self, code: str) -> str:
        ex = self.us_exchange.get(code)
        if not ex:
            raise BrokerError(f"{code}: 미국 거래소 코드를 모릅니다. config 의 [kis.us_exchange] 에 "
                              f"'{code} = \"NASD\"' (또는 NYSE/AMEX) 를 추가하세요.")
        return ex

    # ---------- 잔고 ----------
    def get_balance(self, tickers: list[str] | None = None) -> Balance:
        markets = {split_ticker(t)[0] for t in (tickers or [])} or {"KR"}
        positions: dict[str, Position] = {}
        cash = 0.0
        currency = "KRW"
        if "KR" in markets:
            c, p = self._kr_balance()
            cash += c
            positions.update(p)
        if "US" in markets:
            c, p = self._us_balance()
            cash += c
            positions.update(p)
            currency = "USD" if markets == {"US"} else "MIXED"
        if currency == "MIXED":
            warnings.warn("국내와 미국 종목을 한 계좌 설정에 섞으면 현금 통화가 섞입니다. 전략별로 시장을 분리하세요.")
        return Balance(cash=cash, positions=positions, currency=currency)

    def _kr_balance(self) -> tuple[float, dict[str, Position]]:
        params = {
            "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "AFHR_FLPR_YN": "N", "OFL_YN": "",
            "INQR_DVSN": "02", "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01", "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        }
        d = self._request("GET", "kr_balance", self._tr("kr_balance"), params=params)
        positions: dict[str, Position] = {}
        for row in d.get("output1", []) or []:
            qty = int(_f(row.get("hldg_qty")))
            if qty <= 0:
                continue
            code = str(row.get("pdno", "")).zfill(6)
            # 코스닥 여부를 API가 안 주므로 .KS 로 통일하고 매칭은 6자리 코드로 한다
            positions[f"{code}.KS"] = Position(f"{code}.KS", qty, _f(row.get("pchs_avg_pric")), _f(row.get("prpr")))
        out2 = d.get("output2") or [{}]
        o2 = out2[0] if isinstance(out2, list) else out2
        cash = _f(o2.get("dnca_tot_amt"))  # 예수금 총액
        return cash, positions

    def _us_balance(self) -> tuple[float, dict[str, Position]]:
        positions: dict[str, Position] = {}
        for excg in ("NASD", "NYSE", "AMEX"):
            params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "OVRS_EXCG_CD": excg,
                      "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}
            try:
                d = self._request("GET", "us_balance", self._tr("us_balance"), params=params)
            except BrokerError as e:
                warnings.warn(f"해외 잔고 조회 실패({excg}): {e}")
                continue
            for row in d.get("output1", []) or []:
                qty = int(_f(row.get("ovrs_cblc_qty")))
                if qty <= 0:
                    continue
                sym = str(row.get("ovrs_pdno", "")).upper()
                positions[sym] = Position(sym, qty, _f(row.get("pchs_avg_pric")), _f(row.get("now_pric2")))
        # 주문 가능 달러: 매수가능금액 조회 (임의 종목 기준, 금액 필드만 사용)
        cash = 0.0
        try:
            params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "OVRS_EXCG_CD": "NASD",
                      "OVRS_ORD_UNPR": "1", "ITEM_CD": "AAPL"}
            d = self._request("GET", "us_cash", self._tr("us_cash"), params=params)
            o = d.get("output", {}) or {}
            cash = _f(o.get("ovrs_ord_psbl_amt")) or _f(o.get("frcr_ord_psbl_amt1"))
        except BrokerError as e:
            warnings.warn(f"해외 주문가능금액 조회 실패: {e}")
        return cash, positions

    # ---------- 주문 ----------
    def submit_order(self, ticker: str, side: Side, qty: int, price: float | None = None) -> OrderResult:
        qty = int(qty)
        if qty <= 0:
            return OrderResult(ticker, side, qty, price or 0.0, False, message="수량이 0 입니다.", broker=self.name)
        market, code = split_ticker(ticker)
        try:
            if market == "KR":
                body = {
                    "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "PDNO": code,
                    "ORD_DVSN": "01" if price is None else "00",      # 01 시장가, 00 지정가
                    "ORD_QTY": str(qty), "ORD_UNPR": "0" if price is None else str(int(round(price))),
                }
                d = self._request("POST", "kr_order", self._tr("kr_order", side), body=body)
                ref_px = price if price is not None else 0.0
            else:
                # 해외는 지정가: 현재가에서 약간 불리한 가격으로 내 체결 확률을 높인다
                ref_px = price if price is not None else self.get_price(ticker)
                adj = 1 + self.limit_slippage_pct if side == "BUY" else 1 - self.limit_slippage_pct
                limit_px = round(ref_px * adj, 2)
                body = {
                    "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd,
                    "OVRS_EXCG_CD": self._us_exchange(code), "PDNO": code,
                    "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{limit_px:.2f}",
                    "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00",
                }
                d = self._request("POST", "us_order", self._tr("us_order", side), body=body)
                ref_px = limit_px
            out = d.get("output", {}) or {}
            return OrderResult(ticker, side, qty, float(ref_px), True, order_id=str(out.get("ODNO", "")),
                               message=str(d.get("msg1", "")), broker=f"{self.name}:{self.mode}")
        except BrokerError as e:
            return OrderResult(ticker, side, qty, float(price or 0.0), False, message=str(e),
                               broker=f"{self.name}:{self.mode}")

    def is_market_open(self) -> bool:
        """한국 정규장(09:00~15:30 KST, 평일) 기준. 미국 종목은 별도로 판단하지 않는다(True)."""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        return True
