# app.py
# pip install -U streamlit pandas plotly yfinance xlsxwriter requests beautifulsoup4 lxml

import os, re, math, time
from io import BytesIO
from datetime import date
import html as ihtml
from lxml import etree

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests
from io import StringIO
from typing import Optional 

st.set_page_config(page_title="Market Performance", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "market_timeseries.csv")
META_CSV  = os.path.join(BASE_DIR, "data", "meta.csv")

# -------------------- 인증(선택) --------------------
def get_app_password():
    try:
        return st.secrets["APP_PASSWORD"]
    except Exception:
        return os.getenv("APP_PASSWORD", "")

APP_PW = get_app_password()
if APP_PW:
    pw = st.text_input("비밀번호를 입력하세요", type="password")
    if pw != APP_PW:
        st.stop()

# ---------- 펀드 매핑: 로드 & 검색 유틸 ----------
FUND_MAP_PATHS = [
    os.path.join(BASE_DIR, "data", "fundcodematching.xlsx"),  # 권장
    os.path.join(BASE_DIR, "data", "fundcodematching.xls"),    # 구버전
    os.path.join(BASE_DIR, "fundcodematching.xlsx"),
    os.path.join(BASE_DIR, "fundcodematching.xls"),
]

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _truncate20(s: str) -> str:
    s = str(s).strip()
    return (s[:20] + "…") if len(s) > 20 else s

def _norm(s: str) -> str:
    # 공백/구두점 제거 + 소문자화 (한글 포함 단순 부분일치용)
    import re
    return re.sub(r"[\s\W_]+", "", str(s)).lower()

@st.cache_data(ttl=600, show_spinner=False)
def load_fund_map():
    """
    엑셀/CSV에서 (코드, 펀드명) 매핑 로드.
    지원 컬럼명(대소문자 무관):
      - 코드: 'code','fund_code','펀드코드','코드'
      - 이름: 'name','fund_name','펀드명','이름'
    """
    path = _first_existing(FUND_MAP_PATHS)
    if not path:
        return pd.DataFrame(columns=["code","name","name20","norm_name"])

    # 확장자에 따라 읽기
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx",):
        df = pd.read_excel(path, engine="openpyxl")
    elif ext in (".xls",):
        # xlrd가 필요합니다(로컬 requirements에 이미 포함하신 것으로 알고 있습니다)
        df = pd.read_excel(path)  # engine 자동
    else:
        df = pd.read_csv(path)

    # 컬럼명 정규화
    cols = {c.lower(): c for c in df.columns}
    code_col = next((cols[k] for k in cols if k in ("code","fund_code","펀드코드","코드")), None)
    name_col = next((cols[k] for k in cols if k in ("name","fund_name","펀드명","이름")), None)
    if not code_col or not name_col:
        # 안전장치: A열=코드, B열=펀드명 가정
        df = df.copy()
        df.columns = [f"col{i+1}" for i in range(len(df.columns))]
        name_col, code_col = "col1","col2"   # A=펀드명, B=코드

    out = df[[code_col, name_col]].rename(columns={code_col:"code", name_col:"name"}).copy()
    out["code"] = out["code"].astype(str).str.strip().str.upper()
    out["name"] = out["name"].astype(str).str.strip()
    out = out.dropna(subset=["code","name"]).drop_duplicates(subset=["code"])
    out["name20"] = out["name"].map(_truncate20)
    out["norm_name"] = out["name"].map(_norm)
    return out

FUND_MAP = load_fund_map()
FUND_CODE_SET = set(FUND_MAP["code"]) if not FUND_MAP.empty else set()

def resolve_terms_to_codes(terms: list[str]) -> tuple[list[str], dict]:
    """
    사용자가 입력한 토큰 목록을 → (확정 코드 리스트, 라벨맵)으로 변환
    - 토큰이 이미 티커(코드)이면 그대로
    - 아니면 펀드명 부분일치 검색으로 code를 찾아 추가
    라벨맵: {코드: 표시명(20자 절단)}
    """
    codes = []
    label_map = {}

    # 사전생성: norm_name -> code (다대일 허용 위해 검색은 contains로)
    for t in terms:
        if not t:
            continue
        t_clean = t.strip()
        # 이미 코드로 들어온 경우 (대문자/숫자/기호 혼합 허용)
        if t_clean.upper() in FUND_CODE_SET:
            codes.append(t_clean.upper())
            # label 준비
            row = FUND_MAP.loc[FUND_MAP["code"]==t_clean.upper()].head(1)
            if not row.empty:
                nm = row.iloc[0]["name20"]
                label_map[t_clean.upper()] = nm
            continue

        # 이름 검색 (부분일치)
        tnorm = _norm(t_clean)
        hits = FUND_MAP.loc[FUND_MAP["norm_name"].str.contains(tnorm, na=False)]
        if not hits.empty:
            for code, nm20 in zip(hits["code"], hits["name20"]):
                codes.append(code)
                label_map[code] = nm20

    # 중복 제거(입력 순서 보존)
    seen = set()
    uniq = []
    for c in codes:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    return uniq, label_map

import yfinance as yf

def search_ticker_or_fund(query: str):
    q = query.strip().upper()
    if not q:
        return []

    # 1. 야후 티커 시도
    try:
        test = yf.Ticker(q)
        hist = test.history(period="1d")
        if not hist.empty:
            return [q]   # ✅ 야후에서 바로 데이터 있으면 그대로 반환
    except Exception:
        pass

    # 2. 펀드 매핑 검색
    matches_code = FUND_MAP.loc[FUND_MAP["code"].str.upper() == q]
    matches_name = FUND_MAP.loc[FUND_MAP["name"].str.upper() == q]
    if not matches_code.empty or not matches_name.empty:
        return list(matches_code["code"]) + list(matches_name["code"])

    # 3. 부분검색 (너무 짧은 건 스킵)
    if len(q) > 2:
        hits = FUND_MAP.loc[FUND_MAP["norm_name"].str.contains(q.lower(), na=False)]
        return list(hits["code"])

    return []

def pretty_label_with_fund(code_or_ticker: str) -> str:
    """
    라벨 우선순위:
      1) FUND_MAP에 있으면 name20
      2) KOFIA 코드면 KOFIA에서 fundNm (가능할 때만; 안전 폴백 포함)
      3) 그 외는 야후 shortName/longName → 20자 말줄임
    """
    c = str(code_or_ticker or "").strip().upper()
    if not c:
        return ""

    # 1) 매핑파일 우선
    try:
        if "FUND_MAP" in globals():
            hit = FUND_MAP.loc[FUND_MAP["code"] == c]
            if not hit.empty:
                return str(hit.iloc[0]["name20"])
    except Exception:
        pass

    # 2) KOFIA 코드 → fundNm (헬퍼가 아직 없을 수도 있어 안전 조회)
    try:
        if is_kofia_code(c):
            _kofia = globals().get("_kofia_name", lambda _x: None)
            _trunc = globals().get("_truncate20_ellipsis",
                                   lambda s: (str(s).strip()[:20] + "…") if len(str(s).strip()) > 20 else str(s).strip())
            nm = _kofia(c)
            if nm:
                return _trunc(nm)
    except Exception:
        # 어떤 이유로든 실패하면 다음 단계로 폴백
        pass

    # 3) 야후/기타 폴백 (여기도 실패해도 최종적으로 심볼 그대로 반환)
    try:
        lbl = pretty_label(c)  # 이미 20자 처리 포함
        return lbl
    except Exception:
        return c


# --- KOFIA 펀드명 캐시 조회(하루 캐시) + 20자 말줄임 ---
@st.cache_data(ttl=86400, show_spinner=False)
def _kofia_name(code: str) -> Optional[str]:
    """KOFIA 코드로 펀드명 한 번만 가져와서 캐시"""
    try:
        today = pd.Timestamp.today().date()
        start = today - pd.Timedelta(days=60)  # 짧게 조회해도 fundNm은 함께 옴
        df = fetch_kofia_nav_xml(code, start, today)
        if not df.empty and "fundNm" in df.columns and df["fundNm"].notna().any():
            return str(df["fundNm"].dropna().iloc[0]).strip()
    except Exception:
        pass
    return None

def _truncate20_ellipsis(s: str) -> str:
    s = str(s).strip()
    return (s[:20] + "…") if len(s) > 20 else s


# ==== KOFIA(DIS) XML endpoint helpers ========================================
KOFIA_URL = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES/"

def _truncate_bytes(s: str, max_bytes: int = 15, encoding="utf-8") -> str:
    b = s.encode(encoding)[:max_bytes]
    while True:
        try:
            return b.decode(encoding)
        except UnicodeDecodeError:
            b = b[:-1]

def is_kofia_code(tok: str) -> bool:
    """펀드 코드 판별: 
       - 국내 ISIN: KR + 10 digits (예: KR5370199261)
       - 단축코드: 4~6 digits (예: 19926)
       - 일부 운용사 전용 코드: K + 11 alnum (예: K55235B39924)
    """
    if not tok:
        return False
    t = tok.strip().upper()
    return bool(re.fullmatch(r"(KR\d{10}|K[A-Z0-9]{11}|\d{4,6})", t))


# === 공용 날짜 파서(YYYYMMDD 우선, 그 외 포맷도 수용) ===
def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    1) 8자리 YYYYMMDD 우선 시도
    2) 구분자 통일 후 자유 파싱
    3) 여전히 None이면 YYYYMMDD 재시도
    ※ infer_datetime_format 옵션 제거(신버전 경고 방지)
    """
    raw = s.astype(str).str.strip()

    # 1) YYYYMMDD 직접 파싱
    dt_a = pd.to_datetime(raw, errors="coerce", format="%Y%m%d")
    if dt_a.notna().sum() >= max(3, int(len(raw) * 0.3)):
        return dt_a

    # 2) 구분자 통일 + 패턴 추출 후 파싱
    norm = (raw.str.replace(".", "-", regex=False)
                .str.replace("/", "-", regex=False))
    # 'YYYY-MM-DD' 패턴 우선 추출, 없으면 원문
    norm2 = norm.str.extract(r"(\d{4}-?\d{2}-?\d{2})", expand=False).fillna(norm)
    dt_b = pd.to_datetime(norm2, errors="coerce")

    # 3) 숫자만 남겨 YYYYMMDD 재시도
    norm3 = norm2.str.replace("-", "", regex=False)
    dt_c = pd.to_datetime(norm3, errors="coerce", format="%Y%m%d")

    return dt_b.where(dt_b.notna(), dt_c)

def _kofia_xml_payload(start_ymd: str, end_ymd: str, fund_id: str) -> str:
    return f"""<?xml version="1.0" encoding="utf-8"?>
<message>
  <proframeHeader>
    <pfmAppName>FS-DIS2</pfmAppName>
    <pfmSvcName>DISFundStdPrcStutSO</pfmSvcName>
    <pfmFnName>select</pfmFnName>
  </proframeHeader>
  <systemHeader></systemHeader>
  <DISCondFuncDTO>
    <tmpV30>{start_ymd}</tmpV30>
    <tmpV31>{end_ymd}</tmpV31>
    <tmpV10>0</tmpV10>
    <tmpV12>{fund_id}</tmpV12>
  </DISCondFuncDTO>
</message>""".strip()

# --- (NEW) KOFIA XML에서 펀드명을 어디에 있어도 찾아내기 ---
def _extract_kofia_name_from_xml(xml_bytes: bytes) -> str | None:
    try:
        root = etree.fromstring(xml_bytes)
    except Exception:
        return None

    def _is_noise(txt: str) -> bool:
        # 내부 키/코드처럼 보이거나 특수기호가 많은 값은 제외
        t = txt.strip()
        tl = t.lower()
        if not t or t.isdigit():
            return True
        if "tmpv" in tl or "vgrid" in tl or t.startswith(("tmp", "ext")):
            return True
        if "^" in t or "=" in t or "{" in t or "}" in t:
            return True
        # 영문/한글 알파벳이 거의 없는 값도 제외
        alpha_cnt = sum(ch.isalpha() for ch in t)
        if alpha_cnt < 2:
            return True
        return False

    # 1) 우선 태그 후보들
    name_tags = [
        "fundNm", "fundName",
        "fnccFundNm", "fnccFundNm1", "fnccFundNm2",
        "vGridBtn1", "vGridBtn2", "ext1", "ext2", "tmpV5", "tmpV4"
    ]
    for tag in name_tags:
        vals = root.xpath(f'//*[local-name()="{tag}"]/text()')
        for v in vals:
            s = re.sub(r"<[^>]*>", "", (v or "")).strip()
            s = re.sub(r"\s+", " ", s).strip()
            if s and not _is_noise(s):
                return s

    # 2) 속성(attribute)에도 혹시 이름이 있을 수 있어 스캔
    for node in root.xpath('//*[@*]'):
        for _, attr_val in getattr(node, 'attrib', {}).items():
            s = re.sub(r"<[^>]*>", "", (attr_val or "")).strip()
            s = re.sub(r"\s+", " ", s).strip()
            if s and len(s) >= 3 and not _is_noise(s):
                return s

    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kofia_nav_xml(code: str, start, end) -> pd.DataFrame:
    """
    KOFIA(DIS)에서 펀드 기준가(NAV)를 ['Date', code, (optional) 'fundNm']로 반환.
      - Date  : tmpV1 (YYYYMMDD)
      - Price : tmpV2 (기준가격, 원)
      - Name  : XML 어디에 있든 최대한 찾아 'fundNm'로 동봉
    """
    code = (code or "").strip()
    if not code:
        return pd.DataFrame(columns=["Date"])

    def _yyyymmdd(d): return pd.Timestamp(d).strftime("%Y%m%d")
    payload = _kofia_xml_payload(_yyyymmdd(start), _yyyymmdd(end), code)
    headers = {
        "Content-Type": "text/xml; charset=UTF-8",
        "Accept": "text/xml,application/xml,text/plain,*/*",
        "User-Agent": "Mozilla/5.0",
    }

    # --- 요청 ---
    try:
        r = requests.post(KOFIA_URL, data=payload.encode("utf-8"), headers=headers, timeout=20)
        if not r.ok:
            st.warning(f"KOFIA HTTP 오류({code}): {r.status_code}")
            return pd.DataFrame(columns=["Date"])
        xml_bytes = r.content
    except Exception as e:
        st.warning(f"KOFIA 요청 실패({code}): {e}")
        return pd.DataFrame(columns=["Date"])

    # --- 파싱 ---
    try:
        root = etree.fromstring(xml_bytes)
        nodes = root.xpath('//*[local-name()="DISCondFuncListDTO"]/*')
        if not nodes:
            nodes = root.xpath('//*[local-name()="DISCondFuncDTO"]')

        rows = []
        for n in nodes:
            row = {}
            for c in n.iterchildren():
                try:
                    k = etree.QName(c.tag).localname
                except Exception:
                    k = (str(c.tag).split("}")[-1] if "}" in str(c.tag) else str(c.tag))
                row[k] = (c.text or "").strip()
            if any(v for v in row.values()):
                rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["Date"])

        df = pd.DataFrame(rows).reset_index(drop=True)

        # 필수 컬럼 확인
        if "tmpV1" not in df.columns or "tmpV2" not in df.columns:
            st.warning(f"KOFIA 응답 스키마 변경 가능성({code}). cols={list(df.columns)}")
            return pd.DataFrame(columns=["Date"])

        # 날짜/가격
        dates = pd.to_datetime(df["tmpV1"].astype(str), errors="coerce", format="%Y%m%d")
        price = pd.to_numeric(
            df["tmpV2"].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
            errors="coerce"
        )

        # --- (핵심) 펀드명은 문서 전체에서 스캔 ---
        fund_name = None
        # 1순위: 명시적 컬럼
        if "fundNm" in df.columns and df["fundNm"].notna().any():
            fund_name = str(df["fundNm"].dropna().iloc[0]).strip()
        # 2순위: 문서 전체 스캔 (vGridBtn1, ext1 등 포함)
        if not fund_name:
            fund_name = _extract_kofia_name_from_xml(xml_bytes)

        out = pd.DataFrame({"Date": dates, code: price})
        if fund_name:
            out["fundNm"] = fund_name

        out[code] = pd.to_numeric(out[code], errors="coerce")

        out = out.dropna(subset=["Date"])
        mask = (out["Date"].dt.date >= pd.to_datetime(start).date()) & \
            (out["Date"].dt.date <= pd.to_datetime(end).date())
        out = out.loc[mask].sort_values("Date").reset_index(drop=True)

        cols = ["Date", code] + (["fundNm"] if "fundNm" in out.columns else [])
        return out[cols]

    except Exception as e:
        st.warning(f"KOFIA 파싱 실패({code}): {e}")
        return pd.DataFrame(columns=["Date"])

@st.cache_data(ttl=86400)
def pretty_label(symbol: str) -> str:
    """
    표시용 라벨:
      - KOFIA 코드: 변환 없이 그대로 반환 (예: KR5370199261)
      - 야후 티커: shortName/longName → 없으면 티커
      - 너무 길면 20자 '...' 처리
    """
    s = (symbol or "").strip().upper()
    if not s:
        return ""

    # KOFIA 코드는 그대로
    if is_kofia_code(s):
        return s

    # 야후 티커 이름 가져오기
    try:
        info = yf.Ticker(s).info or {}
    except Exception:
        info = {}
    name = info.get("shortName") or info.get("longName") or s
    return (name[:20] + "...") if len(name) > 20 else name

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_prices_mixed(tickers: tuple, start, end, use_adjust=True) -> pd.DataFrame:
    """
    - 야후 티커는 fetch_yf_prices로
    - KOFIA(펀드코드: ISIN KRxxxxxxxxxx 또는 4~6자리 단축코드)는 fetch_kofia_nav_xml로
    안전하게 합쳐서 반환. 최소한 ['Date'] 컬럼은 항상 보장.
    """
    if not tickers:
        return pd.DataFrame(columns=["Date"])

    kofia_like, yahoo_like = [], []
    for t in tickers:
        t = (t or "").strip()
        if not t:
            continue
        if is_kofia_code(t):
            kofia_like.append(t)
        else:
            yahoo_like.append(t)

    frames = []

    # 1) Yahoo
    if yahoo_like:
        yf_df = fetch_yf_prices(tuple(yahoo_like), start, end, use_adjust=use_adjust)
        if isinstance(yf_df, pd.DataFrame) and not yf_df.empty and "Date" in yf_df.columns:
            frames.append(yf_df)

    # 2) KOFIA
    for code in kofia_like:
        df_k = fetch_kofia_nav_xml(code, start, end)
        if isinstance(df_k, pd.DataFrame) and not df_k.empty and "Date" in df_k.columns:
            frames.append(df_k)
        else:
            st.info(f"KOFIA 데이터 없음 또는 코드 확인 필요: {code}")

    if not frames:
        return pd.DataFrame(columns=["Date"])

    out = frames[0]
    for f in frames[1:]:
        if "Date" not in f.columns:
            continue
        # 🔧 충돌 방지: fundNm 같은 메타컬럼은 제거
        f = f.drop(columns=[c for c in f.columns if c.lower().startswith("fundnm")], errors="ignore")
        out = out.drop(columns=[c for c in out.columns if c.lower().startswith("fundnm")], errors="ignore")
        out = pd.merge(out, f, on="Date", how="outer")

    return out.sort_values("Date").reset_index(drop=True)

# -------------------- 이름→티커 보편 별칭 --------------------
COMMON_ALIASES = {
    "nikkei225": "^N225", "nikkei 225": "^N225", "nikkei": "^N225",
    "shanghai": "000001.SS", "shanghai composite": "000001.SS",
    "kospi": "^KS11", "kosdaq": "^KQ11",
    "sp500": "^GSPC", "s&p500": "^GSPC", "dow": "^DJI", "nasdaq": "^IXIC",
    "eurostoxx50": "^STOXX50E", "euro stoxx 50": "^STOXX50E",
    "ftse100": "^FTSE", "hang seng": "^HSI", "dax": "^GDAXI", "cac40": "^FCHI",
    "dollar index": "DX-Y.NYB", "us dollar index": "DX-Y.NYB",
    "usdkrw": "USDKRW=X", "usdcny": "CNY=X",
}

YAHOO_ALIAS = {
    "USDKRW": "USDKRW=X",
    "USDJPY": "USDJPY=X",
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "NIKKEI": "^N225",
}
def to_yahoo_ticker(code: str) -> str:
    c = (code or "").strip()
    return YAHOO_ALIAS.get(c.upper(), COMMON_ALIASES.get(c.lower(), c))

# --- KRX helpers -------------------------------------------------------------
def _is_krx_symbol(s: str) -> bool:
    """6자리 숫자 또는 .KS/.KQ로 끝나면 KRX 심볼로 간주"""
    if not s:
        return False
    t = str(s).strip().upper()
    return t.endswith((".KS", ".KQ")) or bool(re.fullmatch(r"\d{6}", t))

def _to_krx_code(sym: str) -> Optional[str]:
    """005930.KS -> 005930 / 005930 -> 005930 / 그 외 -> None"""
    if not sym:
        return None
    t = str(sym).strip().upper()
    m = re.match(r"^(\d{6})(?:\.K[QS])?$", t)
    return m.group(1) if m else None


# -------------------- Finviz 크롤러 --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finviz_company(ticker: str):
    """
    Finviz에서 회사 소개와 스냅샷 지표 테이블을 가져와 (profile_text, df_wide) 반환.
    실패 시 (메시지 문자열, 단일 행 DataFrame) 반환.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return "티커를 입력하세요.", pd.DataFrame(columns=["Indicator 1","Value 1","Indicator 2","Value 2"])

    url = f"https://finviz.com/quote.ashx?t={t}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://finviz.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        time.sleep(1.0)  # 요청 간 딜레이
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok:
            return "Finviz 페이지를 불러오지 못했습니다.", pd.DataFrame(
                [{"Indicator 1":"Error","Value 1":"HTTP 실패","Indicator 2":"URL","Value 2":url}]
            )

        html = r.text

        # 회사 소개
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        node = soup.select_one('td.fullview-profile') or soup.select_one('td[class*="fullview-profile"]')
        profile_text = None
        if node:
            prof_raw = node.get_text(separator=" ", strip=True)
            profile_text = " ".join(ihtml.unescape(prof_raw).split())

        # Finviz에 없으면 yfinance 백업
        if not profile_text:
            try:
                info = (yf.Ticker(t).info or {})
                ysum = info.get("longBusinessSummary")
                if ysum:
                    profile_text = ysum.strip()
            except Exception:
                pass
        if not profile_text:
            profile_text = "회사 소개를 찾을 수 없습니다 (Finviz)."

        # 스냅샷 테이블
        tables = pd.read_html(StringIO(html), attrs={"class": "snapshot-table2"})
        if not tables:
            df_wide = pd.DataFrame(
                [{"Indicator 1":"Error","Value 1":"스냅샷 표 없음","Indicator 2":"Ticker","Value 2":t}]
            )
        else:
            snap = tables[0].copy()
            labels, values = [], []
            cols = list(snap.columns)
            for i in range(0, len(cols)-1, 2):
                labels += snap.iloc[:, i].astype(str).tolist()
                values += snap.iloc[:, i+1].astype(str).tolist()

            n = len(labels)
            rows = (n + 1) // 2
            data = []
            for k in range(rows):
                i1, i2 = 2*k, 2*k+1
                c1 = labels[i1] if i1 < n else ""
                v1 = values[i1] if i1 < n else ""
                c2 = labels[i2] if i2 < n else ""
                v2 = values[i2] if i2 < n else ""
                data.append([c1, v1, c2, v2])
            df_wide = pd.DataFrame(data, columns=["Indicator 1","Value 1","Indicator 2","Value 2"])

        return profile_text, df_wide

    except Exception as e:
        return f"Finviz 로딩 실패: {e}", pd.DataFrame(
            [{"Indicator 1":"Error","Value 1":str(e),"Indicator 2":"Ticker","Value 2":t}]
        )

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_naver_overview(sym: str):
    """
    네이버 금융에서 국내 종목 개요 + 주요지표 테이블을 긁어와
    (profile_text, df_wide) 형태로 반환합니다.
    df_wide 컬럼: ["Indicator 1","Value 1","Indicator 2","Value 2"]
    """
    from bs4 import BeautifulSoup

    code6 = _to_krx_code(sym)
    if not code6:
        return "국내 종목 코드를 인식하지 못했습니다.", pd.DataFrame(
            columns=["Indicator 1","Value 1","Indicator 2","Value 2"]
        )

    url = f"https://finance.naver.com/item/main.nhn?code={code6}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        # 네이버는 EUC-KR일 수 있으니 인코딩 지정
        r.encoding = r.apparent_encoding or "euc-kr"
        soup = BeautifulSoup(r.text, "html.parser")

        # 이름/간단 설명
        nm_node = soup.select_one("div.wrap_company h2 a") or soup.select_one("div.wrap_company h2")
        name = nm_node.get_text(" ", strip=True) if nm_node else code6
        corp_info = soup.select_one("div.corp_info")
        desc = corp_info.get_text(" ", strip=True) if corp_info else ""
        profile_text = f"**{name}** — {desc or '네이버 금융 개요'}"

        # 테이블 → (지표, 값) 쌍 수집
        pairs = []
        for tr in soup.select("table tr"):
            ths = [th.get_text(" ", strip=True) for th in tr.select("th")]
            tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
            for k in range(min(len(ths), len(tds))):
                lab, val = ths[k], tds[k]
                if lab and val and len(lab) <= 30:
                    pairs.append((lab, val))

        # 중복 라벨 제거하고 2열 표로 재구성
        uniq, seen = [], set()
        for lab, val in pairs:
            if lab in seen:
                continue
            seen.add(lab); uniq.append((lab, val))

        rows = []
        for i in range(0, len(uniq), 2):
            c1, v1 = uniq[i]
            c2, v2 = uniq[i+1] if i+1 < len(uniq) else ("", "")
            rows.append([c1, v1, c2, v2])

        df_wide = pd.DataFrame(rows, columns=["Indicator 1","Value 1","Indicator 2","Value 2"])
        return profile_text, df_wide

    except Exception as e:
        return f"네이버 페이지 로딩 실패: {e}", pd.DataFrame(
            columns=["Indicator 1","Value 1","Indicator 2","Value 2"]
        )

# -------------------- 야후 검색 --------------------
def yahoo_search(query: str, quotes_count: int = 10):
    """야후 파이낸스 검색(비공식)"""
    q = query.strip()
    results = []
    alias_key = q.lower()
    if alias_key in COMMON_ALIASES:
        results.append({"symbol": COMMON_ALIASES[alias_key],
                        "shortname": f"Alias for '{q}'",
                        "longname": None, "exchDisp": "—", "quoteType": "ALIAS"})
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/plain, */*"}
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    try:
        r = requests.get(url, headers=headers,
                         params={"q": q, "quotesCount": quotes_count, "newsCount": 0},
                         timeout=8)
        if r.ok:
            js = r.json()
            for it in js.get("quotes", []):
                results.append({
                    "symbol": it.get("symbol"),
                    "shortname": it.get("shortname"),
                    "longname": it.get("longname") or it.get("longName"),
                    "exchDisp": it.get("exchDisp"),
                    "quoteType": it.get("quoteType"),
                })
    except Exception:
        pass
    # dedup
    seen, dedup = set(), []
    for x in results:
        sym = x.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym); dedup.append(x)
    return dedup[:quotes_count]

# -------------------- 공용 유틸 --------------------
@st.cache_data(ttl=3600)
def load_base_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")
    return df

@st.cache_data(ttl=600)
def load_meta(csv_path: str) -> str:
    try:
        m = pd.read_csv(csv_path)
        return str(m.iloc[0]["last_updated"])
    except Exception:
        return ""

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_yf_prices(tickers: tuple, start, end, use_adjust=True) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        list(tickers),
        start=str(start),
        end=str(end + pd.Timedelta(days=1)),  # end는 배타
        auto_adjust=use_adjust,
        progress=False,
        threads=True,
        actions=False,
        repair=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        key = "Adj Close" if use_adjust and "Adj Close" in raw.columns.levels[0] else "Close"
        close = raw[key].copy()
    else:
        key = "Adj Close" if use_adjust and "Adj Close" in raw.columns else "Close"
        close = raw[[key]].copy()
        if len(tickers) == 1:
            close.columns = [list(tickers)[0]]
    close = close.sort_index()
    close.index.name = "Date"
    close = close.ffill().bfill(limit=1)
    return close.reset_index()

def reindex_fill_ffill_bfill(df: pd.DataFrame, start, end, ffill_limit: int = 5, kofia_ffill_limit: int = 5) -> pd.DataFrame:
    """
    영업일 기준으로 리인덱스.
    - 야후 열: 짧게 ffill (기본 5일)
    - KOFIA 열: 과도한 단절 방지 위해 동일하게 ffill (기본 5일)
      (bfill은 적용하지 않음)
    """
    all_days = pd.bdate_range(start=start, end=end)
    out = df.set_index("Date").reindex(all_days)

    # 숫자화
    num_cols = [c for c in out.columns if c != "Date"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    # 구분
    kofia_cols = [c for c in out.columns if c != "Date" and is_kofia_code(str(c))]
    other_cols = [c for c in out.columns if c != "Date" and c not in kofia_cols]

    # 야후: 짧게 ffill
    if other_cols:
        out[other_cols] = out[other_cols].ffill(limit=ffill_limit)

    # KOFIA(펀드): 공백 제거용 ffill (짧은 한도 내)
    if kofia_cols:
        out[kofia_cols] = out[kofia_cols].ffill(limit=kofia_ffill_limit)

    return out.rename_axis("Date").reset_index()


def rebase_pct(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame[["Date"] + cols].copy()
    for c in cols:
        s = out[c].dropna()
        out[c] = ((out[c] / s.iloc[0]) - 1.0) * 100.0 if not s.empty else pd.NA
    return out

def drawdown_pct(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame[["Date"] + cols].copy()
    for c in cols:
        s = out[c].astype(float)
        peak = s.cummax()
        out[c] = (s / peak - 1.0) * 100.0
    return out

def format_tail_value(v, mode_key: str) -> str:
    if pd.isna(v): return ""
    return f"{v:+.1f}%" if mode_key in ("pct", "mdd") else f"{v:,.1f}"

# ==================== TAB 1: Market ====================
def tab_market():
    base = load_base_data(DATA_CSV)
    last_updated = load_meta(META_CSV)

    st.title("Market Performance")
    if last_updated:
        st.caption(f"마지막 업데이트(KST): {last_updated} developed by W.I Lee")

    # UI CSS
    st.markdown("""
    <style>
    :root { --ctrl-h: 42px; --pad-y: 8px; }
    div.stTextInput, div[data-testid="stTextInput"],
    div.stDateInput, div[data-testid="stDateInput"] { min-height: var(--ctrl-h) !important; }
    div.stTextInput input, div[data-testid="stTextInput"] input,
    div.stDateInput input,  div[data-testid="stDateInput"] input {
      height: var(--ctrl-h) !important; line-height: var(--ctrl-h) !important;
      padding-top: var(--pad-y) !important; padding-bottom: var(--pad-y) !important;
    }
    div.stButton { min-height: var(--ctrl-h) !important; }
    div.stButton > button,
    div[data-testid="baseButton-primary"] button,
    div[data-testid="baseButton-secondary"] button,
    div[data-testid="baseButton-default"] button {
      height: var(--ctrl-h) !important; padding-top: var(--pad-y) !important; padding-bottom: var(--pad-y) !important;
    }
    div[data-testid="stWidgetLabel"] > label { margin-bottom: 4px !important; }
    [data-testid="stDataFrame"] table td, 
    [data-testid="stDataFrame"] table th { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)

    # 기간 설정
    min_d = base["Date"].min().date()
    max_d = base["Date"].max().date()
    default_start = max(date(2025, 1, 1), min_d)

    # ▼ 한 줄 레이아웃: 시작일 | 종료일 | 표시통화 | 티커입력 | 버튼
    st.session_state.setdefault("m_tickers", "")
    col_s, col_e, col_ccy, col_inp, col_btn = st.columns([1.2, 1.2, 0.9, 3.2, 0.6])

    with col_s:
        start = st.date_input("시작일", value=default_start,
                            min_value=date(2000, 1, 1), max_value=max_d, key="m_start")

    with col_e:
        end = st.date_input("종료일", value=max_d,
                            min_value=date(2000, 1, 1), max_value=max_d, key="m_end")

    with col_ccy:
        ccy = st.selectbox("표시통화", ["LOCAL", "KRW", "USD"], index=0, key="m_ccy")

    with col_inp:
        st.text_input("티커/펀드코드 입력 (야후 + KOFIA 혼용 가능)",
                    key="m_tickers",
                    placeholder="예: SPY, ^KS11, 005930.KS, KR5370199261, 19926")

    with col_btn:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)  # 버튼 수직 정렬
        fetch_clicked = st.button("반영", type="primary", use_container_width=True, key="m_fetch")

    # 체크박스는 다음 줄로 두어 공간 확보(원하면 위 col_ccy에 넣어도 됨)
    use_adj = st.checkbox("조정가격 사용(배당/액면 반영)", value=False, key="m_adj")

    # 기본 CSV 구간
    mask = (base["Date"].dt.date >= start) & (base["Date"].dt.date <= end)
    view = base.loc[mask].copy()

    st.session_state.setdefault("m_extra", [])
    st.session_state.setdefault("m_ycols", [])

    def expand_aliases(seq):
        # 별칭만 확장, KOFIA 코드는 그대로 둠
        out = []
        for t in seq:
            out.append(COMMON_ALIASES.get(t.lower(), t))
        return out

    # 저장된 추가 자산
    saved = tuple(st.session_state["m_extra"])
    if saved:
        fetched_saved = fetch_prices_mixed(saved, start, end, use_adjust=use_adj)
        if not fetched_saved.empty:
            view = pd.merge(view, fetched_saved, on="Date", how="outer").sort_values("Date")

    
    # 신규 추가
    new_only = []
    if fetch_clicked:
        raw_terms = [t for t in re.split(r"[,\s]+", st.session_state.get("m_tickers","")) if t.strip()]
        terms = expand_aliases(raw_terms)

        all_codes, seen = [], set()
        for t in terms:
            t_clean = t.strip().upper()

            # ✅ Market 탭에서는 펀드명 검색 금지 → 정확한 코드만 허용
            if t_clean in FUND_CODE_SET or is_kofia_code(t_clean):
                hits = [t_clean]
            
            # ✅ 추가: 한국 거래소 심볼(.KS/.KQ 또는 6자리)은 야후가 빈 응답이어도 통과
            elif re.fullmatch(r"\d{6}(\.K[QS])?", t_clean):
                hits = [t_clean if '.K' in t_clean else t_clean + '.KS']

            else:
                # 야후 티커 확인
                try:
                    hist = yf.Ticker(t_clean).history(period="1d")
                    hits = [t_clean] if not hist.empty else []
                except Exception:
                    hits = []

            for h in hits:
                if h not in seen:
                    all_codes.append(h)
                    seen.add(h)

        # 5) 이미 있는 컬럼/추가목록 제외 후 새로 가져오기
        already = set(view.columns) | set(st.session_state["m_extra"])
        new_only = [t for t in all_codes if t not in already]

    if new_only:
        with st.spinner(f"가격 불러오는 중... ({', '.join(new_only)})"):
            fetched_now = fetch_prices_mixed(tuple(new_only), start, end, use_adjust=use_adj)

        if not fetched_now.empty:
            drop_cols = [c for c in fetched_now.columns if c.lower().startswith("fundnm")]
            fetched_now = fetched_now.drop(columns=drop_cols, errors="ignore")

            drop_cols2 = [c for c in view.columns if c.lower().startswith("fundnm")]
            view = view.drop(columns=drop_cols2, errors="ignore")

            view = pd.merge(view, fetched_now, on="Date", how="outer").sort_values("Date")
            st.session_state["m_extra"] = sorted(set(st.session_state["m_extra"]) | set(new_only))
            st.session_state["m_ycols"] = list(
                dict.fromkeys(st.session_state.get("m_ycols", []) + new_only)
            )
            st.success(f"추가된 자산: {', '.join(new_only)}")
        else:
            st.info("새로 추가할 항목이 없습니다.")

    # === 통화 변환 (LOCAL/KRW/USD) ===
    ccy = st.session_state.get("m_ccy", "LOCAL")
    if ccy in ("KRW", "USD"):
        usdkrw = None

        # 1) base에 'USDKRW'가 이미 있으면 우선 사용
        if "USDKRW" in view.columns:
            usdkrw = view[["Date", "USDKRW"]].dropna().set_index("Date")["USDKRW"]
        else:
            # 2) 없으면 야후 환율(KRW=X) 조회
            fx = fetch_yf_prices(("KRW=X",), start, end, use_adjust=False)
            if not fx.empty and "KRW=X" in fx.columns:
                usdkrw = fx.set_index("Date")["KRW=X"]

        if usdkrw is not None and not usdkrw.empty:
            tmp = view.set_index("Date")

            # 변환 대상 컬럼(숫자형만) — 환율 컬럼은 제외
            excl = {"USDKRW", "KRW=X"}
            cols = [c for c in tmp.columns if c not in excl]

            for c in cols:
                s = pd.to_numeric(tmp[c], errors="coerce")

                # 한국 자산 판별: .KS/.KQ, KOFIA 코드, 대표지수
                is_kr = (str(c).endswith(".KS") or str(c).endswith(".KQ")
                        or is_kofia_code(str(c)) or str(c) in ("^KS11", "^KQ11"))

                if ccy == "KRW" and not is_kr:
                    # 해외/달러자산 → 원화
                    tmp[c] = s * usdkrw
                elif ccy == "USD" and is_kr:
                    # 원화자산 → 달러
                    tmp[c] = s / usdkrw

            view = tmp.reset_index()
        else:
            st.info("환율(USDKRW / KRW=X) 데이터를 불러오지 못해 LOCAL로 표시합니다.")
    
    # 리인덱싱
    view = reindex_fill_ffill_bfill(view, start, end)

    all_cols = [c for c in view.columns if c != "Date"]
    init_default = all_cols[:min(3, len(all_cols))]
    st.session_state["m_ycols"] = [c for c in st.session_state.get("m_ycols", init_default) if c in all_cols] or init_default
    ycols = st.multiselect("표시할 자산", options=all_cols, key="m_ycols", format_func=pretty_label_with_fund)

    if not ycols:
        st.info("표시할 자산을 선택하세요."); return

    MODE_LABELS = {"price": "가격", "pct": "일반변화율(%)", "pct_log": "로그 변화율(%)", "mdd": "최대 낙폭(MDD)"}
    mode = st.radio("표시 방식", options=list(MODE_LABELS.keys()), index=1,
                    horizontal=True, format_func=lambda k: MODE_LABELS[k], key="m_mode")
    st.markdown("<h5>(1) Return Chart</h5>", unsafe_allow_html=True)

    # ===== 유효성 & 숫자화 =====
    plot_df = view[["Date"] + ycols].copy()
    for c in ycols:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.dropna(subset=ycols, how="all")
    if plot_df.empty:
        st.info("표시 가능한 데이터가 없습니다. 기간/티커를 조정해 주세요.")
        return

       # 라벨 유일화
    from collections import Counter
    ycols = [c for c in ycols if c in plot_df.columns]
    base_map = {c: pretty_label_with_fund(c) for c in ycols}
    cnt = Counter(base_map.values())
    unique_map, used = {}, set()
    for c in ycols:
        base_lbl = base_map[c]
        lbl = base_lbl
        if lbl in used:
            lbl = f"{base_lbl} ({c})"   # 중복이면 심볼만 덧붙여 구분
        unique_map[c] = lbl; used.add(lbl)

    # ===== 데이터 가공 & 그래프 =====
    if mode == "price":
        plot_df_use = plot_df.copy()
        y_title = "가격지수"
        plot_df_disp = plot_df_use.rename(columns=unique_map)
        ycols_disp = list(unique_map.values())  # ← 무조건 라벨만 사용
        fig = px.line(plot_df_disp, x="Date", y=ycols_disp, render_mode="svg")
        fig.update_traces(connectgaps=True)
        fig.update_yaxes(tickformat=",.1f")

    elif mode == "pct":
        plot_df_use = rebase_pct(plot_df, ycols)  # % 단위
        y_title = "누적 수익률 (%)"
        plot_df_disp = plot_df_use.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]
        fig = px.line(plot_df_disp, x="Date", y=ycols_disp, render_mode="svg")
        fig.update_traces(connectgaps=True)
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")

    elif mode == "pct_log":
        pct = rebase_pct(view, ycols).copy()
        for c in ycols:
            pct[c] = (pd.to_numeric(pct[c], errors="coerce") / 100.0) + 1.0  # 배수

        plot_df_disp = pct.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]

        vals = plot_df_disp[ycols_disp].apply(pd.to_numeric, errors="coerce")
        y_show = [c for c in ycols_disp if (vals[c] > 0).any()]
        if not y_show:
            st.info("로그축에 표시할 수 있는 시리즈가 없습니다. 일반 변화율(%)로 확인해 주세요.")
            return

        y_min = float(vals[y_show].min().min())
        y_max = float(vals[y_show].max().max())
        tick_candidates = [0.25, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100]
        tickvals = [v for v in tick_candidates if v > 0 and y_min * 0.95 <= v <= y_max * 1.05] or [1]
        if 1 not in tickvals:
            tickvals = sorted(set(tickvals + [1]))
        ticktext = [f"{(v - 1) * 100:.0f}%" for v in tickvals]

        y_title = "누적 수익률 (%, 로그 간격)"

        fig = px.line(plot_df_disp, x="Date", y=y_show, render_mode="svg")
        fig.update_traces(connectgaps=True)
        fig.update_layout(
            margin=dict(l=10, r=130, t=10, b=10),
            height=520,
            yaxis_title=y_title,
            legend=dict(groupclick="togglegroup"),
            uirevision="mkt",
            xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(type="log", tickvals=tickvals, ticktext=ticktext)

        last_idx = vals.dropna(how="all").index[-1]
        lx = plot_df_disp.loc[last_idx, "Date"]
        for c in y_show:
            sc = plot_df_disp[["Date", c]].dropna()
            if sc.empty:
                continue
            v_last = float(sc.iloc[-1][c])          # 배수
            pct_last = (v_last - 1.0) * 100.0       # %
            fig.add_trace(
                go.Scatter(
                    x=[lx],
                    y=[v_last],
                    mode="markers+text",
                    text=[f"{pct_last:+.1f}%"],
                    textposition="middle right",
                    marker=dict(size=6),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=c,
                )
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": False, "doubleClick": "reset", "displaylogo": False},
        )
        return

    else:  # mdd
        plot_df_use = drawdown_pct(plot_df, ycols)
        y_title = "MDD (%, 낮을수록 심함)"
        plot_df_disp = plot_df_use.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]
        fig = px.line(plot_df_disp, x="Date", y=ycols_disp, render_mode="svg")
        fig.update_traces(connectgaps=True)
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")

    # 공통 레이아웃 & 마커 라벨
    fig.update_layout(
        margin=dict(l=10, r=130, t=10, b=10),
        height=520,
        yaxis_title=y_title,
        legend=dict(groupclick="togglegroup"),
        uirevision="mkt",
        xaxis_rangeslider_visible=False,
    )
    for tr in fig.data:
        tr.legendgroup = tr.name

    if mode == "mdd":
        for c in ycols_disp:
            s = plot_df_disp[c]
            if s.dropna().empty: continue
            idx_min = s.idxmin()
            fig.add_trace(go.Scatter(
                x=[plot_df_disp.loc[idx_min, "Date"]],
                y=[s.loc[idx_min]],
                mode="markers+text",
                text=[format_tail_value(s.loc[idx_min], "mdd")],
                textposition="bottom right",
                marker=dict(size=8),
                showlegend=False, hoverinfo="skip", legendgroup=c))
    else:
        last_row = plot_df_disp.dropna().iloc[-1] if not plot_df_disp.dropna().empty else None
        if last_row is not None:
            lx = last_row["Date"]
            for c in [cl for cl in plot_df_disp.columns if cl != "Date"]:
                sc = plot_df_disp[["Date", c]].dropna()
                if sc.empty: continue
                idx_max = sc[c].idxmax(); x_max = sc.loc[idx_max, "Date"]; y_max = sc.loc[idx_max, c]
                idx_min = sc[c].idxmin(); x_min = sc.loc[idx_min, "Date"]; y_min = sc.loc[idx_min, c]
                fig.add_trace(go.Scatter(x=[x_max], y=[y_max], mode="markers+text",
                                         text=[format_tail_value(y_max, mode)], textposition="top right", textfont=dict(color="blue"),
                                         marker=dict(size=4), showlegend=False, hoverinfo="skip", legendgroup=c))
                if idx_min != idx_max:
                    fig.add_trace(go.Scatter(x=[x_min], y=[y_min], mode="markers+text",
                                             text=[format_tail_value(y_min, mode)], textposition="bottom right", textfont=dict(color="red"),
                                             marker=dict(size=4, color="blue"), showlegend=False, hoverinfo="skip", legendgroup=c))
                is_last_extreme = (x_max == lx) or (x_min == lx)
                if not is_last_extreme:
                    v_last = sc.iloc[-1][c]
                    fig.add_trace(go.Scatter(x=[lx], y=[v_last], mode="markers+text",
                                             text=[format_tail_value(v_last, mode)], textposition="middle right", textfont=dict(color="black"),
                                             marker=dict(size=4, color="black"), showlegend=False, hoverinfo="skip", legendgroup=c))

    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": False, "doubleClick": "reset", "displaylogo": False})

    # ---- (2) 기간별 수익률 스냅샷 ----
    st.markdown("<h5>(2) Periodic Return</h5>", unsafe_allow_html=True)
    price_df = view[["Date"] + ycols].copy()
    for c in ycols:
        price_df[c] = pd.to_numeric(price_df[c], errors="coerce")

    windows = [("1D",1), ("1W",5), ("1M",21), ("3M",63), ("6M",126), ("12M",252), ("36M",756)]
    rows = []
    for c in ycols:
        s = price_df[c].dropna()
        # ⬇️ 변경: pretty_label → pretty_label_with_fund (펀드명 20자 … 적용)
        row = {"자산": pretty_label_with_fund(c)}
        for name, d in windows:
            if not s.empty and len(s) > d:
                val = s.pct_change(d).iloc[-1] * 100.0
                row[f"R_{name}"] = f"{val:+.2f}%"
            else:
                row[f"R_{name}"] = ""
        rows.append(row)
    snap = pd.DataFrame(rows, columns=["자산"] + [f"R_{n}" for n,_ in windows])
    st.dataframe(
        snap,
        use_container_width=True,
        hide_index=True,
        column_config={
            "자산": st.column_config.TextColumn("Asset", width="large"),
            "R_1D": st.column_config.TextColumn("1D"),
            "R_1W": st.column_config.TextColumn("1W"),
            "R_1M": st.column_config.TextColumn("1M"),
            "R_3M": st.column_config.TextColumn("3M"),
            "R_6M": st.column_config.TextColumn("6M"),
            "R_12M": st.column_config.TextColumn("12M"),
            "R_36M": st.column_config.TextColumn("36M"),
        }
    )

    # ---- (3) Risk & Return Metrics ----
    st.markdown("<h5>(3) Risk & Return Metrics</h5>", unsafe_allow_html=True)

    rows = []
    ann = 252  # 연환산 기준 (거래일수)

    for c in ycols:
        s = price_df[c].dropna()
        if s.empty:
            continue
        ret = s.pct_change().dropna()
        if ret.empty:
            continue

        # CAGR
        cagr = (s.iloc[-1] / s.iloc[0]) ** (ann / len(ret)) - 1.0
        vol = ret.std() * (ann ** 0.5)

        # Sharpe
        sharpe = cagr / vol if vol > 0 else float("nan")

        # Sortino (하방 변동성)
        downside = ret[ret < 0]
        dstd = downside.std() * (ann ** 0.5) if not downside.empty else float("nan")
        sortino = cagr / dstd if dstd and dstd > 0 else float("nan")

        # MDD
        cum = (1 + ret).cumprod()
        mdd = (cum / cum.cummax() - 1.0).min()

        # Calmar
        calmar = cagr / abs(mdd) if mdd < 0 else float("nan")

        rows.append([
            pretty_label_with_fund(c),
            f"{cagr*100:.2f}%",
            f"{vol*100:.2f}%",
            f"{sharpe:.2f}",
            f"{sortino:.2f}",  # 🔑 추가된 항목
            f"{mdd*100:.2f}%",
            f"{calmar:.2f}"
        ])

    if rows:
        sumdf = pd.DataFrame(
            rows,
            columns=["Asset","CAGR","연변동성","Sharpe","Sortino","MDD","Calmar"]
        )
        st.dataframe(sumdf, use_container_width=True, hide_index=True)

    # ---- (4) 단일 자산 이동평균 + 캔들/거래량/RSI ----
    st.markdown("<h5>(4) Chart with Candlestick, SMA, Volume & RSI</h5>", unsafe_allow_html=True)

    options = [(pretty_label_with_fund(c), c) for c in ycols]  # (라벨, 코드)
    sel_idx = st.selectbox(
        "자산 선택",
        options=list(range(len(options))),
        format_func=lambda i: options[i][0],
        index=0,
        key="m_ma_one_idx"
    )
    one = options[sel_idx][1]                 # 내부 코드
    one_label = pretty_label_with_fund(one)   # 라벨  
    one_norm = to_yahoo_ticker(one)

    # ---- 데이터 가져오기 ----
    if is_kofia_code(one_norm):
        df_nav = fetch_kofia_nav_xml(one_norm, start, end)
        if df_nav.empty:
            st.warning("해당 펀드의 NAV 데이터를 불러올 수 없습니다.")
            st.stop()

        df = pd.DataFrame({
            "Date": df_nav["Date"],
            "Close": pd.to_numeric(df_nav[one_norm], errors="coerce")
        })
        df["Ticker"] = one_norm
        # High/Low/Volume 없음

    else:
        ohlcv = fetch_yf_ohlcv((one_norm,), start, end, use_adjust=True)
        if ohlcv.empty:
            st.warning("해당 자산의 OHLCV 데이터를 불러올 수 없습니다.")
            st.stop()
        df = ohlcv.copy()

    # ---- (NEW) 봉간격 선택 ----
    freq = st.radio("봉간격 선택", ["일봉", "주봉", "월봉"], horizontal=True)

    df_plot = df.copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    df_plot.set_index("Date", inplace=True)

    if {"Open","High","Low","Close"}.issubset(df_plot.columns):
        if freq == "주봉":
            df_plot = df_plot.resample("W").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
        elif freq == "월봉":
            df_plot = df_plot.resample("ME").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }).dropna()
    else:
        # 펀드 (Close만 존재)
        if freq == "주봉":
            df_plot = df_plot.resample("W").last().dropna()
        elif freq == "월봉":
            df_plot = df_plot.resample("ME").last().dropna()

    df_plot.reset_index(inplace=True)

    # ---- 공통 지표 계산 (SMA, RSI) ----
    if "Close" in df_plot.columns:
        for w in (20, 60, 120):
            df_plot[f"SMA{w}"] = df_plot["Close"].rolling(w).mean()

        def calc_RSI(series, period=14):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.ewm(span=period, adjust=False).mean()
            roll_down = down.ewm(span=period, adjust=False).mean()
            RS = roll_up / roll_down
            return 100 - (100 / (1 + RS))

        df_plot["RSI14"] = calc_RSI(df_plot["Close"], 14)

    # ---- 차트 생성 ----
    fig = go.Figure()

    if {"Open","High","Low","Close"}.issubset(df_plot.columns):
        # ✅ 주식/ETF → 캔들 차트
        fig.add_trace(go.Candlestick(
            x=df_plot["Date"],
            open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"],
            increasing_line_color="red", decreasing_line_color="blue",
            name=one_label
        ))
        if "Volume" in df_plot.columns:
            fig.add_trace(go.Bar(
                x=df_plot["Date"], y=df_plot["Volume"],
                name="Volume", yaxis="y2", opacity=0.4
            ))
        price_domain = [0.45, 1.0]
        vol_domain = [0.25, 0.4]

    else:
        # ✅ 펀드 → 선 차트
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["Close"],
            mode="lines", name=f"{one_label} (Close)", line=dict(color="blue")
        ))
        price_domain = [0.3, 1.0]
        vol_domain = None

    # ---- 이동평균선 ----
    for w in (20, 60, 120):
        if f"SMA{w}" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot[f"SMA{w}"],
                mode="lines", name=f"SMA{w}", line=dict(dash="dot")
            ))

    # ---- RSI ----
    if "RSI14" in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["RSI14"],
            mode="lines", name="RSI(14)",
            line=dict(color="purple", width=1.2),
            yaxis="y3"
        ))
        for lvl, clr in [(30, "blue"), (70, "red")]:
            fig.add_trace(go.Scatter(
                x=df_plot["Date"], y=[lvl]*len(df_plot),
                mode="lines", name=f"RSI {lvl}",
                line=dict(color=clr, dash="dash", width=1),
                yaxis="y3"
            ))

    # ---- 레이아웃 ----
    layout_args = dict(
        title=f"{one_label} — Chart with SMA, Volume & RSI",
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title="Price", domain=price_domain),
        yaxis3=dict(title="RSI", domain=[0.0, 0.2], range=[0,100], showgrid=True),
        height=850,
        margin=dict(l=10, r=120, t=30, b=20),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02)
    )
    if vol_domain:
        layout_args["yaxis2"] = dict(title="Volume", domain=vol_domain, showgrid=False)

    fig.update_layout(**layout_args)

    st.plotly_chart(fig, use_container_width=True)


    # ---- (5) Company / Fund Info ----
    st.markdown("<h5>(5) Company / Fund Info</h5>", unsafe_allow_html=True)

    # (3)에서 선택한 자산(one_norm)과 입력창 state를 동기화 → TSLA로 고정되는 문제 방지
    if st.session_state.get("m_info_symbol_src") != one_norm:
        st.session_state["m_info_symbol"] = one_norm
        st.session_state["m_info_symbol_src"] = one_norm

    c_fv1, c_fv2 = st.columns([0.22, 0.78])
    with c_fv1:
        # value= 사용 금지, state만 사용
        info_sym = st.text_input("티커/코드", key="m_info_symbol").strip().upper()
    with c_fv2:
        st.caption("국내(.KS/.KQ 또는 6자리) → 네이버, 펀드(KR…, 4~6자리 단축) → KOFIA, 그 외 → Finviz")

    if info_sym:
        try:
            # 1) KOFIA 펀드
            if is_kofia_code(info_sym):
                nav_df = fetch_kofia_nav_xml(info_sym, start, end)
                if not nav_df.empty:
                    fund_nm = (nav_df.get("fundNm").dropna().iloc[0]
                            if "fundNm" in nav_df.columns and nav_df["fundNm"].notna().any()
                            else info_sym)
                    st.markdown(f"**{fund_nm}** — KOFIA 등록 펀드")
                    tail = nav_df.dropna(subset=[info_sym]).tail(1)
                    if not tail.empty:
                        d_last = tail["Date"].dt.date.iloc[0]
                        v_last = float(tail[info_sym].iloc[0])
                        st.caption(f"최근 기준가: {d_last} · {v_last:,.2f}")
                else:
                    st.info("KOFIA 기준가 데이터를 찾지 못했습니다.")

            # 2) 국내 주식/ETF (.KS/.KQ 또는 6자리) → 네이버
            elif _is_krx_symbol(info_sym):
                profile_text, df_info = fetch_naver_overview(info_sym)
                st.markdown(profile_text)
                if not df_info.empty:
                    st.dataframe(df_info, use_container_width=True, hide_index=True, height=400)
                else:
                    st.caption("네이버 표 데이터를 찾지 못했습니다.")

            # 3) 그 외 → Finviz
            else:
                with st.spinner("회사 정보 수집 중..."):
                    profile_text, fv_table = fetch_finviz_company(info_sym)
                st.markdown(profile_text)
                if not fv_table.empty:
                    st.dataframe(fv_table, use_container_width=True, hide_index=True, height=400)
                else:
                    st.caption("표시할 Key Metrics가 없습니다.")
        except Exception as e:
            st.warning(f"정보 조회 실패: {e}")
    else:
        st.info("심볼/코드를 입력해 주세요.")

    # ---- 다운로드 ----
    st.markdown("#### 데이터 다운로드")
    dl_df = (rebase_pct(view, ycols) if mode in ("pct","pct_log","mdd") else view)[["Date"] + ycols].copy()
    csv_key = f"mkt_csv_{mode}_{start}_{end}"
    xlsx_key = f"mkt_xlsx_{mode}_{start}_{end}"
    st.download_button(
        "CSV 다운로드",
        data=dl_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"market_{mode}_{start}_{end}.csv",
        mime="text/csv",
        key=csv_key,
    )
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        dl_df.to_excel(w, sheet_name="market", index=False)
    bio.seek(0)
    st.download_button(
        "엑셀 다운로드",
        data=bio.getvalue(),
        file_name=f"market_{mode}_{start}_{end}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=xlsx_key,
    )

# ==================== TAB 2: Portfolio (원본 유지 + 혼합소스 지원) ====================
def guess_currency(ticker: str) -> str:
    t = ticker.upper()
    if is_kofia_code(t):  # KOFIA는 기본 KRW로 가정
        return "KRW"
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KRW"
    return "USD"

def rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    s = pd.Series(dates, index=dates)
    if freq == "M":
        nxt, cur = s.shift(-1).dt.to_period("M"), s.dt.to_period("M")
    elif freq == "Q":
        nxt, cur = s.shift(-1).dt.to_period("Q"), s.dt.to_period("Q")
    else:
        nxt, cur = s.shift(-1).dt.to_period("Y"), s.dt.to_period("Y")
    return (cur != nxt).fillna(True)

def build_portfolio_equity_missing_aware(prices, weights, mode="BH", fee_bps=0.0, reb_freq="M") -> pd.Series:
    tickers = [c for c in prices.columns if c in weights]
    W = pd.Series({t: float(weights[t]) for t in tickers})
    W = W / W.sum() if W.sum() > 0 else W

    valid_rows = prices.notna().any(axis=1)
    if not valid_rows.any(): return pd.Series(dtype=float, name="Portfolio")
    prices = prices.loc[valid_rows.idxmax():]
    rets = prices.pct_change(fill_method=None)

    dates = prices.index; V = 1.0; equity = []
    def apply_cost(value, w_prev, w_tgt):
        turnover = float(abs(w_prev - w_tgt).sum()) / 2.0
        cost = (fee_bps / 10000.0) * turnover
        return value * (1.0 - cost)

    first_mask = prices.iloc[0].notna()
    w_curr = ((W[first_mask] / W[first_mask].sum()).reindex(W.index, fill_value=0.0).values
              if first_mask.any() else W.values)

    rmask = rebalance_mask(dates, reb_freq) if mode == "RB" else pd.Series(False, index=dates)

    for i, dt in enumerate(dates):
        r = rets.iloc[i].fillna(0.0)[tickers].values
        port_ret = float((w_curr * r).sum())
        V *= (1.0 + port_ret); equity.append(V)
        if (1.0 + port_ret) != 0:
            w_curr = w_curr * (1.0 + r) / (1.0 + port_ret)
        if rmask.iloc[i]:
            w_tgt = (W / W.sum()).values if W.sum()!=0 else W.values
            V = apply_cost(V, w_curr, w_tgt); w_curr = w_tgt.copy()

    eq = pd.Series(equity, index=dates, name="Portfolio"); eq.iloc[0] = 1.0
    return eq

def portfolio_metrics(equity: pd.Series) -> dict:
    ret = equity.pct_change(fill_method=None).dropna()
    if ret.empty: return {}
    ann = 252
    cagr = (equity.iloc[-1]) ** (ann / len(ret)) - 1.0
    vol = ret.std() * math.sqrt(ann)
    sharpe = cagr / vol if vol > 0 else float("nan")
    dd = (equity / equity.cummax() - 1.0); mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd, "Calmar": calmar}

def tab_portfolio():
    st.title("Portfolio Analysis")

    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1, 1])
    with c1:
        start = st.date_input("시작일", value=date(2025,1,1),
                              min_value=date(2000,1,1), max_value=date.today(), key="p_start")
    with c2:
        end = st.date_input("종료일", value=date.today(),
                            min_value=date(2000,1,1), max_value=date.today(), key="p_end")
    with c3:
        base_ccy = st.selectbox("기준통화", ["USD", "KRW"], index=0, key="p_ccy")
    with c4:
        fee_bps = st.number_input("거래비용(bps)", min_value=0.0, max_value=200.0, step=1.0, value=0.0, key="p_fee")

    c5, c6 = st.columns([1,1])
    with c5:
        rb_mode = st.selectbox("리밸런싱", ["없음(바이앤홀드)", "매월", "분기", "매년"], index=0, key="p_rbmode")
    with c6:
        bench = st.text_input(
            "벤치마크(옵션, 예: SPY, QQQ, ^GSPC)",
            value="",                      # ✅ 기본값 비움 → 벤치마크 비표시
            placeholder="예: SPY, QQQ, ^GSPC",
            key="p_bench"
        )

    n1, n2, n3 = st.columns(3)
    with n1: name1 = st.text_input("포트폴리오 1 이름", value="포트폴리오 1안", key="p_name1")
    with n2: name2 = st.text_input("포트폴리오 2 이름", value="포트폴리오 2안", key="p_name2")
    with n3: name3 = st.text_input("포트폴리오 3 이름", value="포트폴리오 3안", key="p_name3")

    lite = st.checkbox("경량 모드(주간 리샘플)", value=False, help="브라우저가 느리면 켜 보세요.", key="p_lite")

    # 최초 1회만 기본값 세팅
    if "weights_df" not in st.session_state:
        st.session_state["weights_df"] = pd.DataFrame([
            {"티커":"SPY", "P1(%)":40.0, "P2(%)":33.0, "P3(%)":40.0},
            {"티커":"QQQ", "P1(%)":40.0, "P2(%)":33.0, "P3(%)":40.0},
            {"티커":"TLT", "P1(%)":20.0, "P2(%)":34.0, "P3(%)":20.0},
        ])

    with st.form("weights_form", clear_on_submit=False):
        h1, h2 = st.columns([1.0, 0.14])
        with h1:
            st.markdown("#### 자산 구성 (가로 입력: 1/2/3안)")
        with h2:
            apply_weights = st.form_submit_button("반영", use_container_width=True)

        # ✅ 항상 고정 순서로 정렬 + '이름' 열 추가(읽기전용)
        base_df = st.session_state["weights_df"].copy()
        for c in ["티커","P1(%)","P2(%)","P3(%)"]:
            if c not in base_df.columns:
                base_df[c] = None

        base_df["이름"] = base_df["티커"].apply(
            lambda x: pretty_label_with_fund(str(x)) if pd.notna(x) and str(x).strip() else ""
        )

        display_cols = ["티커","이름","P1(%)","P2(%)","P3(%)"]
        base_df = base_df.reindex(columns=display_cols)

        edited = st.data_editor(
            base_df,
            num_rows="dynamic",
            use_container_width=True,
            key="p_table_fixed",
            column_order=display_cols,      # ✅ 순서 고정
            disabled=["이름"],              # ✅ 읽기전용
            column_config={
                "티커":  st.column_config.TextColumn("티커 또는 펀드코드(KR..., 5~6자리 숫자)"),
                "이름":  st.column_config.TextColumn("종목/펀드명", help="야후/KOFIA/매핑에서 자동 추출", width="large"),
                "P1(%)": st.column_config.NumberColumn("포트폴리오1 (%)", step=1.0, format="%.2f"),
                "P2(%)": st.column_config.NumberColumn("포트폴리오2 (%)", step=1.0, format="%.2f"),
                "P3(%)": st.column_config.NumberColumn("포트폴리오3 (%)", step=1.0, format="%.2f"),
            },
        )

    if apply_weights:
        # 저장 시 내부 로직은 기존과 동일: 이름은 제외하고 고정 순서로만 저장
        keep = ["티커","P1(%)","P2(%)","P3(%)"]
        st.session_state["weights_df"] = edited[keep].copy()
        st.success("가중치를 반영했습니다.")

    # 이후 출력/그래프에서만 이름 반영
    st.write(f"📊 현재 포트폴리오 이름: {name1}, {name2}, {name3}")

    edit_df = st.session_state["weights_df"].copy()

    up = st.file_uploader("CSV 업로드(컬럼: 티커, P1(%), P2(%), P3(%))", type=["csv"], key="p_upload")
    if up:
        try:
            csvdf = pd.read_csv(up)
            need = {"티커","P1(%)","P2(%)","P3(%)"}
            if need.issubset(csvdf.columns):
                # ✅ 고정 순서로 저장 (표/그래프 모두 일관)
                ordered = csvdf[["티커","P1(%)","P2(%)","P3(%)"]].copy()
                st.session_state["weights_df"] = ordered
                edit_df = ordered
                st.info("업로드된 CSV를 사용합니다.")
            else:
                st.warning("CSV에 '티커, P1(%), P2(%), P3(%)' 컬럼이 필요합니다.")
        except Exception as e:
            st.warning(f"CSV 파싱 실패: {e}")

    edit_df = edit_df.dropna(subset=["티커"]).copy()
    edit_df["티커"] = edit_df["티커"].astype(str).str.upper().str.strip()

    w1 = {r["티커"]: float(r["P1(%)"]) for _, r in edit_df.iterrows() if pd.notna(r["P1(%)"]) and r["P1(%)"]!=0}
    w2 = {r["티커"]: float(r["P2(%)"]) for _, r in edit_df.iterrows() if pd.notna(r["P2(%)"]) and r["P2(%)"]!=0}
    w3 = {r["티커"]: float(r["P3(%)"]) for _, r in edit_df.iterrows() if pd.notna(r["P3(%)"]) and r["P3(%)"]!=0}
    if not (w1 or w2 or w3):
        st.warning("최소 한 개 안에 티커와 가중치를 입력하세요."); st.stop()

    for nm, w in [(name1,w1),(name2,w2),(name3,w3)]:
        if w:
            ssum = sum(w.values())
            if abs(ssum - 100.0) > 1e-6:
                st.caption(f"{nm} 가중치 합계: {ssum:.1f}% → 자동 정규화(합 100%)")

    tickers = tuple(sorted(set(list(w1.keys()) + list(w2.keys()) + list(w3.keys()))))
    with st.spinner(f"가격 불러오는 중... ({', '.join(tickers)})"):
        raw_px = fetch_prices_mixed(tickers, start, end, use_adjust=True)
    if raw_px.empty: st.warning("가격 데이터를 가져오지 못했습니다."); st.stop()

    starts = {}
    for col in [c for c in raw_px.columns if c != "Date"]:
        s = raw_px[["Date", col]].dropna()
        starts[col] = s["Date"].min().date() if not s.empty else None
    with st.expander("각 자산 데이터 시작일(상장/설정일 유사)"):
        info_df = pd.DataFrame({"티커": list(starts.keys()),
                                "데이터 시작일": [str(starts[k]) if starts[k] else "-" for k in starts]})
        st.table(info_df)

    px_df = raw_px.copy()
    if st.session_state.get("p_lite", False):
        px_df = px_df.set_index("Date").resample("W-FRI").last().reset_index()

    usdkrw = None
    if "KRW=X" not in px_df.columns:
        fx = fetch_yf_prices(("KRW=X",), start, end, use_adjust=False)
        if not fx.empty:
            fx_df = fx.rename(columns={"KRW=X":"USDKRW"})
            if st.session_state.get("p_lite", False):
                fx_df = fx_df.set_index("Date").resample("W-FRI").last().reset_index()
            usdkrw = fx_df.set_index("Date")["USDKRW"]

    base_ccy = st.session_state.get("p_ccy", "USD")
    if usdkrw is not None:
        tmp = px_df.set_index("Date")
        for c in [c for c in tmp.columns if c != "Date"]:
            is_kr = c.endswith(".KS") or c.endswith(".KQ") or is_kofia_code(c)
            if base_ccy == "KRW" and not is_kr:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce") * usdkrw
            elif base_ccy == "USD" and is_kr:
                tmp[c] = tmp[c] / usdkrw
        px_df = tmp.reset_index()
        px_df = reindex_fill_ffill_bfill(px_df, start, end, ffill_limit=5, kofia_ffill_limit=5)

    prices = px_df.set_index("Date")
    num_cols = [c for c in prices.columns if c != "Date"]
    prices[num_cols] = prices[num_cols].apply(pd.to_numeric, errors="coerce")
    first_valids = prices.apply(lambda s: s.first_valid_index())
    common_start = first_valids.dropna().max()
    if pd.notna(common_start):
        prices = prices.loc[common_start:]



    rb_mode = st.session_state.get("p_rbmode", "없음(바이앤홀드)")
    mode = "BH" if rb_mode.startswith("없음") else "RB"
    freq = "M" if rb_mode.startswith("매월") else ("Q" if rb_mode.startswith("분기") else "A")

    name1 = st.session_state.get("p_name1", "포트폴리오 1안")
    name2 = st.session_state.get("p_name2", "포트폴리오 2안")
    name3 = st.session_state.get("p_name3", "포트폴리오 3안")

    portfolios = []
    for nm, w in [(name1,w1),(name2,w2),(name3,w3)]:
        if not w:
            portfolios.append((nm, pd.Series(dtype=float))); continue
        W = pd.Series(w, dtype=float); W = W / (W.sum() if W.sum()!=0 else 1)
        eq = build_portfolio_equity_missing_aware(prices, W.to_dict(), mode=mode,
                                                  fee_bps=st.session_state.get("p_fee",0.0), reb_freq=freq)
        eq.name = nm; portfolios.append((nm, eq))

    bench = st.session_state.get("p_bench","SPY")
    bench_line = None
    bench_name = bench.strip().upper() if bench.strip() else None
    if bench_name:
        bpx = fetch_yf_prices((bench_name,), start, end, use_adjust=True)
        if not bpx.empty:
            if st.session_state.get("p_lite", False):
                bpx = bpx.set_index("Date").resample("W-FRI").last().reset_index()
            bser = bpx.set_index("Date")[bench_name]
            if usdkrw is not None:
                cur_kr = bench_name.endswith(".KS") or bench_name.endswith(".KQ")
                if base_ccy == "KRW" and not cur_kr: bser = bser * usdkrw
                elif base_ccy == "USD" and cur_kr: bser = bser / usdkrw
            bench_line = (bser / bser.dropna().iloc[0]).rename(bench_name)

    idx = None
    for _, s in portfolios:
        if s is not None and not s.empty:
            idx = s.index if idx is None else idx.union(s.index)
    if bench_line is not None:
        idx = bench_line.index if idx is None else idx.union(bench_line.index)
    if idx is None: idx = prices.index

    # --- Show Holdings 표시 토글 (Streamlit 1.25+ 에서는 st.toggle, 낮은 버전이면 st.checkbox 사용) ---
    try:
        show_individual = st.toggle("Show Holdings", value=False, help="포트폴리오와 구성 종목의 수익률/낙폭을 함께 표시")
    except Exception:
        show_individual = st.checkbox("Show Holdings", value=False, help="포트폴리오와 구성 종목의 수익률/낙폭을 함께 표시")

    # 포트폴리오 누적수익(%) DataFrame
    df_plot = pd.DataFrame(index=idx).sort_index()
    for nm, s in portfolios:
        if s is not None and not s.empty:
            df_plot[nm] = (s / s.iloc[0] - 1.0) * 100.0
    if bench_line is not None:
        df_plot[bench_line.name] = (bench_line / bench_line.dropna().iloc[0] - 1.0) * 100.0

    # ✅ (추가) Show Holdings이 켜져 있으면 구성 종목도 함께 표시
    if show_individual:
        # prices: 위에서 만든, 통화변환/리샘플/공통 시작일 정렬 후의 가격 테이블 (index=Date)
        indiv = prices[[c for c in prices.columns if c != "Date"]].copy()
        # 누적수익(%)로 변환
        for c in indiv.columns:
            s = indiv[c].dropna()
            if s.empty:
                continue
            ret = (indiv[c] / s.iloc[0] - 1.0) * 100.0
            label = pretty_label_with_fund(c)  # 보기 좋은 라벨
            df_plot[label] = ret.reindex(df_plot.index)

    df_plot = df_plot.reset_index().rename(columns={"index": "Date"})


    fig = px.line(df_plot, x="Date", y=[c for c in df_plot.columns if c != "Date"], render_mode="svg")
    fig.update_traces(line=dict(dash="solid"))
    fig.update_layout(margin=dict(l=10, r=110, t=10, b=10), height=480,
                      yaxis_title=f"누적 수익률 (%) — 기준통화: {base_ccy}",
                      uirevision="pf1", xaxis_rangeslider_visible=False)
    fig.update_yaxes(ticksuffix="%")
    for tr in fig.data: tr.legendgroup = tr.name

    last = df_plot.dropna().iloc[-1]; lx = last["Date"]
    for c in df_plot.columns[1:]:
        sc = df_plot[["Date", c]].dropna()
        if sc.empty: continue
        v_last = sc.iloc[-1][c]
        fig.add_trace(go.Scatter(x=[lx], y=[v_last], mode="markers+text",
                                 text=[f"{v_last:+.1f}%"], textposition="middle right",
                                 marker=dict(size=6), showlegend=False, hoverinfo="skip", legendgroup=c))
        imin = sc[c].idxmin()
        x_min, y_min = sc.loc[imin, "Date"], sc.loc[imin, c]
        fig.add_trace(go.Scatter(x=[x_min], y=[y_min], mode="markers+text",
                                 text=[f"{y_min:+.1f}%"], textposition="bottom right",
                                 marker=dict(size=8), showlegend=False, hoverinfo="skip", legendgroup=c))
    st.plotly_chart(fig, use_container_width=True)

    comp = pd.DataFrame(index=idx).sort_index()
    for nm, s in portfolios:
        if not s.empty:
            comp[nm] = (s / s.cummax() - 1.0) * 100.0

    if bench_line is not None:
        bench_equity = (bench_line / bench_line.dropna().iloc[0]).reindex(comp.index).ffill()
        bench_equity.iloc[0] = 1.0
        comp[bench_name] = (bench_equity / bench_equity.cummax() - 1.0) * 100.0

    # ✅ (추가) Show Holdings이 켜져 있으면 종목 MDD도 함께 표시
    if show_individual:
        indiv = prices[[c for c in prices.columns if c != "Date"]].copy()
        for c in indiv.columns:
            s = indiv[c].dropna()
            if s.empty:
                continue
            eq = (indiv[c] / s.iloc[0]).reindex(comp.index).ffill()   # 누적지수화
            label = pretty_label_with_fund(c)
            comp[label] = (eq / eq.cummax() - 1.0) * 100.0

    comp = comp.reset_index().rename(columns={"index":"Date"})
    mdd_cols = [c for c in comp.columns if c != "Date"]
    if mdd_cols:
        fig2 = px.line(comp, x="Date", y=mdd_cols, render_mode="svg")
        fig2.update_layout(margin=dict(l=10, r=110, t=10, b=10), height=300, yaxis_title="MDD (%)",
                           uirevision="pf2", xaxis_rangeslider_visible=False)
        fig2.update_yaxes(ticksuffix="%", rangemode="tozero")
        for tr in fig2.data: tr.legendgroup = tr.name
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 요약 지표")
    rows = []
    for nm, s in portfolios:
        if not s.empty:
            m = portfolio_metrics(s)
            rows.append([nm, f"{m['CAGR']*100:.2f}%", f"{m['Vol']*100:.2f}%", f"{m['Sharpe']:.2f}",
                         f"{m['MDD']*100:.2f}%", f"{m['Calmar']:.2f}"])
    if bench_line is not None:
        bench_eq = (bench_line / bench_line.dropna().iloc[0]).reindex(idx).ffill()
        bench_eq.iloc[0] = 1.0
        m = portfolio_metrics(bench_eq)
        rows.append([bench_name, f"{m['CAGR']*100:.2f}%", f"{m['Vol']*100:.2f}%", f"{m['Sharpe']:.2f}",
                     f"{m['MDD']*100:.2f}%", f"{m['Calmar']:.2f}"])
    if rows:
        sumdf = pd.DataFrame(rows, columns=["포트폴리오/벤치","CAGR","연변동성","Sharpe","MDD","Calmar"])
        st.table(sumdf)

    out = pd.DataFrame(index=idx).sort_index().rename_axis("Date").reset_index()
    for nm, s in portfolios:
        if not s.empty:
            out[f"{nm}_Value"] = s.reindex(out["Date"]).astype(float).values
            out[f"{nm}_Return(%)"] = (s.reindex(out["Date"]) / s.iloc[0] - 1.0).astype(float) * 100.0
            out[f"{nm}_Drawdown(%)"] = (s.reindex(out["Date"]) / s.cummax().reindex(out["Date"]) - 1.0).astype(float) * 100.0
    if bench_line is not None:
        bench_eq = (bench_line / bench_line.dropna().iloc[0]).reindex(idx).ffill()
        bench_eq.iloc[0] = 1.0
        out[f"{bench_name}_Value"] = bench_eq.reindex(out["Date"]).astype(float).values
        out[f"{bench_name}_Return(%)"] = (bench_eq.reindex(out["Date"]) / bench_eq.iloc[0] - 1.0).astype(float) * 100.0
        out[f"{bench_name}_Drawdown(%)"] = (bench_eq.reindex(out["Date"]) / bench_eq.cummax().reindex(out["Date"]) - 1.0).astype(float) * 100.0

    csv_key  = f"pf_csv_{base_ccy}_{st.session_state.get('p_start', date(2020,1,1))}_{st.session_state.get('p_end', date.today())}"
    xlsx_key = f"pf_xlsx_{base_ccy}_{st.session_state.get('p_start', date(2020,1,1))}_{st.session_state.get('p_end', date.today())}"

    st.download_button(
        "CSV 다운로드",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"portfolio_compare_{base_ccy}_{st.session_state.get('p_start', date(2020,1,1))}_{st.session_state.get('p_end', date.today())}.csv",
        mime="text/csv",
        key=csv_key,
    )

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        out.to_excel(w, sheet_name="portfolio", index=False)
    bio.seek(0)

    st.download_button(
        "엑셀 다운로드",
        data=bio.getvalue(),
        file_name=f"portfolio_compare_{base_ccy}_{st.session_state.get('p_start', date(2020,1,1))}_{st.session_state.get('p_end', date.today())}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=xlsx_key,
    )

# ==================== TAB 3: Analysis (글로벌·펀드 혼합) ====================
def _coerce_date(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "M":
        return pd.to_datetime(s).dt.tz_localize(None)
    out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if out.notna().any():
        return out.dt.tz_localize(None)
    s2 = s.astype(str).str.replace(".", "-", regex=False).str.strip()
    out2 = pd.to_datetime(s2, errors="coerce", format="%Y-%m-%d")
    return out2.dt.tz_localize(None)

def _to_number(x: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce")
    return pd.to_numeric(x.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                         errors="coerce")

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_yf_ohlcv(tickers: tuple, start, end, use_adjust=True) -> pd.DataFrame:
    cols_out = ["Date","Ticker","Open","High","Low","Close","Volume"]
    if not tickers:
        return pd.DataFrame(columns=cols_out)

    raw = yf.download(
        list(tickers),
        start=str(start),
        end=str(end + pd.Timedelta(days=1)),
        auto_adjust=use_adjust,
        progress=False,
        threads=True,
        actions=False,
        repair=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=cols_out)

    if isinstance(raw.columns, pd.MultiIndex):
        def pick(k): 
            return raw[k] if k in raw.columns.levels[0] else pd.DataFrame(index=raw.index)
        open_ = pick("Open"); high = pick("High"); low = pick("Low")
        close = pick("Close"); vol = pick("Volume")
        open_.index.name = close.index.name = "Date"

        dfO = open_.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Open")
        dfC = close.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Close")
        dfH = high.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="High")
        dfL = low.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Low")
        dfV = vol.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Volume")

        out = dfO.merge(dfC, on=["Date","Ticker"], how="left")\
                 .merge(dfH, on=["Date","Ticker"], how="left")\
                 .merge(dfL, on=["Date","Ticker"], how="left")\
                 .merge(dfV, on=["Date","Ticker"], how="left")
    else:
        tkr = list(tickers)[0]
        # ✅ 환율/지수 (Close만 있는 경우 보정)
        if "Close" in raw.columns and "Open" not in raw.columns:
            out = pd.DataFrame({
                "Date": raw.index,
                "Ticker": tkr,
                "Close": raw["Close"]
            }).reset_index(drop=True)
            return out   # 🔑 여기서 함수 끝냄

        # ✅ 일반 주식/ETF (OHLCV 모두 있음)
        out = pd.DataFrame({
            "Date": raw.index,
            "Ticker": tkr,
            "Open": raw.get("Open"),
            "High": raw.get("High"),
            "Low":  raw.get("Low"),
            "Close":raw.get("Close"),
            "Volume": raw.get("Volume"),
        })
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)

def build_pair_series(df_long: pd.DataFrame, t1: str, t2: str):
    p = df_long.pivot(index="Date", columns="Ticker", values="Close")
    if t1 not in p.columns or t2 not in p.columns:
        return pd.DataFrame()
    out = pd.DataFrame(index=p.index)
    out["ratio"] = p[t1] / p[t2]
    out["spread_pct"] = (p[t1]/p[t1].iloc[0] - p[t2]/p[t2].iloc[0]) * 100.0
    return out.dropna()

def compute_price_indicators(df_long: pd.DataFrame, ma_windows=(20,60,120), vol_window=60):
    if df_long.empty:
        return {"data": df_long.copy()}

    def f(g):
        g = g.sort_values("Date").copy()
        g["ret"] = g["Close"].pct_change(fill_method=None)
        g["vol_ann"] = g["ret"].rolling(vol_window).std() * (252 ** 0.5)
        base = g["Close"] / g["Close"].iloc[0]
        g["MDD"] = (base / base.cummax() - 1.0)
        if {"High","Low","Close"}.issubset(g.columns):
            prev_close = g["Close"].shift(1)
            tr = pd.concat([(g["High"]-g["Low"]).abs(),
                            (g["High"]-prev_close).abs(),
                            (g["Low"]-prev_close).abs()], axis=1).max(axis=1)
            g["ATR"] = tr.rolling(14).mean()
            g["ATR_pct"] = g["ATR"] / g["Close"]
        else:
            g["ATR_pct"] = pd.NA
        if "Volume" in g.columns and g["Volume"].notna().any():
            d = g["ret"].fillna(0.0).apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
            g["OBV"] = (d * g["Volume"].fillna(0)).cumsum()
            pv = (g["Close"] * g["Volume"].fillna(0)).astype(float)
            g["PV_Z"] = (pv - pv.rolling(60).mean()) / (pv.rolling(60).std() + 1e-12)
        else:
            g["OBV"] = pd.NA; g["PV_Z"] = pd.NA
        return g

    try:
        data = (
            df_long.groupby("Ticker", group_keys=False)
            .apply(lambda g: f(g).assign(Ticker=g.name), include_groups=False)
            .reset_index(drop=True)
        )
    except TypeError:
        data = (
            df_long.groupby("Ticker", group_keys=False)
            .apply(lambda g: f(g).assign(Ticker=g.name))
            .reset_index(drop=True)
        )
    return {"data": data}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices_mixed_long(tokens: tuple, start, end, use_adjust=True) -> pd.DataFrame:
    """
    야후 OHLCV(long) + KOFIA Close(long) 결합 → columns: Date, Ticker, Close, High, Low, Volume
    KOFIA는 Close만 채워짐(High/Low/Volume은 NaN).
    """
    cols_out = ["Date", "Ticker", "Close", "High", "Low", "Volume"]
    if not tokens:
        return pd.DataFrame(columns=cols_out)

    yf_list   = [t for t in tokens if not is_kofia_code(t)]
    fund_list = [t for t in tokens if is_kofia_code(t)]

    out_frames = []

    # 1) Yahoo OHLCV (long)
    if yf_list:
        out_frames.append(fetch_yf_ohlcv(tuple(yf_list), start, end, use_adjust=use_adjust))

    # 2) KOFIA NAV (Close만)
    for code in fund_list:
        try:
            df = fetch_kofia_nav_xml(code, start, end)

            # 비거나 Date 자체도 없으면 건너뜀
            if df is None or df.empty or "Date" not in df.columns:
                continue

            # 가격 컬럼 결정: 보통 code, 없으면 Date 제외 첫 컬럼
            price_col = code if code in df.columns else next(
                (c for c in df.columns if c != "Date"), None
            )
            if not price_col:
                st.info(f"KOFIA {code}: 가격 컬럼을 찾지 못해 건너뜀 (cols={list(df.columns)})")
                continue

            # 표준 컬럼 구성
            tmp = df[["Date", price_col]].copy()
            tmp = tmp.rename(columns={price_col: "Close"})
            tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
            tmp["Ticker"] = code
            tmp["High"] = pd.NA
            tmp["Low"] = pd.NA
            tmp["Volume"] = pd.NA

            out_frames.append(tmp[["Date", "Ticker", "Close", "High", "Low", "Volume"]])

        except Exception as e:
            st.warning(f"KOFIA 조회 실패({code}): {e}")


    if not out_frames:
        return pd.DataFrame(columns=cols_out)

    out = pd.concat(out_frames, ignore_index=True, sort=False)
    out = out[cols_out].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    num_cols = [c for c in out.columns if c not in ("Date","Ticker")]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    return out.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

def tab_research_global():
    st.title("가격 리서치 (글로벌·펀드 포함)")

    c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
    with c1:
        raw = st.text_input("티커/펀드코드(쉼표/공백 구분) — 예: 005930.KS, 069500.KS, SPY, ^N225, KR5370199261, 19926",
                            value="069500.KS, SPY, ^KS11")
    with c2:
        start = st.date_input("시작일", value=date(2021,1,1), min_value=date(2000,1,1), max_value=date.today())
    with c3:
        end = st.date_input("종료일", value=date.today(), min_value=date(2000,1,1), max_value=date.today())

    use_adj = st.checkbox("조정가격(배당/분할 반영) 사용(야후에만 해당)", value=True)
    parsed = [t for t in re.split(r"[,\s]+", raw.upper()) if t.strip()]

    st.markdown("**펀드 NAV 파일 업로드(CSV/XLSX)** — 컬럼 예시: `일자, 기준가` 또는 `Date, Close`")
    fund_files = st.file_uploader("여러 개 업로드 가능", type=["csv","xls","xlsx"], accept_multiple_files=True)

    with st.spinner("데이터 불러오는 중..."):
        yf_fd_long = fetch_prices_mixed_long(tuple(parsed), start, end, use_adjust=use_adj) if parsed else \
                     pd.DataFrame(columns=["Date","Ticker","Close","High","Low","Volume"])

        # 업로드 파일은 Ticker를 파일명으로 부여
        extra_frames = []
        for f in fund_files or []:
            try:
                df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
                cols = {c.strip(): c for c in df.columns}
                date_col  = next((cols[c] for c in cols if c.lower() in ("date","일자","기준일","기준일자")), None)
                close_col = next((cols[c] for c in cols if c.lower() in ("close","기준가","nav","가격")), None)

                if not (date_col and close_col):
                    st.warning(f"파일 형식 인식 실패: {f.name} (필요 컬럼: Date/일자 + Close/기준가)")
                    continue

                tmp = df[[date_col, close_col]].copy()
                tmp.columns = ["Date","Close"]
                tmp["Date"]  = pd.to_datetime(tmp["Date"], errors="coerce")
                tmp["Close"] = pd.to_numeric(tmp["Close"], errors="coerce")
                tmp["Ticker"] = os.path.splitext(os.path.basename(f.name))[0].upper()
                tmp["High"] = pd.NA; tmp["Low"] = pd.NA; tmp["Volume"] = pd.NA
                extra_frames.append(tmp[["Date","Ticker","Close","High","Low","Volume"]])

            except Exception as e:
                st.warning(f"파일 파싱 실패({f.name}): {e}")

    frames = [df for df in [yf_fd_long] + extra_frames if df is not None and not df.empty]
    base_long = (pd.concat(frames, ignore_index=True)
                 if frames else pd.DataFrame(columns=["Date","Ticker","Close","High","Low","Volume"]))
    if base_long.empty:
        st.warning("유효한 데이터가 없습니다. 티커/파일/기간을 확인하세요.")
        return

    data = compute_price_indicators(base_long)["data"]

    tB, tC, tD = st.tabs(["⚠️ 변동성·MDD·ATR/OBV", "🔗 상관·페어", "🧪 시나리오"])

    with tB:
        all_tickers = sorted(data["Ticker"].unique())
        sel2 = st.multiselect("자산 선택", options=all_tickers,
                              default=all_tickers[:min(4,len(all_tickers))], key="rgB_sel")
        if sel2:
            disp = data[data["Ticker"].isin(sel2)].copy()
            fig1 = px.line(disp, x="Date", y="vol_ann", color="Ticker",
                           title="롤링 변동성(연율화, 창=60)", render_mode="svg")
            fig1.update_layout(height=300, margin=dict(l=10,r=110,t=30,b=10),
                               uirevision="rgB1", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig1, use_container_width=True)

            disp["MDD(%)"] = disp["MDD"] * 100.0
            fig2 = px.line(disp, x="Date", y="MDD(%)", color="Ticker",
                           title="MDD(%) — 낮을수록 낙폭 큼", render_mode="svg")
            fig2.update_layout(height=300, margin=dict(l=10,r=110,t=30,b=10),
                               uirevision="rgB2", xaxis_rangeslider_visible=False)
            fig2.update_yaxes(ticksuffix="%", rangemode="tozero")
            st.plotly_chart(fig2, use_container_width=True)

            if "ATR_pct" in disp and disp["ATR_pct"].notna().any():
                disp["ATR%(%)"] = disp["ATR_pct"] * 100.0
                fig3 = px.line(disp, x="Date", y="ATR%(%)", color="Ticker",
                               title="ATR% (가격 대비 평균 진폭)", render_mode="svg")
                fig3.update_layout(height=280, margin=dict(l=10,r=110,t=30,b=10),
                                   uirevision="rgB3", xaxis_rangeslider_visible=False)
                fig3.update_yaxes(ticksuffix="%")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("High/Low가 없는 자산만 선택되어 ATR%는 생략됨.")

            if "OBV" in disp and disp["OBV"].notna().any():
                voltab1, voltab2 = st.tabs(["OBV", "Price×Volume Z-score"])
                with voltab1:
                    fig4 = px.line(disp, x="Date", y="OBV", color="Ticker", title="OBV", render_mode="svg")
                    fig4.update_layout(height=260, margin=dict(l=10,r=110,t=30,b=10),
                                       uirevision="rgB4", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig4, use_container_width=True)
                with voltab2:
                    fig5 = px.line(disp, x="Date", y="PV_Z", color="Ticker", title="PV Z-score (창=60)", render_mode="svg")
                    fig5.update_layout(height=260, margin=dict(l=10,r=110,t=30,b=10),
                                       uirevision="rgB5", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig5, use_container_width=True)
            else:
                st.caption("선택 자산에 거래량(Volume)이 없어 거래대금 지표는 숨김 처리됨.")

    with tC:
        pvt_ret = data.pivot(index="Date", columns="Ticker", values="ret").dropna(how="all")
        win = st.slider("상관계수 윈도우(거래일)", 30, 252, 120, 10)
        cor = pvt_ret.tail(win).corr().round(2) if not pvt_ret.empty else pd.DataFrame()
        st.markdown("**상관행렬 (최근 윈도우)**")
        st.dataframe(cor)

        c1, c2 = st.columns(2)
        with c1:
            t1 = st.selectbox("자산 1", options=sorted(data["Ticker"].unique()), key="pair_t1")
        with c2:
            t2 = st.selectbox("자산 2", options=[x for x in sorted(data["Ticker"].unique()) if x != t1], key="pair_t2")
        pair = build_pair_series(base_long, t1, t2)
        if not pair.empty:
            fig6 = px.line(pair.reset_index(), x="Date", y="ratio", title=f"비율 {t1}/{t2}", render_mode="svg")
            fig6.update_layout(height=280, margin=dict(l=10,r=110,t=30,b=10),
                               uirevision="rgC1", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig6, use_container_width=True)

            fig7 = px.line(pair.reset_index(), x="Date", y="spread_pct", title=f"스프레드(리베이스%) {t1} vs {t2}", render_mode="svg")
            fig7.update_layout(height=280, margin=dict(l=10,r=110,t=30,b=10),
                               uirevision="rgC2", xaxis_rangeslider_visible=False)
            fig7.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("선택한 두 자산의 가격 데이터를 동시에 가져오지 못했습니다.")

    with tD:
        st.markdown("**가격 충격(±%) 시나리오 — 즉시 손익 계산**")
        uniq = sorted(data["Ticker"].unique())
        shocks = {}
        cols = st.columns(min(4, max(1,len(uniq))))
        for i, t in enumerate(uniq):
            with cols[i % len(cols)]:
                shocks[t] = st.slider(f"{t} 충격(%)", -30, 30, 0, 1) / 100.0
        if shocks:
            show = pd.DataFrame({"Ticker": list(shocks.keys()),
                                 "즉시손익(%)": [f"{v*100:+.1f}%" for v in shocks.values()]})
            st.dataframe(show.set_index("Ticker"))

# -------------------- 상단 검색바 --------------------
st.markdown("""
<style>
  .gs-row .stButton>button { height: 32px; padding: 4px 12px; }
  .gs-row div[data-testid="stTextInput"] input { height: 32px; padding: 4px 10px; }
  div[data-testid="stHorizontalBlock"] { margin-bottom: 0.25rem !important; }
</style>
""", unsafe_allow_html=True)

with st.container():
    c_label, c_input, c_btn = st.columns([0.10, 0.70, 0.12], gap="small")
    with c_label:
        st.markdown("**티커검색**")
    with c_input:
        q_global = st.text_input(
            "티커검색",
            key="g_search_query",
            placeholder="예: SPY, ^KS11, 005930.KS, kospi",
            label_visibility="collapsed",
        )
    with c_btn:
        search_clicked = st.button("검색", key="g_search_btn", use_container_width=True)

if search_clicked and q_global.strip():
    with st.spinner("야후에서 검색 중..."):
        results = yahoo_search(q_global, quotes_count=10)
    if not results:
        st.info("검색 결과가 없습니다.")
    else:
        st.caption("검색 결과")
        for item in results:
            sym  = item.get("symbol", "")
            name = item.get("shortname") or item.get("longname") or ""
            exch = item.get("exchDisp") or ""
            qt   = item.get("quoteType") or ""
            st.markdown(f"**{sym}** — {name} · {exch} · {qt}")

def safe_tab(render_fn, name: str):
    try:
        render_fn()
    except Exception as e:
        st.error(f"[{name}] 탭 렌더링 중 오류가 발생했습니다.")
        st.exception(e)

# 탭 생성
tab1, tab2, tab3 = st.tabs(["Market", "Portfolio", "Analysis"])
with tab1:  safe_tab(tab_market, "Market")
with tab2:  safe_tab(tab_portfolio, "Portfolio")
with tab3:  safe_tab(tab_research_global, "Analysis")


# 깃허브 사이트
# https://github.com/anfwonil/investment4us

# 배포
# cd C:\Users\woori\Desktop\top10
# git add -A
# git commit -m "chore: update data and backup folders"
# git push

# git pull --rebase origin main
# git push origin main


#   g
# git commit -m "update: latest changes from run_update"
# git push origin main
   



# git add requirements.txt
# git commit -m "chore: update requirements.txt (add lxml)"

# 실행 참고:   KR5370199261
# cd C:\Users\woori\Desktop\top10
# & "C:\Users\woori\anaconda3\python.exe" -m streamlit run ".\app.py"