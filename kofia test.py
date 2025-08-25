# kofia_probe.py
# pip install -U streamlit pandas requests lxml

import re
from datetime import date
import pandas as pd
import streamlit as st
import requests
from lxml import etree
from io import BytesIO

st.set_page_config(page_title="KOFIA NAV Probe", layout="wide")

KOFIA_URL = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES/"

def is_kofia_code(tok: str) -> bool:
    if not tok: return False
    t = tok.strip().upper()
    return bool(re.fullmatch(r"(KR\d{10}|\d{4,6})", t))

def _yyyymmdd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")

def _parse_date_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()

    dt_a = pd.to_datetime(raw, errors="coerce", format="%Y%m%d")
    if dt_a.notna().sum() >= max(3, int(len(raw) * 0.3)):
        return dt_a

    norm = (raw.str.replace(".", "-", regex=False)
                .str.replace("/", "-", regex=False))
    norm2 = norm.str.extract(r"(\d{4}-?\d{2}-?\d{2})", expand=False).fillna(norm)
    dt_b = pd.to_datetime(norm2, errors="coerce")

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

def fetch_kofia_raw(code: str, start, end) -> pd.DataFrame:
    payload = _kofia_xml_payload(_yyyymmdd(start), _yyyymmdd(end), code)
    headers = {
        "Content-Type": "text/xml; charset=UTF-8",
        "Accept": "text/xml,application/xml,text/plain,*/*",
        "User-Agent": "Mozilla/5.0",
    }
    r = requests.post(KOFIA_URL, data=payload.encode("utf-8"), headers=headers, timeout=20)
    r.raise_for_status()

    root = etree.fromstring(r.content)
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

    return pd.DataFrame(rows)

def auto_pick_date_price(df: pd.DataFrame):
    def _is_tmpv(col):  return bool(re.fullmatch(r"tmpV\d+", str(col)))
    def _name_hit(col):
        s = str(col).lower()
        return any(k in s for k in ["date","dt","일자","기준","nav","가격","close","stnav","stdnav","base"])
    cand_cols = [c for c in df.columns if _is_tmpv(c) or _name_hit(c)]
    if not cand_cols:
        cand_cols = list(df.columns)

    def _clean_str(s: pd.Series) -> pd.Series:
        return (s.astype(str)
                 .str.replace(",", "", regex=False)
                 .str.replace(" ", "", regex=False)
                 .str.strip())

    stats = {}
    for c in cand_cols:
        raw = df[c]
        cs  = _clean_str(raw)
        as_num = pd.to_numeric(cs, errors="coerce")
        as_dt  = _parse_date_series(raw)
        stats[c] = {
            "num_ratio": as_num.notna().mean(),
            "num_std": float(as_num.std(skipna=True)) if as_num.notna().any() else 0.0,
            "dt_ratio": as_dt.notna().mean(),
            "dt_count": int(as_dt.notna().sum()),
            "ymd8_ratio": cs.str.fullmatch(r"\d{8}").fillna(False).mean(),
            "uniq_vals": cs.replace("", pd.NA).dropna().nunique(),
            "name": str(c).lower(),
        }

    def date_score(c):
        s = stats[c]
        return s["dt_ratio"] + 0.30*s["ymd8_ratio"] + 0.0005*s["dt_count"]

    date_col = max(cand_cols, key=date_score) if cand_cols else None
    if date_col and stats[date_col]["dt_count"] < 2:
        date_col = None
    def price_score(c):
        s = stats[c]
        name_boost = 0.15 if any(k in s["name"] for k in ["nav","기준","close","std","stnav"]) else 0.0
        return s["num_ratio"] + 0.0001*s["num_std"] + name_boost
    price_candidates = [c for c in cand_cols if c != date_col and stats[c]["uniq_vals"] >= 2]
    price_col = max(price_candidates, key=price_score) if price_candidates else None
    if date_col is None:
        date_col = max(cand_cols, key=lambda c: stats[c]["ymd8_ratio"]) if cand_cols else None
    if price_col is None:
        price_col = max([c for c in cand_cols if c != date_col],
                        key=lambda c: stats[c]["num_ratio"], default=None)
    return date_col, price_col, stats

st.title("KOFIA NAV Probe (원본 검증 전용)")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    code = st.text_input("펀드코드 (ISIN KR… 또는 5~6자리)", value="KR5370199261")
with c2:
    start = st.date_input("시작일", value=date(2021,1,1), min_value=date(2000,1,1))
with c3:
    end = st.date_input("종료일", value=date.today(), min_value=date(2000,1,1))

run = st.button("조회", type="primary")
st.caption("※ 이 도구는 원본 응답을 직접 확인해 '날짜/기준가' 컬럼을 정확히 고정하기 위한 용도입니다.")

if run:
    if not is_kofia_code(code):
        st.error("코드 형식이 아닙니다. ISIN(KR로 시작) 또는 4~6자리 숫자.")
        st.stop()
    with st.spinner("KOFIA 원본 조회중…"):
        try:
            raw = fetch_kofia_raw(code, start, end)
        except Exception as e:
            st.error(f"요청 실패: {e}")
            st.stop()
    if raw.empty:
        st.warning("빈 응답입니다. 기간/코드 확인.")
        st.stop()

    st.subheader("1) 원본 컬럼")
    st.code(str(list(raw.columns)), language="python")

    st.subheader("2) 원본 head/tail")
    st.dataframe(pd.concat([raw.head(5), raw.tail(5)]), use_container_width=True)

    st.subheader("3) 자동 매핑 결과(추정)")
    dcol, pcol, stats = auto_pick_date_price(raw)
    st.write(f"- 후보 Date 컬럼(추정): **{dcol}**")
    st.write(f"- 후보 Price 컬럼(추정): **{pcol}**")

    # 통계 테이블
    stat_df = (pd.DataFrame(stats).T
               .sort_values("num_ratio", ascending=False)
               .reset_index().rename(columns={"index":"col"}))
    st.dataframe(stat_df, use_container_width=True)

    # 미리보기(보정 전/후)
    if dcol and pcol:
        out = raw[[dcol, pcol]].copy()
        out.columns = ["Date", code]
        out["Date"] = _parse_date_series(out["Date"])
        out[code] = pd.to_numeric(out[code].astype(str).str.replace(",","").str.replace(" ",""), errors="coerce")
        out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

        st.subheader("4) 추정 매핑 프리뷰 (보정 전)")
        st.dataframe(pd.concat([out.head(5), out.tail(5)]), use_container_width=True)

        # /10000 보정 여부 자동 판단 및 토글
        median_val = out[code].dropna().median() if out[code].notna().any() else None
        suggest_scale = bool(median_val and median_val > 1000)
        use_scale = st.checkbox(f"/10000 보정 적용 (중앙값 {median_val:,.2f} → 권장={suggest_scale})",
                                value=suggest_scale)
        if use_scale:
            out[code] = out[code] / 10000.0

        st.subheader("5) (선택) 보정 후 프리뷰")
        st.dataframe(pd.concat([out.head(5), out.tail(5)]), use_container_width=True)

        # 저장
        bio = BytesIO()
        out.to_csv(bio, index=False, encoding="utf-8-sig")
        bio.seek(0)
        st.download_button(
            "CSV 다운로드 (매핑 결과)",
            data=bio.getvalue(),
            file_name=f"kofia_{code}_{start}_{end}.csv",
            mime="text/csv",
        )
    else:
        st.error("자동 식별 실패: 어느 컬럼이 날짜/가격인지 직접 지정이 필요합니다.")
