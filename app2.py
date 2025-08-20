# app.py
# pip install -U streamlit pandas yfinance plotly xlsxwriter

import time
from datetime import date, datetime
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Dashboard - Safe Base", layout="wide")

# 1) 유틸: 티커 전처리(공백/쉼표/줄바꿈 모두 허용)
def parse_tickers(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip() for chunk in raw.splitlines() for p in chunk.split(",")]
    return [p for p in parts if p]

# 2) 야후 404/빈데이터 대비: 단일 티커 히스토리 안전 호출 + 재시도
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_one_ticker(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    # 재시도: 일시적 404/네트워크 문제 방지 (3회)
    last_err = None
    for attempt in range(3):
        try:
            # 주의: auto_adjust 기본 True (yfinance>=0.2.40)
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            # 일부 티커는 존재해도 특정 구간이 비어 있을 수 있음 → 빈 DF 처리
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.reset_index().rename(columns={"Date": "date"})  # 날짜 컬럼 통일
                df["ticker"] = ticker
                return df
            # 빈 데이터도 ‘성공’으로 보되, 호출자는 빈 DF로 판단
            return pd.DataFrame(columns=["date","Open","High","Low","Close","Adj Close","Volume","ticker"])
        except Exception as e:
            last_err = e
            time.sleep(0.7)  # 살짝 간격
    # 3회 실패 → 에러를 문자열로 포장해서 상위에서 알림
    raise RuntimeError(f"[{ticker}] 데이터 요청 실패: {last_err}")

# 3) 여러 티커 안전 수집
def fetch_many(tickers: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    frames = []
    issues: Dict[str,str] = {}
    for t in tickers:
        try:
            df = fetch_one_ticker(t, start, end, interval)
            frames.append(df)
            if df.empty:
                issues[t] = "빈 데이터 (기간 밖이거나 상장 이전, 휴장 등)"
        except Exception as e:
            issues[t] = str(e)
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["date","Open","High","Low","Close","Adj Close","Volume","ticker"])
    return out, issues

# 4) 최고/최저 포인트 표시용 헬퍼
def minmax_points(df: pd.DataFrame, value_col: str = "Adj Close"):
    idx_max = df[value_col].idxmax()
    idx_min = df[value_col].idxmin()
    if pd.isna(idx_max) or pd.isna(idx_min):
        return None, None
    p_max = df.loc[idx_max, ["date", value_col]].to_dict()
    p_min = df.loc[idx_min, ["date", value_col]].to_dict()
    return p_max, p_min

# 5) 사이드바
st.sidebar.header("⚙️ 설정")
default_tickers = "^GSPC, ^IXIC, ^KS11"  # S&P500, Nasdaq, KOSPI
raw_tickers = st.sidebar.text_area("티커(쉼표/줄바꿈 구분):", value=default_tickers, height=90)

# 날짜: 기본 2021-01-01 (요청 이력 반영), 최저 시작은 2000-01-01로 가이드
min_start = date(2000, 1, 1)
default_start = date(2021, 1, 1)
start_date = st.sidebar.date_input("시작일", value=default_start, min_value=min_start, max_value=date.today())
end_date = st.sidebar.date_input("종료일", value=date.today(), min_value=start_date, max_value=date.today())

interval = st.sidebar.selectbox("빈도", ["1d","1wk","1mo"], index=0)

st.sidebar.info(
    "📌 KR 종목은 종종 접미사 필요: 코스피 `.KS`, 코스닥 `.KQ`\n"
    "예) 삼성전자 `005930.KS`, 카카오 `035720.KS`\n"
    "지수 예) 코스피 `^KS11`, S&P500 `^GSPC`"
)

tickers = parse_tickers(raw_tickers)

st.title("📊 야후 파이낸스 대시보드 (404 안전판)")
st.caption("빈 데이터/404/상장 이전/휴장 처리, 최고·최저 포인트, 다운로드 포함")

# 6) 데이터 로드
if not tickers:
    st.warning("티커를 입력하세요. 예: `^GSPC, ^IXIC, ^KS11`")
    st.stop()

with st.spinner("데이터 불러오는 중..."):
    df_all, issues = fetch_many(tickers, start_date.isoformat(), end_date.isoformat(), interval)

# 7) 이슈 리포트
if issues:
    st.subheader("⚠️ 수집 이슈")
    for t, msg in issues.items():
        st.write(f"- **{t}**: {msg}")

if df_all.empty:
    st.error("유효한 데이터가 없습니다. 티커/기간/빈도를 확인하세요.")
    st.stop()

# 8) 가격 탭
tab1, tab2 = st.tabs(["가격(최고/최저 표기)", "변화율(최고/최저 수익률)"])

with tab1:
    st.subheader("종가(Adj Close) 차트")
    # 각 티커별 라인 + 각자 최고/최저 마커
    fig = go.Figure()
    for t, g in df_all.groupby("ticker"):
        g = g.sort_values("date")
        fig.add_trace(go.Scatter(
            x=g["date"], y=g["Adj Close"], mode="lines", name=t, hovertemplate="%{x}<br>%{y:.2f}<extra>"+t+"</extra>"
        ))
        p_max, p_min = minmax_points(g, "Adj Close")
        if p_max:
            fig.add_trace(go.Scatter(
                x=[p_max["date"]], y=[p_max["Adj Close"]],
                mode="markers+text", text=["최고"], textposition="top center",
                marker_symbol="triangle-up", marker_size=10, name=f"{t} 최고"
            ))
        if p_min:
            fig.add_trace(go.Scatter(
                x=[p_min["date"]], y=[p_min["Adj Close"]],
                mode="markers+text", text=["최저"], textposition="bottom center",
                marker_symbol="triangle-down", marker_size=10, name=f"{t} 최저"
            ))
    fig.update_layout(hovermode="x unified", margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # 원본 데이터 다운로드
    csv = df_all.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV 다운로드", data=csv, file_name="prices_raw.csv", mime="text/csv")

with tab2:
    st.subheader("변화율(일/주/월 수익률) 차트")
    # 빈도별 수익률 계산(단순 pct_change)
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("date").copy()
        df["ret"] = df["Adj Close"].pct_change()
        return df

    df_ret = df_all.groupby("ticker", group_keys=False).apply(compute_returns).dropna(subset=["ret"])
    if df_ret.empty:
        st.info("수익률을 계산할 데이터가 부족합니다.")
    else:
        fig2 = go.Figure()
        for t, g in df_ret.groupby("ticker"):
            fig2.add_trace(go.Scatter(
                x=g["date"], y=(g["ret"]*100), mode="lines", name=t,
                hovertemplate="%{x}<br>%{y:.2f}%<extra>"+t+"</extra>"
            ))
            # 수익률 최고/최저 표시
            idx_max = g["ret"].idxmax()
            idx_min = g["ret"].idxmin()
            if pd.notna(idx_max):
                fig2.add_trace(go.Scatter(
                    x=[g.loc[idx_max,"date"]], y=[g.loc[idx_max,"ret"]*100],
                    mode="markers+text", text=["최고수익률"], textposition="top center",
                    marker_symbol="star", marker_size=10, name=f"{t} 최고수익률"
                ))
            if pd.notna(idx_min):
                fig2.add_trace(go.Scatter(
                    x=[g.loc[idx_min,"date"]], y=[g.loc[idx_min,"ret"]*100],
                    mode="markers+text", text=["최저수익률"], textposition="bottom center",
                    marker_symbol="x", marker_size=10, name=f"{t} 최저수익률"
                ))
        fig2.update_layout(hovermode="x unified", yaxis_title="수익률(%)", margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # 수익률 데이터 다운로드
        csv2 = df_ret[["date","ticker","ret"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 수익률 CSV 다운로드", data=csv2, file_name="returns.csv", mime="text/csv")

# 9) 디버그 섹션: 실제 요청 구간/최초·최후 날짜 확인
with st.expander("🔎 디버그 정보"):
    st.write("요청 파라미터:", {"tickers": tickers, "start": start_date.isoformat(), "end": end_date.isoformat(), "interval": interval})
    by_ticker_range = (
        df_all.groupby("ticker")
              .agg(first_date=("date","min"), last_date=("date","max"), rows=("date","count"))
              .reset_index()
    )
    st.dataframe(by_ticker_range)

st.success("로드 완료 ✅")
