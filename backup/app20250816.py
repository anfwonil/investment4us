# app.py
# pip install streamlit pandas plotly yfinance xlsxwriter
import os, re
from io import BytesIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="글로벌 시장 대시보드", layout="wide")

# -------------------- 경로 --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "market_timeseries.csv")
META_CSV = os.path.join(BASE_DIR, "data", "meta.csv")

# -------------------- 비번(선택) --------------------
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

# -------------------- 세션 기본값 --------------------
if "extra_tickers" not in st.session_state:
    st.session_state["extra_tickers"] = []
if "ycols" not in st.session_state:
    st.session_state["ycols"] = []

# -------------------- 로더/유틸 --------------------
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

@st.cache_data(ttl=600, show_spinner=False)
def fetch_yf_prices(tickers: tuple, start, end, use_adjust=True) -> pd.DataFrame:
    """야후에서 추가 티커 다운로드. 반환: Date + 각 티커 컬럼"""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(
        list(tickers),
        start=str(start),
        end=str(end + pd.Timedelta(days=1)),  # end는 배타
        auto_adjust=use_adjust,
        progress=False,
        threads=True,
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

def rebase_pct(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """0%부터 시작하는 누적수익률(%)"""
    out = frame[["Date"] + cols].copy()
    for c in cols:
        s = out[c].dropna()
        out[c] = ((out[c] / s.iloc[0]) - 1) * 100.0 if not s.empty else pd.NA
    return out

def drawdown_pct(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """최대 낙폭(MDD) 시계열 (%): (가격/누적최고 - 1)*100, 항상 0 이하"""
    out = frame[["Date"] + cols].copy()
    for c in cols:
        s = out[c].astype(float)
        peak = s.cummax()
        out[c] = (s / peak - 1.0) * 100.0
    return out

def format_tail_value(v, mode_key: str) -> str:
    if pd.isna(v):
        return ""
    return f"{v:+.1f}%" if mode_key in ("pct", "mdd") else f"{v:,.1f}"

# -------------------- 기본 데이터 --------------------
if not os.path.exists(DATA_CSV):
    st.error(f"데이터 파일이 없습니다: {DATA_CSV}\n먼저 update.py를 실행하세요.")
    st.stop()

base = load_base_data(DATA_CSV)
last_updated = load_meta(META_CSV)

st.title("글로벌 시장 대시보드")
if last_updated:
    st.caption(f"마지막 업데이트(KST): {last_updated} · 기본 데이터는 하루 1회 update.py로 갱신")

# -------------------- 기간/티커 입력 --------------------
min_d, max_d = base["Date"].min().date(), base["Date"].max().date()
default_start = max(min_d, (max_d - pd.Timedelta(days=365)))

left, right = st.columns([1.4, 1])
with left:
    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("시작일", value=default_start, min_value=min_d, max_value=max_d, key="start")
    with c2:
        end = st.date_input("종료일", value=max_d, min_value=min_d, max_value=max_d, key="end")

with right:
    st.markdown("**사용자 지정 티커 추가** (쉼표/스페이스 구분)")
    tickers_text = st.text_input("예: F, TSLA, QQQ, 005930.KS", value="", placeholder="티커 입력 후 버튼 클릭")
    use_adj = st.checkbox("조정가격 사용(배당/액면 반영)", value=True)
    fetch_clicked = st.button("추가 티커 불러오기", type="primary")

# -------------------- 기본 데이터(기간 필터) --------------------
mask = (base["Date"].dt.date >= start) & (base["Date"].dt.date <= end)
view = base.loc[mask].copy()

# 저장된 추가 티커 자동 병합
saved = tuple(st.session_state.get("extra_tickers", []))
if saved:
    fetched_saved = fetch_yf_prices(saved, start, end, use_adjust=use_adj)
    if not fetched_saved.empty:
        view = pd.merge(view, fetched_saved, on="Date", how="outer").sort_values("Date")

# -------------------- 버튼: 새 티커 즉시 병합 --------------------
if fetch_clicked:
    parsed = [t.upper() for t in re.split(r"[,\s]+", tickers_text) if t.strip()]
    already = set(view.columns) | set(st.session_state["extra_tickers"])
    new_only = [t for t in parsed if t not in already]
    if not new_only:
        st.info("새로 추가할 티커가 없습니다(이미 추가되었거나 공란).")
    else:
        with st.spinner(f"야후파이낸스에서 불러오는 중... ({', '.join(new_only)})"):
            fetched_now = fetch_yf_prices(tuple(new_only), start, end, use_adjust=use_adj)
        if fetched_now.empty:
            st.warning("유효한 데이터가 없습니다. 티커를 다시 확인해 주세요.")
        else:
            view = pd.merge(view, fetched_now, on="Date", how="outer").sort_values("Date")
            st.session_state["extra_tickers"] = sorted(set(st.session_state["extra_tickers"]) | set(new_only))
            st.session_state["ycols"] = list(dict.fromkeys(st.session_state.get("ycols", []) + new_only))
            st.success(f"추가된 티커: {', '.join(new_only)}")

# -------------------- 결측/휴일/상장전 처리 --------------------
all_days = pd.date_range(start=start, end=end, freq="D")
view = (
    view.set_index("Date")
        .reindex(all_days)
        .ffill()      # 일반 결측은 전일값
        .bfill()      # 시작 구간은 첫 유효값으로 채움(상장 전 평평)
        .rename_axis("Date")
        .reset_index()
)
num_cols = [c for c in view.columns if c != "Date"]
view[num_cols] = view[num_cols].apply(pd.to_numeric, errors="coerce")

# -------------------- 표시 자산 선택 --------------------
all_cols = [c for c in view.columns if c != "Date"]
init_default = all_cols[:min(3, len(all_cols))]
wanted = [c for c in st.session_state.get("ycols", init_default) if c in all_cols] or init_default
st.session_state["ycols"] = wanted
ycols = st.multiselect("표시할 자산", options=all_cols, key="ycols")

# -------------------- 표시 방식 --------------------
MODE_LABELS = {
    "price": "가격",
    "pct": "변화율(0% 시작)",
    "mdd": "최대 낙폭(MDD)",
}
mode = st.radio(
    "표시 방식",
    options=list(MODE_LABELS.keys()),
    index=1,  # 기본: 변화율
    horizontal=True,
    format_func=lambda k: MODE_LABELS[k],
    key="mode"
)

# 유효 컬럼만
ycols = [c for c in ycols if c in view.columns]
if not ycols:
    st.warning("선택한 자산의 데이터가 현재 범위에 없습니다. 날짜/티커를 조정해 주세요.")
    st.stop()

# -------------------- 변환/차트 데이터 --------------------
if mode == "price":
    plot_df = view[["Date"] + ycols]
    y_title = "가격/지수"
elif mode == "pct":
    plot_df = rebase_pct(view, ycols)
    y_title = "누적 수익률 (%)"
else:  # mdd
    plot_df = drawdown_pct(view, ycols)  # 0에서 아래로 내려가는 음수 값
    y_title = "MDD (최대 낙폭, %)"

plot_df[ycols] = plot_df[ycols].apply(pd.to_numeric, errors="coerce")
bad = [c for c in ycols if plot_df[c].dropna().empty]
if bad:
    st.warning(f"데이터가 없어 제외: {', '.join(bad)}")
    ycols = [c for c in ycols if c not in bad]
    if not ycols:
        st.stop()
    if mode == "price":
        plot_df = view[["Date"] + ycols]
    elif mode == "pct":
        plot_df = rebase_pct(view, ycols)
    else:
        plot_df = drawdown_pct(view, ycols)

# -------------------- 차트 --------------------
fig = px.line(plot_df, x="Date", y=ycols)
fig.update_layout(margin=dict(l=10, r=130, t=10, b=10), height=520, yaxis_title=y_title)

if mode in ("pct", "mdd"):
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
else:
    fig.update_yaxes(tickformat=",.1f")

if mode in ("price", "pct"):
    # 끝값 라벨
    last_row = plot_df.iloc[-1]
    last_x = last_row["Date"]
    for c in ycols:
        val = last_row[c]
        if pd.isna(val):
            continue
        fig.add_trace(
            go.Scatter(
                x=[last_x], y=[val],
                mode="markers+text",
                text=[format_tail_value(val, mode)],
                textposition="middle right",
                marker=dict(size=6),
                showlegend=False,
                hoverinfo="skip",
            )
        )
else:
    # MDD 모드: 각 자산의 '가장 많이 떨어진 지점'에 라벨(-32.5% 등)
    for c in ycols:
        s = plot_df[c]
        if s.dropna().empty:
            continue
        idx_min = s.idxmin()
        x_min = plot_df.loc[idx_min, "Date"]
        y_min = s.loc[idx_min]
        fig.add_trace(
            go.Scatter(
                x=[x_min], y=[y_min],
                mode="markers+text",
                text=[format_tail_value(y_min, "mdd")],
                textposition="bottom right",
                marker=dict(size=8),
                showlegend=False,
                hoverinfo="skip",
            )
        )

st.plotly_chart(fig, use_container_width=True)

# -------------------- 표/다운로드 --------------------
st.subheader("표 / 다운로드")
st.dataframe(plot_df, use_container_width=True)

csv_bytes = plot_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "CSV 다운로드 (현재 보기)",
    csv_bytes,
    file_name=f"market_{start}_{end}_{MODE_LABELS[mode]}.csv",
    mime="text/csv",
)

def to_excel_bytes(df_dict: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, d in df_dict.items():
            d.to_excel(writer, sheet_name=name, index=False)
    bio.seek(0)
    return bio.read()

xlsx = to_excel_bytes({MODE_LABELS[mode]: plot_df})
st.download_button(
    "엑셀 다운로드 (현재 보기)",
    xlsx,
    file_name=f"market_{start}_{end}_{mode}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

with st.expander("도움말"):
    st.markdown(
        """
- **기본 데이터**: `data/market_timeseries.csv` (하루 1회 `update.py`로 갱신)
- **사용자 지정 티커**: `F, TSLA, QQQ, 005930.KS` 등 입력 → **버튼 클릭** 즉시 반영
- **변화율(0% 시작)**: 시작일 기준 누적 수익률
- **MDD**: (가격/누적최고 − 1)×100. 그래프는 0에서 아래로 내려가며, 각 자산의 **최대 낙폭 지점에 값 라벨** 표시
- **휴일/상장 전**: 일 단위 재색인 + ffill + bfill(무제한)
"""
    )
