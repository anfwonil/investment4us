# app.py
# pip install -U streamlit pandas plotly yfinance xlsxwriter requests beautifulsoup4 lxml

import os, re, math, time
from io import BytesIO
from datetime import date
import html as ihtml

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests

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

# -------------------- 범례용 라벨 --------------------
@st.cache_data(ttl=86400)
def pretty_label(symbol: str) -> str:
    """야후에서 짧은 이름을 가져와 사람이 읽기 쉬운 라벨 생성. 길면 10자 + '...'"""
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}
    name = info.get("shortName") or info.get("longName") or symbol
    country = info.get("country") or info.get("exchange") or ""
    qtype = (info.get("quoteType") or "").upper()
    label = " · ".join([v for v in [name, country, qtype] if v])
    return (label[:10] + "...") if len(label) > 10 else label

# -------------------- 이름→티커 보편 별칭 --------------------
COMMON_ALIASES = {
    "nikkei225": "^N225", "nikkei 225": "^N225", "nikkei": "^N225",
    "shanghai": "000001.SS", "shanghai composite": "000001.SS",
    "kospi": "^KS11", "kosdaq": "^KQ11",
    "sp500": "^GSPC", "s&p500": "^GSPC", "dow": "^DJI", "nasdaq": "^IXIC",
    "eurostoxx50": "^STOXX50E", "euro stoxx 50": "^STOXX50E",
    "ftse100": "^FTSE", "hang seng": "^HSI", "dax": "^GDAXI", "cac40": "^FCHI",
    "dollar index": "DX-Y.NYB", "us dollar index": "DX-Y.NYB",
    "usdkrw": "KRW=X", "usdcny": "CNY=X",
}

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
        soup = BeautifulSoup(html, "html.parser")  # lxml 없어도 동작
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
        tables = pd.read_html(html, attrs={"class": "snapshot-table2"})
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

def reindex_fill_ffill_bfill(df: pd.DataFrame, start, end) -> pd.DataFrame:
    all_days = pd.date_range(start=start, end=end, freq="D")
    out = (df.set_index("Date").reindex(all_days).ffill().bfill()
           .rename_axis("Date").reset_index())
    num_cols = [c for c in out.columns if c != "Date"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    return out

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

    st.title("MARKET PERFORMANCE")
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

    left, right = st.columns([1.4, 1])
    with left:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("시작일", value=default_start,
                                  min_value=date(2000, 1, 1), max_value=max_d, key="m_start")
        with c2:
            end = st.date_input("종료일", value=max_d,
                                min_value=date(2000, 1, 1), max_value=max_d, key="m_end")
    with right:
        st.session_state.setdefault("m_tickers", "")
        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            st.text_input("티커 입력",
                          key="m_tickers",
                          placeholder="예: SPY, ^KS11, 005930.KS")
        with col_btn:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            fetch_clicked = st.button("반영", type="primary", use_container_width=True, key="m_fetch")
        use_adj = st.checkbox("조정가격 사용(배당/액면 반영)", value=True, key="m_adj")

    # 기본 CSV 구간
    mask = (base["Date"].dt.date >= start) & (base["Date"].dt.date <= end)
    view = base.loc[mask].copy()

    st.session_state.setdefault("m_extra", [])
    st.session_state.setdefault("m_ycols", [])

    def expand_aliases(seq):
        return [COMMON_ALIASES.get(t.lower(), t) for t in seq]

    # 저장된 추가 티커
    saved = tuple(st.session_state["m_extra"])
    if saved:
        fetched_saved = fetch_yf_prices(saved, start, end, use_adjust=use_adj)
        if not fetched_saved.empty:
            view = pd.merge(view, fetched_saved, on="Date", how="outer").sort_values("Date")

    # 신규 추가
    if fetch_clicked:
        parsed = [t.upper() for t in re.split(r"[,\s]+", st.session_state.get("m_tickers","")) if t.strip()]
        parsed = expand_aliases(parsed)
        already = set(view.columns) | set(st.session_state["m_extra"])
        new_only = [t for t in parsed if t not in already]
        if new_only:
            with st.spinner(f"야후에서 불러오는 중... ({', '.join(new_only)})"):
                fetched_now = fetch_yf_prices(tuple(new_only), start, end, use_adjust=use_adj)
            if not fetched_now.empty:
                view = pd.merge(view, fetched_now, on="Date", how="outer").sort_values("Date")
                st.session_state["m_extra"] = sorted(set(st.session_state["m_extra"]) | set(new_only))
                st.session_state["m_ycols"] = list(
                    dict.fromkeys(st.session_state.get("m_ycols", []) + new_only)
                )
                st.success(f"추가된 티커: {', '.join(new_only)}")
        else:
            st.info("새로 추가할 티커가 없습니다.")

    # 리인덱싱
    view = reindex_fill_ffill_bfill(view, start, end)

    all_cols = [c for c in view.columns if c != "Date"]
    init_default = all_cols[:min(3, len(all_cols))]
    st.session_state["m_ycols"] = [c for c in st.session_state.get("m_ycols", init_default) if c in all_cols] or init_default
    ycols = st.multiselect("표시할 자산", options=all_cols, key="m_ycols")
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
    base_map = {c: pretty_label(c) for c in ycols}
    cnt = Counter(base_map.values())
    unique_map, used = {}, set()
    for c in ycols:
        base_lbl = base_map[c]
        lbl = base_lbl if cnt[base_lbl] == 1 else f"{base_lbl} ({c})"
        k = 2
        while lbl in used:
            lbl = f"{base_lbl} ({c})[{k}]"; k += 1
        unique_map[c] = lbl; used.add(lbl)

    # ===== 데이터 가공 & 그래프 =====
    if mode == "price":
        plot_df_use = plot_df.copy()
        y_title = "가격지수"
        plot_df_disp = plot_df_use.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]
        fig = px.line(plot_df_disp, x="Date", y=ycols_disp, render_mode="svg")
        fig.update_yaxes(tickformat=",.1f")

    elif mode == "pct":
        plot_df_use = rebase_pct(plot_df, ycols)  # % 단위
        y_title = "누적 수익률 (%)"
        plot_df_disp = plot_df_use.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]
        fig = px.line(plot_df_disp, x="Date", y=ycols_disp, render_mode="svg")
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")

    elif mode == "pct_log":
        # (1) 일반 변화율(%) → (2) 배수(= 1 + %/100)로 변환하여 로그축에 그리되,
        #     눈금은 %처럼 보이도록 커스텀 라벨을 사용
        pct = rebase_pct(view, ycols).copy()
        for c in ycols:
            pct[c] = (pd.to_numeric(pct[c], errors="coerce") / 100.0) + 1.0  # 배수

        plot_df_disp = pct.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]

        # 양수(로그 가능) 시리즈만 표시
        vals = plot_df_disp[ycols_disp].apply(pd.to_numeric, errors="coerce")
        y_show = [c for c in ycols_disp if (vals[c] > 0).any()]
        if not y_show:
            st.info("로그축에 표시할 수 있는 시리즈가 없습니다. 일반 변화율(%)로 확인해 주세요.")
            return

        # 로그축 눈금: 배수값 → 라벨은 (배수-1)*100%
        y_min = float(vals[y_show].min().min())
        y_max = float(vals[y_show].max().max())
        tick_candidates = [0.25, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100]
        tickvals = [v for v in tick_candidates if v > 0 and y_min * 0.95 <= v <= y_max * 1.05] or [1]
        if 1 not in tickvals:
            tickvals = sorted(set(tickvals + [1]))
        ticktext = [f"{(v - 1) * 100:.0f}%" for v in tickvals]

        y_title = "누적 수익률 (%, 로그 간격)"

        fig = px.line(plot_df_disp, x="Date", y=y_show, render_mode="svg")
        fig.update_layout(
            margin=dict(l=10, r=130, t=10, b=10),
            height=520,
            yaxis_title=y_title,
            legend=dict(groupclick="togglegroup"),
            uirevision="mkt",
            xaxis_rangeslider_visible=False,
        )
        fig.update_yaxes(type="log", tickvals=tickvals, ticktext=ticktext)

        # 마지막값 라벨(%) 표시
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
        return  # ⬅️ 로그 모드에서는 여기서 종료



    else:  # mdd
        plot_df_use = drawdown_pct(plot_df, ycols)  
        y_title = "MDD (%, 낮을수록 심함)"
        plot_df_disp = plot_df_use.rename(columns=unique_map)
        ycols_disp = [unique_map.get(c, c) for c in ycols]
        fig = px.line(plot_df_disp, x="Date", y=ycols_disp, render_mode="svg")
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
                                         text=[format_tail_value(y_max, mode)], textposition="top right",
                                         marker=dict(size=8), showlegend=False, hoverinfo="skip", legendgroup=c))
                if idx_min != idx_max:
                    fig.add_trace(go.Scatter(x=[x_min], y=[y_min], mode="markers+text",
                                             text=[format_tail_value(y_min, mode)], textposition="bottom right",
                                             marker=dict(size=8), showlegend=False, hoverinfo="skip", legendgroup=c))
                is_last_extreme = (x_max == lx) or (x_min == lx)
                if not is_last_extreme:
                    v_last = sc.iloc[-1][c]
                    fig.add_trace(go.Scatter(x=[lx], y=[v_last], mode="markers+text",
                                             text=[format_tail_value(v_last, mode)], textposition="middle right",
                                             marker=dict(size=6), showlegend=False, hoverinfo="skip", legendgroup=c))

    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": False, "doubleClick": "reset", "displaylogo": False})

    # ---- (2) 기간별 수익률 스냅샷 ----
    st.markdown("<h5>(2) Periodic Return</h5>", unsafe_allow_html=True)
    price_df = view[["Date"] + ycols].copy()
    for c in ycols:
        price_df[c] = pd.to_numeric(price_df[c], errors="coerce")

    windows = [("1W",5), ("1M",21), ("3M",63), ("6M",126), ("12M",252), ("36M", 756)]
    rows = []
    for c in ycols:
        s = price_df[c].dropna()
        row = {"자산": pretty_label(c)}
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
        column_config={
            "자산": st.column_config.TextColumn("Asset", width="large"),
            "R_1W": st.column_config.TextColumn("1W"),
            "R_1M": st.column_config.TextColumn("1M"),
            "R_3M": st.column_config.TextColumn("3M"),
            "R_6M": st.column_config.TextColumn("6M"),
            "R_12M": st.column_config.TextColumn("12M"),
            "R_36M": st.column_config.TextColumn("36M"),
        }
    )

    # ---- (3) 단일 자산 이동평균 ----
    one = st.selectbox("자산 선택", options=ycols, index=0, key="m_ma_one")
    one_label = pretty_label(one)

    one_df = view[["Date", one]].copy()
    one_df[one] = pd.to_numeric(one_df[one], errors="coerce")
    for w in (20, 60, 120):
        one_df[f"SMA{w}"] = one_df[one].rolling(w).mean()

    st.markdown("<h5>(3) Chart with Moving Averages</h5>", unsafe_allow_html=True)
    disp_ma = one_df.rename(columns={one: one_label})
    fig_ma = px.line(disp_ma, x="Date", y=one_label,
                     title=f"{one_label} — Close & SMA(20/60/120)", render_mode="svg")
    for w in (20, 60, 120):
        col = f"SMA{w}"
        if col in disp_ma:
            fig_ma.add_trace(go.Scatter(
                x=disp_ma["Date"], y=disp_ma[col],
                mode="lines", name=f"SMA{w}",
                line=dict(dash="dot")))
    fig_ma.update_layout(height=600, margin=dict(l=10, r=130, t=30, b=10),
                         uirevision="mkt_ma", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_ma, use_container_width=True)

    # ---- (4) Company Info (Finviz) ----
    st.markdown("<h5>(4) Company Info</h5>", unsafe_allow_html=True)
    default_fv = one if re.match(r"^[A-Za-z]{1}[A-Za-z0-9\.\-]{0,9}$", str(one)) else "TSLA"
    c_fv1, c_fv2 = st.columns([0.22, 0.78])
    with c_fv1:
        finviz_ticker = st.text_input("Ticker", value=default_fv, key="m_finviz_ticker")
    with c_fv2:
        st.caption("예: AAPL, TSLA, MSFT (지수/통화 티커는 기업데이터가 없을 수 있습니다)")

    if (finviz_ticker or "").strip():
        with st.spinner("회사 정보 수집 중..."):
            _out = fetch_finviz_company(finviz_ticker)
            if isinstance(_out, tuple) and len(_out) == 3:
                profile_text, fv_table, _ = _out
            else:
                profile_text, fv_table = _out

        st.markdown(profile_text)
        if not fv_table.empty:
            st.dataframe(fv_table, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("표시할 Key Metrics가 없습니다.")

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

# ==================== TAB 2: Portfolio (원본 유지) ====================
def guess_currency(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".KS") or t.endswith(".KQ"): return "KRW"
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
    st.title("PORTFOLIO ANALYSIS")

    c1, c2, c3, c4 = st.columns([1.2, 1.1, 1, 1])
    with c1:
        start = st.date_input("시작일", value=date(2020,1,1),
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
        bench = st.text_input("벤치마크(옵션, 예: SPY, QQQ, ^GSPC)", value="SPY", key="p_bench")

    n1, n2, n3 = st.columns(3)
    with n1: name1 = st.text_input("포트폴리오 1 이름", value="포트폴리오 1안", key="p_name1")
    with n2: name2 = st.text_input("포트폴리오 2 이름", value="포트폴리오 2안", key="p_name2")
    with n3: name3 = st.text_input("포트폴리오 3 이름", value="포트폴리오 3안", key="p_name3")

    lite = st.checkbox("경량 모드(주간 리샘플)", value=False, help="브라우저가 느리면 켜 보세요.", key="p_lite")

    default_df = pd.DataFrame([
        {"티커":"SPY", "P1(%)":40.0, "P2(%)":33.0, "P3(%)":40.0},
        {"티커":"QQQ", "P1(%)":40.0, "P2(%)":33.0, "P3(%)":40.0},
        {"티커":"TLT", "P1(%)":20.0, "P2(%)":34.0, "P3(%)":20.0},
    ])
    st.session_state.setdefault("weights_df", default_df.copy())

    with st.form("weights_form", clear_on_submit=False):
        h1, h2 = st.columns([1.0, 0.14])
        with h1:
            st.markdown("#### 자산 구성 (가로 입력: 1/2/3안)")
        with h2:
            apply_weights = st.form_submit_button("반영", use_container_width=True)

        edited = st.data_editor(
            st.session_state["weights_df"],
            num_rows="dynamic",
            use_container_width=True,
            key="p_table",
            column_config={
                "티커": st.column_config.TextColumn("티커"),
                "P1(%)": st.column_config.NumberColumn(f"{name1}(%)", step=1.0, format="%.2f"),
                "P2(%)": st.column_config.NumberColumn(f"{name2}(%)", step=1.0, format="%.2f"),
                "P3(%)": st.column_config.NumberColumn(f"{name3}(%)", step=1.0, format="%.2f"),
            },
        )

    if "apply_weights" in locals() and apply_weights:
        st.session_state["weights_df"] = edited.copy()
        st.success("가중치를 반영했습니다.")

    edit_df = st.session_state["weights_df"].copy()

    up = st.file_uploader("CSV 업로드(컬럼: 티커, P1(%), P2(%), P3(%))", type=["csv"], key="p_upload")
    if up:
        try:
            csvdf = pd.read_csv(up)
            need = {"티커","P1(%)","P2(%)","P3(%)"}
            if need.issubset(csvdf.columns):
                st.session_state["weights_df"] = csvdf[list(need)].copy()
                edit_df = st.session_state["weights_df"]; st.info("업로드된 CSV를 사용합니다.")
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
        raw_px = fetch_yf_prices(tickers, start, end, use_adjust=True)
    if raw_px.empty: st.warning("가격 데이터를 가져오지 못했습니다."); st.stop()

    starts = {}
    for col in [c for c in raw_px.columns if c != "Date"]:
        s = raw_px[["Date", col]].dropna()
        starts[col] = s["Date"].min().date() if not s.empty else None
    with st.expander("각 티커 데이터 시작일(상장일 유사)"):
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
            is_kr = c.endswith(".KS") or c.endswith(".KQ")
            if base_ccy == "KRW" and not is_kr:
                tmp[c] = tmp[c] * usdkrw
            elif base_ccy == "USD" and is_kr:
                tmp[c] = tmp[c] / usdkrw
        px_df = tmp.reset_index()

    prices = px_df.set_index("Date")

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

    df_plot = pd.DataFrame(index=idx).sort_index()
    for nm, s in portfolios:
        if not s.empty:
            df_plot[nm] = (s / s.iloc[0] - 1.0) * 100.0
    if bench_line is not None:
        df_plot[bench_line.name] = (bench_line / bench_line.dropna().iloc[0] - 1.0) * 100.0
    df_plot = df_plot.reset_index().rename(columns={"index":"Date"})

    fig = px.line(df_plot, x="Date", y=[c for c in df_plot.columns if c != "Date"], render_mode="svg")
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
            comp[f"{nm} MDD(%)"] = (s / s.cummax() - 1.0) * 100.0
    if bench_line is not None:
        bench_equity = (1.0 + (bench_line - bench_line.dropna().iloc[0]) / bench_line.dropna().iloc[0]).reindex(comp.index).ffill()
        bench_equity.iloc[0] = 1.0
        comp[f"{bench_name} MDD(%)"] = (bench_equity / bench_equity.cummax() - 1.0) * 100.0

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

# ==================== TAB 3: Analysis (트렌드/모멘텀 제거) ====================
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
    cols_out = ["Date","Ticker","Close","High","Low","Volume"]
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
    )
    if raw.empty:
        return pd.DataFrame(columns=cols_out)

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = list(raw.columns.levels[0])
        def pick(k):
            return raw[k].copy() if k in lvl0 else pd.DataFrame(index=raw.index, columns=raw.columns.levels[1])
        close = pick("Close"); high = pick("High"); low = pick("Low"); vol = pick("Volume")
        close.index.name = high.index.name = low.index.name = vol.index.name = "Date"

        dfC = close.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Close")
        dfH = high.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="High")
        dfL = low.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Low")
        dfV = vol.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Volume")
        out = dfC.merge(dfH, on=["Date","Ticker"], how="left")\
                 .merge(dfL, on=["Date","Ticker"], how="left")\
                 .merge(dfV, on=["Date","Ticker"], how="left")
    else:
        cols = raw.columns.tolist()
        def getc(k): return raw[k] if k in cols else pd.Series(index=raw.index, dtype=float)
        tkr = list(tickers)[0]
        out = pd.DataFrame({
            "Date": raw.index,
            "Ticker": tkr,
            "Close": _to_number(getc("Close")),
            "High":  _to_number(getc("High")),
            "Low":   _to_number(getc("Low")),
            "Volume":_to_number(getc("Volume")),
        })

    out["Date"] = _coerce_date(out["Date"])
    out = out.sort_values(["Ticker","Date"]).reset_index(drop=True)
    return out

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

def tab_research_global():
    st.title("가격 리서치 (글로벌·펀드 포함)")

    c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
    with c1:
        raw = st.text_input("야후 티커(쉼표/공백 구분) — 예: 005930.KS, 069500.KS, SPY, ^N225, ^KS11",
                            value="069500.KS, SPY, ^KS11")
    with c2:
        start = st.date_input("시작일", value=date(2021,1,1), min_value=date(2000,1,1), max_value=date.today())
    with c3:
        end = st.date_input("종료일", value=date.today(), min_value=date(2000,1,1), max_value=date.today())

    use_adj = st.checkbox("조정가격(배당/분할 반영) 사용", value=True)
    parsed = [t for t in re.split(r"[,\s]+", raw.upper()) if t.strip()]

    st.markdown("**펀드 NAV 파일 업로드(CSV/XLSX)** — 컬럼 예시: `일자, 기준가` 또는 `Date, Close`")
    fund_files = st.file_uploader("여러 개 업로드 가능", type=["csv","xls","xlsx"], accept_multiple_files=True)

    with st.spinner("데이터 불러오는 중..."):
        yf_long = fetch_yf_ohlcv(tuple(parsed), start, end, use_adjust=use_adj) if parsed else \
                  pd.DataFrame(columns=["Date","Ticker","Close","High","Low","Volume"])
        fd_long = pd.DataFrame(columns=["Date","Ticker","Close"])

    frames = [df for df in [yf_long, fd_long] if df is not None and not df.empty]
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

# 탭 생성
tab1, tab2, tab3 = st.tabs(["Market", "Portfolio", "Analysis"])
with tab1:  tab_market()
with tab2:  tab_portfolio()
with tab3:  tab_research_global()

# 깃허브 사이트
# https://github.com/anfwonil/investment4us

# 배포
# cd C:\Users\woori\Desktop\top10
# git add -A
# git commit -m "chore: update data and backup folders"
# git push
#   

# git add requirements.txt
# git commit -m "chore: update requirements.txt (add lxml)"


# 실행 참고:
# cd C:\Users\woori\Desktop\top10
# & "C:\Users\woori\anaconda3\python.exe" -m streamlit run ".\app.py"
