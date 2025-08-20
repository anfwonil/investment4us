# app.py
# pip install -U streamlit pandas plotly yfinance xlsxwriter requests

import os, re, math
from io import BytesIO
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests

st.set_page_config(page_title="글로벌 시장 대시보드", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "data", "market_timeseries.csv")
META_CSV = os.path.join(BASE_DIR, "data", "meta.csv")

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

# -------------------- (NEW) 범례 예쁜 라벨 유틸 --------------------
@st.cache_data(ttl=86400)
def pretty_label(symbol: str) -> str:
    """
    yfinance에서 이름/국가/종류를 가져와 범례용 라벨 생성.
    너무 길면 15자 + '...'
    """
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}
    name = info.get("shortName") or info.get("longName") or symbol
    country = info.get("country") or info.get("exchange") or ""
    qtype = (info.get("quoteType") or "").upper()
    label = " · ".join([v for v in [name, country, qtype] if v])
    return (label[:15] + "...") if len(label) > 15 else label

# -------------------- (NEW) 이름→티커 검색 유틸 --------------------
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

def yahoo_search(query: str, quotes_count: int = 10):
    """야후 파이낸스 검색(비공식). 결과: [{symbol, shortname, longname, exchDisp, quoteType}, ...]"""
    q = query.strip()
    results = []
    alias_key = q.lower()
    if alias_key in COMMON_ALIASES:
        results.append({
            "symbol": COMMON_ALIASES[alias_key],
            "shortname": f"Alias for '{q}'",
            "longname": None, "exchDisp": "—", "quoteType": "ALIAS",
        })
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

def reindex_fill_ffill_only(df: pd.DataFrame, start, end) -> pd.DataFrame:
    all_days = pd.date_range(start=start, end=end, freq="D")
    out = (df.set_index("Date").reindex(all_days).ffill()
           .rename_axis("Date").reset_index())
    num_cols = [c for c in out.columns if c != "Date"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    return out

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

# ==================== TAB 1: 시장/티커(+MDD) ====================
def tab_market():
    base = load_base_data(DATA_CSV)
    last_updated = load_meta(META_CSV)
    st.title("글로벌 시장 대시보드")
    if last_updated:
        st.caption(f"마지막 업데이트(KST): {last_updated} · 기본 데이터는 하루 1회 update.py로 갱신")

    min_d, max_d = base["Date"].min().date(), base["Date"].max().date()
    default_start = max(date(2021, 1, 1), min_d)

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
        st.markdown("**사용자 지정 티커 추가** (쉼표/스페이스 구분)")
        if "m_tickers" not in st.session_state:
            st.session_state["m_tickers"] = ""
        tickers_text = st.text_input("예: F, TSLA, QQQ, 005930.KS",
                                     value=st.session_state["m_tickers"], key="m_tickers")
        use_adj = st.checkbox("조정가격 사용(배당/액면 반영)", value=True, key="m_adj")
        fetch_clicked = st.button("추가 티커 불러오기", type="primary", key="m_fetch")

        st.divider()
        st.markdown("**이름으로 티커 검색** (예: nikkei225, euro stoxx 50, dollar index)")
        q = st.text_input("티커/지수/통화 이름", value="", key="m_search_query")
        if st.button("야후에서 검색", key="m_search_btn"):
            if q.strip():
                with st.spinner("야후에서 검색 중..."):
                    results = yahoo_search(q, quotes_count=10)

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

    # 기본 CSV 구간
    mask = (base["Date"].dt.date >= start) & (base["Date"].dt.date <= end)
    view = base.loc[mask].copy()

    if "m_extra" not in st.session_state:
        st.session_state["m_extra"] = []
    if "m_ycols" not in st.session_state:
        st.session_state["m_ycols"] = []

    def expand_aliases(seq):
        out = []
        for t in seq:
            out.append(COMMON_ALIASES.get(t.lower(), t))
        return out

    saved = tuple(st.session_state["m_extra"])
    if saved:
        fetched_saved = fetch_yf_prices(saved, start, end, use_adjust=use_adj)
        if not fetched_saved.empty:
            view = pd.merge(view, fetched_saved, on="Date", how="outer").sort_values("Date")

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
                st.session_state["m_ycols"] = list(dict.fromkeys(st.session_state.get("m_ycols", []) + new_only))
                st.success(f"추가된 티커: {', '.join(new_only)}")
        else:
            st.info("새로 추가할 티커가 없습니다.")

    view = reindex_fill_ffill_bfill(view, start, end)

    all_cols = [c for c in view.columns if c != "Date"]
    init_default = all_cols[:min(3, len(all_cols))]
    st.session_state["m_ycols"] = [c for c in st.session_state.get("m_ycols", init_default) if c in all_cols] or init_default
    ycols = st.multiselect("표시할 자산", options=all_cols, key="m_ycols")
    if not ycols:
        st.info("표시할 자산을 선택하세요."); return

    MODE_LABELS = {"price": "가격", "pct": "변화율(0% 시작)", "mdd": "최대 낙폭(MDD)"}
    mode = st.radio("표시 방식", options=list(MODE_LABELS.keys()), index=1,
                    horizontal=True, format_func=lambda k: MODE_LABELS[k], key="m_mode")

    # ----- 데이터 가공(계산은 원본 티커 기준) -----
    if mode == "price":
        plot_df = view[["Date"] + ycols]; y_title = "가격지수"
    elif mode == "pct":
        plot_df = rebase_pct(view, ycols); y_title = "누적 수익률 (%)"
    else:
        plot_df = drawdown_pct(view, ycols); y_title = "MDD (%, 낮을수록 심함)"

    # ----- (NEW) 그래프용 컬럼 라벨을 예쁜 이름으로 치환 -----
    name_map = {c: pretty_label(c) for c in ycols}
    plot_df_disp = plot_df.rename(columns=name_map)
    ycols_disp = [name_map[c] for c in ycols]

    # ---- 라인 차트 ----
    fig = px.line(plot_df_disp, x="Date", y=ycols_disp)
    fig.update_layout(
        margin=dict(l=10, r=130, t=10, b=10),
        height=520,
        yaxis_title=y_title,
        legend=dict(groupclick="togglegroup"),
    )
    if mode in ("pct", "mdd"):
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    else:
        fig.update_yaxes(tickformat=",.1f")

    for tr in fig.data:
        tr.legendgroup = tr.name

    if mode == "mdd":
        for c in ycols_disp:
            s = plot_df_disp[c]
            if s.dropna().empty:
                continue
            idx_min = s.idxmin()
            fig.add_trace(
                go.Scatter(
                    x=[plot_df_disp.loc[idx_min, "Date"]],
                    y=[s.loc[idx_min]],
                    mode="markers+text",
                    text=[format_tail_value(s.loc[idx_min], "mdd")],
                    textposition="bottom right",
                    marker=dict(size=8),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=c,
                )
            )
    else:
        last_row = plot_df_disp.iloc[-1]
        lx = last_row["Date"]
        for c in ycols_disp:
            sc = plot_df_disp[["Date", c]].dropna()
            if sc.empty:
                continue
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
                v_last = last_row[c]
                if not pd.isna(v_last):
                    fig.add_trace(go.Scatter(x=[lx], y=[v_last], mode="markers+text",
                                             text=[format_tail_value(v_last, mode)], textposition="middle right",
                                             marker=dict(size=6), showlegend=False, hoverinfo="skip", legendgroup=c))
    st.plotly_chart(fig, use_container_width=True)

    # 다운로드(원본 티커 기준)
    st.markdown("#### 데이터 다운로드")
    dl_df = plot_df[["Date"] + ycols].copy()
    st.download_button("CSV 다운로드", dl_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"market_{mode}_{start}_{end}.csv", mime="text/csv")
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        dl_df.to_excel(w, sheet_name="market", index=False)
    bio.seek(0)
    st.download_button("엑셀 다운로드", bio.read(),
                       file_name=f"market_{mode}_{start}_{end}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------- 포트폴리오 유틸 --------------------
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
    rets = prices.pct_change()

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
        mask_today = prices.iloc[i].notna()
        r = rets.iloc[i].fillna(0.0)[tickers].values
        port_ret = float((w_curr * r).sum())
        V *= (1.0 + port_ret); equity.append(V)
        if (1.0 + port_ret) != 0: w_curr = w_curr * (1.0 + r) / (1.0 + port_ret)
        if rmask.iloc[i]:
            if mask_today.any():
                w_tgt = (W[mask_today] / W[mask_today].sum()).reindex(W.index, fill_value=0.0).values
            else:
                w_tgt = W.values
            V = apply_cost(V, w_curr, w_tgt); w_curr = w_tgt.copy()

    eq = pd.Series(equity, index=dates, name="Portfolio"); eq.iloc[0] = 1.0
    return eq

def portfolio_metrics(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    if ret.empty: return {}
    ann = 252
    cagr = (equity.iloc[-1]) ** (ann / len(ret)) - 1.0
    vol = ret.std() * math.sqrt(ann)
    sharpe = cagr / vol if vol > 0 else float("nan")
    dd = (equity / equity.cummax() - 1.0); mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MDD": mdd, "Calmar": calmar}

# ==================== TAB 2: 포트폴리오 ====================
def tab_portfolio():
    st.title("포트폴리오 분석")

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

    # 경량 모드(주간 리샘플)
    lite = st.checkbox("경량 모드(주간 리샘플)", value=False, help="브라우저가 느리면 켜 보세요.", key="p_lite")

    default_df = pd.DataFrame([
        {"티커":"SPY", "P1(%)":40.0, "P2(%)":33.0, "P3(%)":40.0},
        {"티커":"QQQ", "P1(%)":40.0, "P2(%)":33.0, "P3(%)":40.0},
        {"티커":"TLT", "P1(%)":20.0, "P2(%)":34.0, "P3(%)":20.0},
    ])
    if "weights_df" not in st.session_state:
        st.session_state["weights_df"] = default_df.copy()

    st.markdown("#### 자산 구성 (가로 입력: 1/2/3안)")
    edit_df = st.data_editor(
        st.session_state["weights_df"], num_rows="dynamic", use_container_width=True, key="p_table",
        column_config={
            "티커": st.column_config.TextColumn("티커"),
            "P1(%)": st.column_config.NumberColumn(f"{name1}(%)", step=1.0),
            "P2(%)": st.column_config.NumberColumn(f"{name2}(%)", step=1.0),
            "P3(%)": st.column_config.NumberColumn(f"{name3}(%)", step=1.0),
        }
    )
    st.session_state["weights_df"] = edit_df.copy()

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

    # 각 티커 시작일
    starts = {}
    for col in [c for c in raw_px.columns if c != "Date"]:
        s = raw_px[["Date", col]].dropna()
        starts[col] = s["Date"].min().date() if not s.empty else None
    with st.expander("각 티커 데이터 시작일(상장일 유사)"):
        info_df = pd.DataFrame({"티커": list(starts.keys()),
                                "데이터 시작일": [str(starts[k]) if starts[k] else "-" for k in starts]})
        st.table(info_df)

    px_df = reindex_fill_ffill_only(raw_px, start, end)
    if lite:
        tmp = px_df.set_index("Date").resample("W-FRI").last().reset_index()
        px_df = tmp

    usdkrw = None
    if base_ccy in ("USD","KRW"):
        fx = fetch_yf_prices(("KRW=X",), start, end, use_adjust=False)
        if not fx.empty:
            fx_df = reindex_fill_ffill_only(fx.rename(columns={"KRW=X":"USDKRW"}), start, end)
            if lite: fx_df = fx_df.set_index("Date").resample("W-FRI").last().reset_index()
            usdkrw = fx_df.set_index("Date")["USDKRW"]

    if base_ccy == "KRW" and usdkrw is not None:
        tmp = px_df.set_index("Date")
        for c in [c for c in tmp.columns if c != "Date"]:
            if guess_currency(c) == "USD": tmp[c] = tmp[c] * usdkrw
        px_df = tmp.reset_index()
    elif base_ccy == "USD" and usdkrw is not None:
        tmp = px_df.set_index("Date")
        for c in [c for c in tmp.columns if c != "Date"]:
            if guess_currency(c) == "KRW": tmp[c] = tmp[c] / usdkrw
        px_df = tmp.reset_index()

    prices = px_df.set_index("Date")

    mode = "BH" if rb_mode.startswith("없음") else "RB"
    freq = "M" if rb_mode.startswith("매월") else ("Q" if rb_mode.startswith("분기") else "A")

    portfolios = []
    for nm, w in [(name1,w1),(name2,w2),(name3,w3)]:
        if not w:
            portfolios.append((nm, pd.Series(dtype=float))); continue
        W = pd.Series(w, dtype=float); W = W / (W.sum() if W.sum()!=0 else 1)
        eq = build_portfolio_equity_missing_aware(prices, W.to_dict(), mode=mode, fee_bps=fee_bps, reb_freq=freq)
        eq.name = nm; portfolios.append((nm, eq))

    bench_line = None
    bench_name = bench.strip().upper() if bench.strip() else None
    if bench_name:
        bpx = fetch_yf_prices((bench_name,), start, end, use_adjust=True)
        if not bpx.empty:
            bpx = reindex_fill_ffill_only(bpx, start, end)
            if lite: bpx = bpx.set_index("Date").resample("W-FRI").last().reset_index()
            bser = bpx.set_index("Date")[bench_name]
            if usdkrw is not None:
                cur = guess_currency(bench_name)
                if base_ccy == "KRW" and cur == "USD": bser = bser * usdkrw
                elif base_ccy == "USD" and cur == "KRW": bser = bser / usdkrw
            bench_line = (bser / bser.dropna().iloc[0]).rename(bench_name)

    # 누적수익률(%) 차트
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

    fig = px.line(df_plot, x="Date", y=[c for c in df_plot.columns if c != "Date"])
    fig.update_layout(margin=dict(l=10, r=110, t=10, b=10), height=480,
                      yaxis_title=f"누적 수익률 (%) — 기준통화: {base_ccy}")
    fig.update_yaxes(ticksuffix="%")
    for tr in fig.data: tr.legendgroup = tr.name

    # 마지막값 & 최저점 라벨
    last = df_plot.dropna().iloc[-1]; lx = last["Date"]
    for c in df_plot.columns[1:]:
        sc = df_plot[["Date", c]].dropna()
        if sc.empty: 
            continue
        v_last = sc.iloc[-1][c]
        fig.add_trace(go.Scatter(
            x=[lx], y=[v_last], mode="markers+text",
            text=[f"{v_last:+.1f}%"], textposition="middle right",
            marker=dict(size=6), showlegend=False, hoverinfo="skip",
            legendgroup=c
        ))
        imin = sc[c].idxmin()
        x_min, y_min = sc.loc[imin, "Date"], sc.loc[imin, c]
        fig.add_trace(go.Scatter(
            x=[x_min], y=[y_min], mode="markers+text",
            text=[f"{y_min:+.1f}%"], textposition="bottom right",
            marker=dict(size=8), showlegend=False, hoverinfo="skip",
            legendgroup=c
        ))
    st.plotly_chart(fig, use_container_width=True)

    # MDD(%) 비교 — 벤치마크 포함
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
        fig2 = px.line(comp, x="Date", y=mdd_cols)
        fig2.update_layout(margin=dict(l=10, r=110, t=10, b=10), height=300, yaxis_title="MDD (%)")
        fig2.update_yaxes(ticksuffix="%", rangemode="tozero")
        for tr in fig2.data: tr.legendgroup = tr.name
        st.plotly_chart(fig2, use_container_width=True)

    # 요약 지표 표 — 벤치마크 포함
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

    # 다운로드만 제공 (큰 표는 제거)
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

    st.download_button("CSV 다운로드", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"portfolio_compare_{base_ccy}_{start}_{end}.csv", mime="text/csv")
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        out.to_excel(w, sheet_name="portfolio", index=False)
    bio.seek(0)
    st.download_button("엑셀 다운로드", bio.read(),
                       file_name=f"portfolio_compare_{base_ccy}_{start}_{end}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==================== MAIN ====================
tab1, tab2 = st.tabs(["시장/티커", "포트폴리오"])
with tab1:
    tab_market()
with tab2:
    tab_portfolio()

# cd C:\Users\woori\Desktop\top10
# & "C:\Users\woori\anaconda3\python.exe" -m streamlit run ".\app.py"
