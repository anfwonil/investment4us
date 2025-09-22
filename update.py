# update.py  (v1.2, safe backfill)
# pip install --upgrade yfinance pandas
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

# -------------------- 설정 --------------------
TZ = "Asia/Seoul"
START = pd.Timestamp("2000-01-01", tz=TZ).date()
# 내일(배타)까지 요청 → 시차/마감 이슈 완화
END_EXCLUSIVE = (pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)).date()

TICKERS = ["^KS11", "^GSPC", "KRW=X"]
LABELS  = {"^KS11": "KOSPI", "^GSPC": "SP500", "KRW=X": "USDKRW"}

OUT_DIR  = Path(r"C:\Users\woori\Desktop\top10\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV  = OUT_DIR / "market_timeseries.csv"
OUT_META = OUT_DIR / "meta.csv"
OUT_LOG  = OUT_DIR / "update.log"

SAFETY_BACKFILL_DAYS = 7     # ← 최근 N일을 재수집해서 ‘늦게시 날짜’ 자동 회복
PAUSE_SEC_BETWEEN_TRIES = 5  # 재시도 대기
DOWNLOAD_TRIES = 3

# -------------------- 유틸 --------------------
def log(msg: str):
    stamp = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")

def download_prices(tickers, start_date, end_date, tries=DOWNLOAD_TRIES, pause=PAUSE_SEC_BETWEEN_TRIES):
    last_err = None
    for k in range(tries):
        try:
            df = yf.download(
                tickers,
                start=str(start_date),
                end=str(end_date),   # end는 배타
                interval="1d",       # 명시
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            log(f"다운로드 오류(시도 {k+1}/{tries}): {e}")
        time.sleep(pause)
    if last_err:
        raise last_err
    raise RuntimeError("데이터 다운로드 실패")

def choose_close(df, tickers_for_single):
    """
    멀티/단일 인덱스 → 'Adj Close' 우선, 없으면 'Close'
    단일 티커일 때는 실제 전달받은 tickers_for_single[0]으로 열 이름 설정
    """
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).unique().tolist()
        col0 = "Adj Close" if "Adj Close" in lvl0 else "Close"
        out = df[col0].copy()
    else:
        use = "Adj Close" if "Adj Close" in df.columns else "Close"
        out = df[[use]].copy()
        # 단일 티커 시, 전역 TICKERS가 아닌 실제 전달된 tickers 사용
        name = tickers_for_single[0] if isinstance(tickers_for_single, (list, tuple)) and len(tickers_for_single) == 1 else (tickers_for_single if isinstance(tickers_for_single, str) else "TICKER")
        out.columns = [name]
    out = out.sort_index()
    out.index.name = "Date"
    return out

def relabel_columns(df):
    return df.rename(columns=lambda c: LABELS.get(c, c))

def fill_small_gaps(df):
    # 공휴일/결측 이슈에 대비해 소규모 끊김 보간(앞/뒤 1칸)
    return df.ffill().bfill(limit=1)

# -------------------- 메인 --------------------
def main():
    log("=== 실행 시작 (update.py v1.2) ===")

    # 증분 시작일 계산
    if OUT_CSV.exists():
        base = pd.read_csv(OUT_CSV, parse_dates=["Date"]).sort_values("Date")
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"])
        last_ts = base["Date"].max()  # pandas.Timestamp (tz naive)
        # 최근 N일을 백필 재수집
        fetch_start = (last_ts - pd.Timedelta(days=SAFETY_BACKFILL_DAYS)).date()
        if fetch_start < START:
            fetch_start = START
        log(f"기존 CSV 발견. 마지막 날짜: {last_ts.date()} → 백필 {SAFETY_BACKFILL_DAYS}일 적용 새로 받을 시작일: {fetch_start}")
    else:
        base = None
        fetch_start = START
        log("기존 CSV 없음. 전체 구간 최초 다운로드.")

    if fetch_start >= END_EXCLUSIVE:
        log("신규 받을 구간이 없습니다. (이미 최신)")
        pd.DataFrame({"last_updated": [pd.Timestamp.now(tz=TZ).isoformat()]}).to_csv(
            OUT_META, index=False, encoding="utf-8-sig"
        )
        print("✅ Already up-to-date.")
        log("=== 실행 종료 ===")
        return

    # 다운로드
    raw = download_prices(TICKERS, fetch_start, END_EXCLUSIVE)

    # 클로즈 계열 선택
    # (주의) choose_close는 단일 티커명 보정 위해 tickers 전달 필요
    # 여기선 다중 티커이므로 이름 보정 분기는 실행되지 않지만, 안전성 유지
    close_raw = choose_close(raw, TICKERS if isinstance(TICKERS, (list, tuple)) else [TICKERS])

    # 라벨링 전 원티커 기준 최신일자 로그
    try:
        latest_by_ticker_raw = {
            t: pd.to_datetime(close_raw[t].dropna().index.max()).date() for t in close_raw.columns
        }
        log(f"원티커 기준 최신일자: {latest_by_ticker_raw}")
    except Exception as _:
        pass

    # 컬럼 라벨 교체
    close = relabel_columns(close_raw)

    # 간극 보간(소규모)
    close = fill_small_gaps(close)

    # 예상 최신일자 대비 실제 최신일자 경고(참고용)
    try:
        expected_latest = (pd.Timestamp.now(tz=TZ).normalize() - pd.Timedelta(days=1)).date()
        actual_latest = pd.to_datetime(close.index.max()).date()
        if actual_latest < expected_latest:
            log(f"[경고] 수집 최신일자가 기대보다 이전입니다. 기대: {expected_latest}, 실제: {actual_latest}")
    except Exception as _:
        pass

    # 병합(증분)
    if base is not None:
        base = base.set_index("Date")
        # base는 라벨 적용 된 컬럼이어야 함(기존 파일도 LABELS 기준으로 저장됨)
        df = pd.concat([base, close], axis=0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = close

    # 저장: CSV는 tz 없는 naive datetime으로
    out_df = df.reset_index()
    out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.tz_localize(None)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 메타 저장
    pd.DataFrame({"last_updated": [pd.Timestamp.now(tz=TZ).isoformat()]}).to_csv(
        OUT_META, index=False, encoding="utf-8-sig"
    )

    # 라벨 적용 후 컬럼 기준 최신일자 로그
    try:
        latest_by_label = {
            c: pd.to_datetime(df.index[df[c].notna()].max()).date() for c in df.columns
        }
        log(f"라벨 기준 최신일자: {latest_by_label}")
    except Exception as _:
        pass

    log(f"저장 완료: {OUT_CSV}")
    print("✅ Updated:", OUT_CSV)
    log("=== 실행 종료 ===")

# -------------------- 엔트리 --------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[에러 종료] {e}")
        raise
