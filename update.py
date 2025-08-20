# update.py  (v1.1)
# pip install yfinance pandas
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

TZ = "Asia/Seoul"
START = pd.Timestamp("2000-01-01", tz=TZ).date()
# 내일(배타)까지 요청 → 시차/마감 이슈 완화
END_EXCLUSIVE = (pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)).date()

TICKERS = ["^KS11", "^GSPC", "KRW=X"]
LABELS  = {"^KS11":"KOSPI", "^GSPC":"SP500", "KRW=X":"USDKRW"}

OUT_DIR  = Path(r"C:\Users\woori\Desktop\top10\data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV  = OUT_DIR / "market_timeseries.csv"
OUT_META = OUT_DIR / "meta.csv"
OUT_LOG  = OUT_DIR / "update.log"

def log(msg: str):
    stamp = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d %H:%M:%S")
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")

def download_prices(tickers, start_date, end_date, tries=3, pause=5):
    last_err = None
    for k in range(tries):
        try:
            df = yf.download(
                tickers,
                start=str(start_date),  # 문자열/날짜 모두 OK
                end=str(end_date),      # end는 배타
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

def choose_close(df):
    # 멀티/단일 인덱스 → Adj Close 우선, 없으면 Close
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0).unique().tolist()
        col0 = "Adj Close" if "Adj Close" in lvl0 else "Close"
        out = df[col0].copy()
    else:
        use = "Adj Close" if "Adj Close" in df.columns else "Close"
        out = df[[use]].copy()
        out.columns = [TICKERS[0]]  # 단일 티커일 때 열 이름 보정
    out = out.sort_index()
    out.index.name = "Date"
    return out

def relabel_columns(df):
    return df.rename(columns=lambda c: LABELS.get(c, c))

def fill_small_gaps(df):
    return df.ffill().bfill(limit=1)

def main():
    log("=== 실행 시작 (update.py v1.1) ===")

    # 증분 시작일 계산: tz 안 붙이고 Timestamp로 통일 → 마지막날 + 1일
    if OUT_CSV.exists():
        base = pd.read_csv(OUT_CSV, parse_dates=["Date"]).sort_values("Date")
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"])
        last_ts = base["Date"].max()                # pandas.Timestamp
        fetch_start = (last_ts + pd.Timedelta(days=1)).date()  # date
        log(f"기존 CSV 발견. 마지막 날짜: {last_ts.date()} → 새로 받을 시작일: {fetch_start}")
    else:
        base = None
        fetch_start = START
        log("기존 CSV 없음. 전체 구간 최초 다운로드.")

    if fetch_start >= END_EXCLUSIVE:
        log("신규 받을 구간이 없습니다. (이미 최신)")
        pd.DataFrame({"last_updated":[pd.Timestamp.now(tz=TZ).isoformat()]}).to_csv(
            OUT_META, index=False, encoding="utf-8-sig"
        )
        print("✅ Already up-to-date.")
        log("=== 실행 종료 ===")
        return

    raw = download_prices(TICKERS, fetch_start, END_EXCLUSIVE)
    close = choose_close(raw)
    close = relabel_columns(close)
    close = fill_small_gaps(close)

    # 병합(증분)
    if base is not None:
        base = base.set_index("Date")
        df = pd.concat([base, close], axis=0)
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        df = close

    # 저장: CSV는 tz 없는 naive datetime으로
    out_df = df.reset_index()
    out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.tz_localize(None)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 메타 저장
    pd.DataFrame({"last_updated":[pd.Timestamp.now(tz=TZ).isoformat()]}).to_csv(
        OUT_META, index=False, encoding="utf-8-sig"
    )

    log(f"저장 완료: {OUT_CSV}")
    print("✅ Updated:", OUT_CSV)
    log("=== 실행 종료 ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[에러 종료] {e}")
        raise
