"""Read-only export of daily bars from the main system's duckdb into CSV per symbol."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    DUCKDB_PATH,
    EXPORT_CSV_DIR,
    HISTORY_START,
    SQL_FILTER,
    classify_symbol,
    to_qlib_code,
)

FIELDS = ["open", "high", "low", "close", "volume", "amount"]

# bars.ts 是 TIMESTAMPTZ。裸的 CAST(ts AS DATE) 按 UTC 取日，而北京交易日零点是 UTC 的
# 前一天 16:00 —— 差 8 小时就是差一个交易日（实测：会话时区设为 UTC 时最早一根
# `CAST(min(ts) AS DATE)` 得 1990-12-18，折成 Asia/Shanghai 才回到 1990-12-19）。
# AT TIME ZONE 'Asia/Shanghai' 显式落到北京日历、得到 tz-naive TIMESTAMP，结果不再取决于
# 宿主机时区。不折也能取到数（本 venv 无 pytz，fetch_df 并不报错），但取出的日期会随
# 主机 TZ 漂移整整一天 —— 那正是本项目要用交叉校验去抓的静默错位，不能留在这里。
QUERY = f"""
    SELECT symbol, ts AT TIME ZONE 'Asia/Shanghai' AS ts, {", ".join(FIELDS)}
    FROM bars
    WHERE {SQL_FILTER}
      AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) >= '{HISTORY_START}'
"""


def connect_readonly() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@dataclass
class HygieneReport:
    rows_in: int = 0
    rows_out: int = 0
    symbols: int = 0
    dropped_by_classify: int = 0
    dup_rows: int = 0
    bad_price_rows: int = 0

    def as_text(self) -> str:
        return (
            f"rows_in            {self.rows_in:>7d}\n"
            f"rows_out           {self.rows_out:>7d}\n"
            f"symbols            {self.symbols:>7d}\n"
            f"dropped_by_classify {self.dropped_by_classify:>7d}\n"
            f"dup_rows            {self.dup_rows:>7d}\n"
            f"bad_price_rows      {self.bad_price_rows:>7d}"
        )


def load_bars(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, int]:
    """返回 (已剔除未分类行的 DataFrame, 剔除前的行数)。

    剔除前的行数必须显式带出来：classify 的丢行发生在这一层、在 HygieneReport 构造之前，
    不传出去 `dropped_by_classify` 就只能恒为 0 —— 一个说谎的计数器比没有计数器更糟。
    """
    df = con.execute(QUERY).fetch_df()
    rows_pre_classify = len(df)
    df["date"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d")
    df["asset"] = df["symbol"].map(classify_symbol)
    unmapped = int(df["asset"].isna().sum())
    if unmapped:
        print(f"[warn] {unmapped} rows dropped: symbol not classified")
    df = df.dropna(subset=["asset"])
    keep = ["symbol", "date", "asset"] + FIELDS
    return df.loc[:, keep], rows_pre_classify


def write_csvs(df: pd.DataFrame, out_dir: Path, rows_in: int) -> HygieneReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 导出窗口一变，磁盘上残留的就是上一轮的陈旧 CSV，dump_bin 会把它们当成当前数据吃进去。
    # 所以先清空 *.csv 再写，写完核对文件数。
    for stale in sorted(out_dir.glob("*.csv")):
        stale.unlink()
    rep = HygieneReport(rows_in=rows_in, dropped_by_classify=rows_in - len(df))
    # dump_bin 会静默按日期去重；重复行必须在导出前暴露出来而不是交给它处理
    dup = df.duplicated(subset=["symbol", "date"]).sum()
    if dup:
        print(f"[warn] {dup} duplicate (symbol, date) rows removed")
        df = df.drop_duplicates(subset=["symbol", "date"])
    rep.dup_rows = int(dup)

    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if bad:
        print(f"[warn] {bad} rows with non-positive price removed")
        df = df[~(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    rep.bad_price_rows = int(bad)

    df = df.sort_values(["symbol", "date"])
    rep.rows_out = len(df)
    rep.symbols = int(df["symbol"].nunique())

    written = 0
    for symbol, sub in df.groupby("symbol", sort=True):
        path = out_dir / f"{to_qlib_code(symbol)}.csv"
        sub = sub.drop(columns=["symbol", "asset"]).copy()
        # 数据本身已是前复权，factor=1.0 表示 Qlib 侧不再做二次复权
        sub["factor"] = 1.0
        sub.to_csv(path, index=False)
        written += 1
    if written != rep.symbols:
        raise AssertionError(f"expected {rep.symbols} CSVs in {out_dir}, wrote {written}")
    return rep


def main() -> int:
    con = connect_readonly()
    try:
        df, rows_in = load_bars(con)
    finally:
        con.close()
    report = write_csvs(df, EXPORT_CSV_DIR, rows_in)
    print(report.as_text())
    # 实测（CN + 1d + qfq + ts >= HISTORY_START）= 194,738 行 / 698 标的 / 529 交易日
    # 低于 90% 说明过滤条件或源库状态变了，必须硬失败
    if report.rows_out < 194_738 * 0.9:
        print("[error] exported row count far below expectation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
