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
    SQL_FILTER,
    classify_symbol,
    to_qlib_code,
)

FIELDS = ["open", "high", "low", "close", "volume", "amount"]

# bars.ts 是 TIMESTAMP WITH TIME ZONE，存的即北京墙钟的交易日零点。
# 原样取出 timestamptz 会让 duckdb 为构造 tz-aware 列去 import pytz（本 venv 没有，
# 且不许装），所以先 AT TIME ZONE 'Asia/Shanghai' 显式折成北京时间、得到 tz-naive
# TIMESTAMP。写死时区而非依赖主机 TZ 设置：换机器也不会把交易日整体错开一天。
QUERY = f"""
    SELECT symbol, ts AT TIME ZONE 'Asia/Shanghai' AS ts, {", ".join(FIELDS)}
    FROM bars
    WHERE {SQL_FILTER}
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


def load_bars(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(QUERY).fetch_df()
    df["date"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d")
    df["asset"] = df["symbol"].map(classify_symbol)
    unmapped = int(df["asset"].isna().sum())
    if unmapped:
        print(f"[warn] {unmapped} rows dropped: symbol not classified")
    df = df.dropna(subset=["asset"])
    keep = ["symbol", "date", "asset"] + FIELDS
    return df.loc[:, keep]


def write_csvs(df: pd.DataFrame, out_dir: Path) -> HygieneReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = HygieneReport(rows_in=len(df))
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
    rep.symbols = df["symbol"].nunique()

    for symbol, sub in df.groupby("symbol", sort=True):
        path = out_dir / f"{to_qlib_code(symbol)}.csv"
        sub = sub.drop(columns=["symbol", "asset"]).copy()
        # 数据本身已是前复权，factor=1.0 表示 Qlib 侧不再做二次复权
        sub["factor"] = 1.0
        sub.to_csv(path, index=False)
    return rep


def main() -> int:
    con = connect_readonly()
    try:
        df = load_bars(con)
    finally:
        con.close()
    report = write_csvs(df, EXPORT_CSV_DIR)
    print(report.as_text())
    # 实测 CN + 1d + qfq 共 314,676 行（加 market 过滤前是 360,089）
    # 低于 90% 说明过滤条件或源库状态变了，必须硬失败
    if report.rows_out < 314_676 * 0.9:
        print("[error] exported row count far below expectation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
