"""只读约束 + 导出体检。前两条保护的是主系统正在写入的真钱数据库，优先级高于功能测试。"""

import duckdb
import pandas as pd
import pytest

from config.settings import DUCKDB_PATH, HISTORY_START
from tools.export_bars import FIELDS, HygieneReport, connect_readonly, load_bars, write_csvs


def test_writing_to_the_source_database_is_rejected():
    """必须走我们自己的 connect_readonly()。

    这条测试的名字就是它的目的：一旦有人把 export_bars.py 里的 read_only=True 删掉，
    这里必须红。自己另开一个 duckdb.connect(..., read_only=True) 是测不到那一点的。
    """
    con = connect_readonly()
    try:
        # 同一个连接要能正常读，否则"写不进去"可能只是因为连错了地方
        assert con.execute("SELECT count(*) FROM bars").fetchone()[0] > 0
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE _qoder_probe (x INTEGER)")
    finally:
        con.close()


def test_read_only_flag_alone_rejects_a_write():
    """独立于我方代码再断言一次：read_only=True 这个连接属性本身就拒绝写。"""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE _qoder_probe (x INTEGER)")
    finally:
        con.close()


def test_export_is_reproducible_and_market_filtered():
    con = connect_readonly()
    try:
        df, rows_in = load_bars(con)
    finally:
        con.close()
    # 实测：CN + 1d + qfq + ts>=HISTORY_START 共 698 个标的
    assert len(set(df["symbol"])) == 698
    assert df["asset"].isin(["stock", "etf", "index"]).all()
    # (?:...) 非捕获组：断言不变，但避免 pandas 对「有捕获组却只判布尔」的 UserWarning
    assert df["symbol"].str.contains(r"\.(?:SH|SZ|BJ)$").all()
    assert not df["symbol"].str.endswith((".US", ".HK", ".KR", ".GLOBAL")).any()
    # load_bars 必须把 classify 剔除前的行数显式带出来
    assert isinstance(rows_in, int) and rows_in >= len(df)


def _fake_bars(symbols):
    """内存库里的 bars，列与真库一致 —— 用来测真库永远不触发的那些分支。"""
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE bars (symbol VARCHAR, market VARCHAR, frequency VARCHAR, "
        "adjust VARCHAR, ts TIMESTAMPTZ, " + ", ".join(f"{f} DOUBLE" for f in FIELDS) + ")"
    )
    # 三列过滤条件按 SQL_FILTER 给真值，好让这些行只被 classify 剔掉
    con.executemany(
        "INSERT INTO bars VALUES (?, 'CN', '1d', 'qfq', ?, 1.0, 2.0, 0.5, 1.5, 100.0, 1000.0)",
        [(s, f"{HISTORY_START} 00:00:00+08") for s in symbols],
    )
    return con


def test_dropped_by_classify_is_a_live_counter(tmp_path):
    """I1：classify 丢的行必须同时进 rows_in、rows_out 和 dropped_by_classify。"""
    con = _fake_bars(["600519.SH", "000001.SZ", "BOGUS.XX", "KOSPI.KR"])
    try:
        df, rows_in = load_bars(con)
    finally:
        con.close()
    assert rows_in == 4            # 剔除前
    assert len(df) == 2            # 剔除后：两个非 CN 代码不分类
    rep = write_csvs(df, tmp_path, rows_in)
    assert (rep.rows_in, rep.dropped_by_classify, rep.rows_out) == (4, 2, 2)
    assert rep.rows_in - rep.dropped_by_classify - rep.dup_rows - rep.bad_price_rows == rep.rows_out
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["SH600519.csv", "SZ000001.csv"]


def test_stale_csvs_are_pruned_and_the_file_count_is_checked(tmp_path):
    """导出窗口收窄后，磁盘上的旧 CSV 是陈旧产物，dump_bin 会把它们当数据吃进去。"""
    (tmp_path / "SH600519.csv").write_text("date,STALE\n2018-01-01,1\n", encoding="utf-8")
    (tmp_path / "SZ999999.csv").write_text("date,STALE\n2018-01-01,1\n", encoding="utf-8")
    df = pd.DataFrame(
        [
            {"symbol": "600519.SH", "date": "2025-01-02", "asset": "stock",
             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0, "amount": 20.0},
            {"symbol": "600519.SH", "date": "2025-01-03", "asset": "stock",
             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0, "amount": 20.0},
            {"symbol": "000001.SZ", "date": "2025-01-02", "asset": "stock",
             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0, "amount": 20.0},
        ]
    )
    rep = write_csvs(df, tmp_path, rows_in=len(df))
    assert rep.symbols == 2
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["SH600519.csv", "SZ000001.csv"]
    assert "STALE" not in (tmp_path / "SH600519.csv").read_text(encoding="utf-8")


def test_hygiene_report_prints_every_counter():
    """六个计数一个都不能少，否则丢行会重新变得隐形。"""
    text = HygieneReport(
        rows_in=10, rows_out=8, symbols=2, dropped_by_classify=1, dup_rows=1, bad_price_rows=0
    ).as_text()
    for key in ("rows_in", "rows_out", "symbols", "dropped_by_classify", "dup_rows", "bad_price_rows"):
        assert key in text
    assert text.splitlines()[0].split()[-1] == "10"
