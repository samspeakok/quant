import pytest
import duckdb

from config.settings import DUCKDB_PATH, EXPORT_CSV_DIR
from tools.export_bars import load_bars, connect_readonly, HygieneReport


def test_writing_to_the_source_database_is_rejected(tmp_path):
    """主系统正在写这个库。我们的连接必须是只读的，且这一约束要有测试兜底。"""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    with pytest.raises(Exception):
        con.execute("CREATE TABLE _qoder_probe (x INTEGER)")
    con.close()


def test_export_is_reproducible_and_market_filtered():
    con = connect_readonly()
    df = load_bars(con)
    con.close()
    # 实测：CN + 1d + qfq 共 698 个标的
    assert len(set(df["symbol"])) == 698
    assert df["asset"].isin(["stock", "etf", "index"]).all()
    # (?:...) 非捕获组：断言不变，但避免 pandas 对「有捕获组却只判布尔」的 UserWarning
    assert df["symbol"].str.contains(r"\.(?:SH|SZ|BJ)$").all()
    assert not df["symbol"].str.endswith((".US", ".HK", ".KR", ".GLOBAL")).any()
