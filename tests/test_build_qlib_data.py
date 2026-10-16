from pathlib import Path

import pytest

from config.settings import QLIB_DIR


pytestmark = pytest.mark.skipif(
    not (QLIB_DIR / "calendars" / "day.txt").exists(),
    reason="qlib_data not built; run: python tools/export_bars.py && python tools/build_qlib_data.py",
)


def test_calendar_is_strictly_increasing_and_unique():
    dates = Path(QLIB_DIR / "calendars" / "day.txt").read_text(encoding="utf-8").split()
    assert dates == sorted(set(dates))
    # 收窄导出后日历应约 529 个交易日；若这里出现数千，说明时间下界没生效，
    # 指数的 1990 年数据又把 NaN 填充撑回来了
    assert 400 <= len(dates) <= 600, f"calendar has {len(dates)} days, expected ~529"


def test_three_pool_files_exist_with_expected_sizes():
    """实测 CN 日线前复权：股票 626 / ETF 62 / 指数 10。"""
    expected = {"cn_stocks": 626, "cn_etf": 62, "cn_index": 10}
    for name, count in expected.items():
        path = QLIB_DIR / "instruments" / f"{name}.txt"
        assert path.exists(), f"missing {path}"
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == count, f"{name}: got {len(lines)} expected {count}"


def test_no_index_leaks_into_the_tradable_stock_pool():
    """指数混进可交易池会凭空抬高回测收益 —— 这是本项目最不可接受的一类错误。"""
    def members(name):
        return {
            ln.split("\t")[0]
            for ln in (QLIB_DIR / "instruments" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    stocks, index = members("cn_stocks"), members("cn_index")
    assert not (stocks & index)
    assert "SH000001" not in stocks   # 上证指数
    assert "SH000905" not in stocks   # 中证500
    assert "SH000300" in index


def test_benchmark_instrument_has_features():
    bench = QLIB_DIR / "features" / "sh000300"
    assert bench.exists() and (bench / "close.day.bin").exists()
