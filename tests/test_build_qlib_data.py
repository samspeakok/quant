from pathlib import Path
import subprocess
import sys

import pytest

from config.settings import EXPORT_CSV_DIR, QLIB_DIR, VENDOR_DUMP_BIN
from tools.build_qlib_data import verify_store


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


# ---------------------------------------------------------------- verify_store 闸门
# 闸门必须是绝对的：n_all 与 n_feat 都由同一次 dump 产出，两者相等只证明"这次 dump 自洽"，
# 不证明"该有的标的都在"。用 --limit_nums=3 跑一遍就能得到 3 == 3 的平凡通过，随后
# write_pools 会写出 3 成员的池、main() 返回 0。锚点取 tools/export_bars.py 独立产出的
# CSV 张数（导出器先清空 *.csv 再断言 written == symbols，故 1 个 CSV == 1 个标的）。

def _instruments(qlib_dir: Path) -> list[str]:
    return [
        ln.split("\t")[0]
        for ln in (qlib_dir / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _complete_feature_dirs(qlib_dir: Path) -> set[str]:
    return {
        d.name
        for d in (qlib_dir / "features").iterdir()
        if d.is_dir() and (d / "close.day.bin").exists()
    }


def _synthetic_store(root: Path, codes: list[str], feature_dirs: list[str], binned: list[str]) -> Path:
    """搭一个最小库：all.txt 有 codes，features/ 下有 feature_dirs，其中 binned 带 close.day.bin。"""
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text("2024-08-05\n2024-08-06\n", encoding="utf-8")
    (root / "instruments").mkdir()
    (root / "instruments" / "all.txt").write_text(
        "".join(f"{c}\t2024-08-05\t2024-08-06\n" for c in codes), encoding="utf-8"
    )
    for name in set(feature_dirs) | set(binned):
        d = root / "features" / name
        d.mkdir(parents=True)
        if name in binned:
            (d / "close.day.bin").write_bytes(b"\x00\x00\x00\x00")
    return root


def _csv_dir(root: Path, names: list[str]) -> Path:
    d = root / "csv"
    d.mkdir()
    for name in names:
        (d / f"{name}.csv").write_text("symbol,date,close\n", encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def truncated_store(tmp_path_factory) -> Path:
    """真跑一次 --limit_nums=3 的 dump 到临时库；绝不写 qlib_data/，只读导出 CSV。"""
    if not VENDOR_DUMP_BIN.exists() or not EXPORT_CSV_DIR.exists():
        pytest.skip("需要 vendor/qlib/scripts/dump_bin.py 与 tools/_export/csv/ 才能建临时库")
    store = tmp_path_factory.mktemp("truncated_store")
    cmd = [
        sys.executable, str(VENDOR_DUMP_BIN), "dump_all",
        f"--data_path={EXPORT_CSV_DIR}",
        f"--qlib_dir={store}",                      # 临时目录，不是 settings.QLIB_DIR
        "--freq=day", "--date_field_name=date", "--symbol_field_name=symbol",
        "--exclude_fields=date", "--max_workers=4", "--limit_nums=3",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, f"dump_bin 失败：{proc.stdout[-500:]}{proc.stderr[-500:]}"
    return store


def test_gate_anchor_count_is_consistent_with_the_expected_pools():
    """CSV 张数是闸门的绝对锚点，它必须与既定池期望 626+62+10 同源。"""
    assert len(list(EXPORT_CSV_DIR.glob("*.csv"))) == 626 + 62 + 10


def test_real_store_passes_the_absolute_gate():
    """磁盘上已交付的库（529 日历日 / 698 标的 / 698 个 close.day.bin）必须通过绝对闸门。"""
    verify_store(QLIB_DIR, EXPORT_CSV_DIR)


def test_truncated_dump_passes_the_relative_gate_but_not_the_absolute_one(truncated_store):
    codes = _instruments(truncated_store)
    bins = _complete_feature_dirs(truncated_store)
    n_csv = len(list(EXPORT_CSV_DIR.glob("*.csv")))

    # 这三行就是旧闸门被"平凡满足"的现场证据：日历看着正常、bin 数与 all.txt 一致。
    # 若把 verify_store 改回只比较 n_feat 与 n_all，下面的 raises 就不会抛 → 本测试变红。
    assert len(codes) == len(bins)
    assert len(codes) < n_csv
    assert (truncated_store / "calendars" / "day.txt").read_text().split()

    with pytest.raises(RuntimeError) as exc:
        verify_store(truncated_store, EXPORT_CSV_DIR)
    msg = str(exc.value)
    assert str(len(codes)) in msg and str(n_csv) in msg, f"闸门消息没带上两个数字：{msg}"
    # 必须在写池之前拦住：临时库里不该出现池文件
    assert not (truncated_store / "instruments" / "cn_stocks.txt").exists()


def test_feature_dir_without_close_bin_cannot_pass(tmp_path):
    """目录在、close.day.bin 不在 → 缺 bin 的那个不能靠别的目录凑数蒙过去。"""
    codes = ["SH000001", "SH000016"]
    store = _synthetic_store(tmp_path / "store", codes, ["sh000001", "sh000016"], ["sh000001"])
    csvs = _csv_dir(tmp_path, codes)
    with pytest.raises(RuntimeError) as exc:
        verify_store(store, csvs)
    msg = str(exc.value)
    assert "1" in msg and "2" in msg
    assert "sh000016" in msg, f"消息要点名缺 bin 的目录：{msg}"


def test_equal_counts_with_wrong_feature_dirs_cannot_pass(tmp_path):
    """数量相等但身份不对：all.txt 是 SH000001/SH000016，features/ 却是 sh000001/sh999999。"""
    codes = ["SH000001", "SH000016"]
    store = _synthetic_store(tmp_path / "store", codes, ["sh000001", "sh999999"], ["sh000001", "sh999999"])
    csvs = _csv_dir(tmp_path, codes)
    assert len(_instruments(store)) == len(_complete_feature_dirs(store))  # 纯计数会放行
    with pytest.raises(RuntimeError) as exc:
        verify_store(store, csvs)
    msg = str(exc.value)
    assert "sh000016" in msg and "sh999999" in msg, f"消息要点名缺失/多余：{msg}"


def test_empty_export_dir_fails_closed(tmp_path):
    """锚点自己为空时不许"0 == 0"通过。"""
    codes = ["SH000001", "SH000016"]
    store = _synthetic_store(tmp_path / "store", codes, [c.lower() for c in codes], [c.lower() for c in codes])
    empty = tmp_path / "csv"
    empty.mkdir()
    with pytest.raises(RuntimeError):
        verify_store(store, empty)
