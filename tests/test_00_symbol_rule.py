import pytest

from config.settings import HELDOUT_START, POOLS, SEGMENTS, classify_symbol, to_qlib_code


@pytest.mark.parametrize(
    "symbol, expected",
    [
        # 沪市主板与科创板
        ("600519.SH", "stock"),
        ("601318.SH", "stock"),
        ("603288.SH", "stock"),
        ("605499.SH", "stock"),
        ("688256.SH", "stock"),
        # 深市主板与创业板
        ("000001.SZ", "stock"),
        ("002594.SZ", "stock"),
        ("300750.SZ", "stock"),
        ("301158.SZ", "stock"),
        ("302132.SZ", "stock"),
        # 北交所
        ("832004.BJ", "stock"),
        ("430139.BJ", "stock"),
        ("889117.BJ", "stock"),
        # 北证系列指数：899xxx 必须是 index，不能落进可交易股票池
        ("899001.BJ", "index"),
        # ETF：15/16 开头在深市，51/56/58 与 52 开头在上海
        ("159611.SZ", "etf"),
        ("510300.SH", "etf"),
        ("520880.SH", "etf"),
        ("562500.SH", "etf"),
        ("588000.SH", "etf"),
        # 指数：000 开头在上海是指数，399 开头在深市是指数
        ("000300.SH", "index"),
        ("000001.SH", "index"),
        ("000905.SH", "index"),
        ("399006.SZ", "index"),
        # 非 CN 市场一律拒绝，绝不能进导出
        (".DJI.US", None),
        ("^HSI.HK", None),
        ("DXY.GLOBAL", None),
        ("KOSPI.KR", None),
        # 无法识别
        ("BOGUS", None),
    ],
)
def test_classify_symbol(symbol, expected):
    assert classify_symbol(symbol) == expected


def test_shanghai_000_prefix_is_index_not_stock():
    """000 开头在深圳是股票、在上海是指数。这是 asset_type 标签出错的确切位置。"""
    assert classify_symbol("000001.SH") == "index"
    assert classify_symbol("000001.SZ") == "stock"


def test_to_qlib_code():
    assert to_qlib_code("600519.SH") == "SH600519"
    assert to_qlib_code("000001.SZ") == "SZ000001"
    assert to_qlib_code("159611.SZ") == "SZ159611"


def test_benchmark_is_a_classified_index():
    from config.settings import BENCHMARK

    assert classify_symbol("000300.SH") == "index"
    assert to_qlib_code("000300.SH") == BENCHMARK


def test_segments_are_ordered_and_disjoint():
    assert SEGMENTS["train"][1] < SEGMENTS["valid"][0]
    assert SEGMENTS["valid"][1] < SEGMENTS["test"][0]


def test_segments_are_the_exact_measured_windows():
    """钉住四个端点。只断言顺序的话，整段窗口平移一年也照样全绿。"""
    assert SEGMENTS == {
        "train": ("2025-01-01", "2025-08-31"),
        "valid": ("2025-09-01", "2025-12-31"),
        "test": ("2026-01-01", "2026-09-30"),
    }


def test_heldout_tail_is_excluded_from_every_segment():
    """数据到 2026-10-14，这段尾部不进任何区间，留作样本外参照。"""
    for name, (_start, end) in SEGMENTS.items():
        assert end < HELDOUT_START, f"{name} leaks into the held-out tail"


def test_pools_map_to_the_three_classifiers():
    assert set(POOLS.values()) == {"stock", "etf", "index"}


# ---------------------------------------------------------------- A4: settings 常量
# SQL_FILTER / REGION / FREQ / 六个路径是 Task 2-10 的**共同前提**，此前一条测试都没钉住：
# SQL_FILTER 少一个条件要到下游行数对不上才暴露，REGION 写错（"us"）会让日历与复权口径整体错位，
# FREQ 写错（"1min"）让 dump_bin 与所有 `*.day.bin` 的读取点一起对不上，路径写错更是直接读空库。
# 这一节把它们钉成字面量等值 —— 出错要在这条链的最上游当场红。

def test_sql_filter_is_the_exact_cn_daily_qfq_premise():
    """Task 2 的查询、Task 1 的 698 张导出 CSV 都吃这一串；它是全工程最大的单点前提。"""
    from config.settings import SQL_FILTER

    # 三个条件一个都不能少：少 market 会混进 US/HK/KR/GLOBAL 的 ~4.5 万行（幽灵交易日），
    # 少 frequency 会混进分钟线，少 adjust 会让复权口径混用。
    assert SQL_FILTER == "frequency = '1d' AND adjust = 'qfq' AND market = 'CN'"


def test_region_and_freq_are_the_qlib_store_coordinates():
    """qlib.init(region=...) 与 provider 的 `{field}.{FREQ}.bin` 命名都直接吃这两个值。"""
    from config.settings import FREQ, REGION

    assert REGION == "cn", f"REGION 必须是 qlib 的 cn 市场枚举：{REGION!r}"
    assert FREQ == "day", f"FREQ 必须是 day：库里所有 bin 都按 close.day.bin 命名，{FREQ!r} 会让读取点全空"


def test_heldout_start_is_the_measured_tail_boundary():
    """数据到 2026-10-14，留出尾部从这天起；它不属于任何 segment（上一条已验）。"""
    assert HELDOUT_START == "2026-10-01"


def test_path_constants_are_repo_relative_or_an_existing_external_db():
    """四个库内路径必须由 ROOT 拼出来；数据库是唯一的外部路径，它必须真的存在。"""
    from config.settings import ARTIFACTS, DUCKDB_PATH, EXPORT_CSV_DIR, QLIB_DIR, ROOT, VENDOR_DUMP_BIN

    repo_paths = {
        "VENDOR_DUMP_BIN": VENDOR_DUMP_BIN,
        "EXPORT_CSV_DIR": EXPORT_CSV_DIR,
        "QLIB_DIR": QLIB_DIR,
        "ARTIFACTS": ARTIFACTS,
    }
    for name, path in repo_paths.items():
        assert path.is_absolute(), f"{name} 必须是绝对路径，实际 {path}"
        assert path.is_relative_to(ROOT), (
            f"{name} 不再挂在 ROOT ({ROOT}) 下：{path}。它一旦跑到仓库外面，"
            "qlib_data / 导出 CSV / vendor 就会各读各的，所有层比对同时失效"
        )

    assert not DUCKDB_PATH.is_relative_to(ROOT), "市场数据库是仓库外的既有资产，被搬进来说明导出源变了"
    if not DUCKDB_PATH.exists():
        pytest.skip(f"外部市场数据库不在本机：{DUCKDB_PATH}（导出步骤需要它，路径常量本身已经断过）")
    assert DUCKDB_PATH.is_file(), f"{DUCKDB_PATH} 存在但不是文件"


# ------------------------------------------------------- A3：闸门测试的 skip 必须逐条挂
# 这条守卫**放在本文件而不是 tests/test_build_qlib_data.py** 是刻意的：本模块不依赖 qlib_data/，
# 所以在"全新 clone、没有库"的环境里它照样执行；而写在闸门模块里的话，一旦有人把模块级
# `pytestmark = skipif(没有库)` 加回来（A3 修掉的正是这条），守卫会跟着一起 SKIP ——
# 恰好在唯一会出问题的那个环境下失明。
#
# 依赖分档（tests/test_build_qlib_data.py 顶部）：
#   requires_store  读已交付的 qlib_data/（6 条）
#   requires_export 只读 tools/_export/csv（锚点张数、--limit_nums=3 的截断库）
#   无标记          只用 tmp_path 造临时库的探针：任何环境必须真的跑

def test_the_gate_module_skips_are_per_test_not_module_wide():
    """没有 qlib_data/ 时，绝对闸门测试必须照跑，而不是整模块静默 SKIP。"""
    import tests.test_build_qlib_data as gate

    module_marks = {m.name for m in getattr(gate, "pytestmark", [])}
    assert not (module_marks & {"skipif", "skip", "xfail"}), (
        f"tests/test_build_qlib_data.py 又出现了模块级标记 {sorted(module_marks)}：全新 clone 会把"
        "只依赖 tools/_export/csv + vendor dump_bin 的绝对闸门测试一起静默 SKIP"
    )

    must_run_without_the_store = [
        "test_feature_dir_without_close_bin_cannot_pass",
        "test_equal_counts_with_wrong_feature_dirs_cannot_pass",
        "test_empty_export_dir_fails_closed",
        "test_write_pools_fails_closed_when_an_instrument_is_not_classified",
        "test_write_pools_fails_closed_when_two_pools_share_an_instrument",
    ]

    must_keep_the_store_skip = [
        "test_calendar_is_strictly_increasing_and_unique",
        "test_three_pool_files_exist_with_expected_sizes",
        "test_no_index_leaks_into_the_tradable_stock_pool",
        "test_benchmark_instrument_has_features",
        "test_real_store_passes_the_absolute_gate",
        "test_write_pools_reanchors_the_delivered_store",
    ]

    # 名字也要钉住：测试被改名/删掉时这条守卫不许悄悄少查一项。
    for name in must_run_without_the_store + must_keep_the_store_skip:
        assert hasattr(gate, name), f"tests/test_build_qlib_data.py 里找不到 {name}：分档清单已经过期"

    for name in must_run_without_the_store:
        marks = {m.name for m in getattr(getattr(gate, name), "pytestmark", [])}
        assert not (marks & {"skipif", "skip", "xfail"}), (
            f"{name} 只用 tmp_path 造库，却被挂上 {sorted(marks)}：它本该在任何机器上跑"
        )
    for name in must_keep_the_store_skip:
        skipifs = [m for m in getattr(getattr(gate, name), "pytestmark", []) if m.name == "skipif"]
        assert skipifs, f"{name} 读的是交付库，却没了逐条 skip（全新 clone 上它会直接报错）"
        reason = skipifs[0].kwargs["reason"]
        assert "tools/export_bars.py" in reason and "tools/build_qlib_data.py" in reason, (
            f"{name} 的 skip 理由必须点名重建命令，实际：{reason}"
        )
