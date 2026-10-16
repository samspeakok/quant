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
