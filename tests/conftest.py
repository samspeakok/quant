import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DUCKDB_PATH, HISTORY_START, QLIB_DIR, REGION, SEGMENTS  # noqa: E402

_HAS_DATA = (QLIB_DIR / "calendars" / "day.txt").exists()


@pytest.fixture(scope="session")
def qlib_ready():
    if not _HAS_DATA:
        pytest.skip(
            "qlib_data/ missing. Rebuild with:\n"
            "  python tools/export_bars.py && python tools/build_qlib_data.py"
        )
    import qlib

    qlib.init(provider_uri=str(QLIB_DIR), region=REGION)
    return True


@pytest.fixture(scope="session")
def db():
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="session")
def pool_members(qlib_ready):
    from qlib.data import D

    def members(pool: str) -> list[str]:
        return sorted(D.list_instruments(D.instruments(market=pool), as_list=True))

    return members


# Alpha158 的构造参数只写这一份：alpha158_handler 用它建对象，I1 的泄漏守卫断言同一份对象。
# 守卫因此不需要"读 fixture 源码"就能钉住真正喂给 qlib 的那四个时间参数。
# 改这里 = 改交付给 Task 8/9 的东西，tests/test_03_alpha_factors.py::
# test_no_processor_may_fit_on_data_beyond_the_train_window 会当场拦住 fit 窗口越界的改法。
ALPHA158_KWARGS: dict = {
    "instruments": "cn_stocks",
    "start_time": HISTORY_START,
    "end_time": SEGMENTS["test"][1],
    "fit_start_time": HISTORY_START,
    "fit_end_time": SEGMENTS["train"][1],
}


@pytest.fixture(scope="session")
def alpha158_handler(qlib_ready):
    """Task 6 产出、Task 7/8 复用的因子矩阵。

    放在 conftest 而不是 test_03 里：pytest 的 fixture **不跨测试模块可见**，而 Task 7 的
    `dataset` 要吃它 —— 定义在 tests/test_03_alpha_factors.py 里时，tests/test_04_dataset_split.py
    只会报 `fixture 'alpha158_handler' not found`。Alpha158 全池构建是本工程最慢的一步，
    session 级只建一次，后续断言全部复用同一个对象。

    end_time 用 SEGMENTS["test"][1] 而不是简报的 SEGMENTS["valid"][1]：DatasetH.prepare(seg)
    只是在 handler 已经载入内存的那一帧上按日期切片，不会回头再读数据层
    （qlib/data/dataset/__init__.py:171 `_prepare_seg` -> :181 `self.handler.fetch(slc)`）。
    按简报只载到 2025-12-31 时，test 段（2026-01-01..2026-09-30）实测直接空掉：
    `AssertionError: test is empty`，并连带 `ValueError: min()/max() iterable argument is empty`。
    留出尾部 HELDOUT_START 之后仍然不载进来。

    交付形态 —— Task 8 直接继承，不要靠猜：`Alpha158` 的默认 `infer_processors=[]`
    （`qlib/contrib/data/handler.py:105`），所以 `fetch(col_set="feature")` / `dataset.prepare(...)`
    拿到的是**原始、未归一化、含 NaN** 的矩阵，与 raw 帧逐元素相同（实测 159208x158、
    306421 个 NaN 单元格；其中 `VWAP0` 一整列 159208 个全 NaN —— 源字段 `$vwap` 在本库里
    不存在，见 task-5-7 报告"偏离 4"）。简报里"features 已被 Fillna 填成 0"那句说的是
    **Alpha360** 的默认处理器链（ProcessInf/ZScoreNorm/Fillna，`handler.py:41-45`），不是这个
    fixture；`tests/test_03_alpha_factors.py::test_handler_delivers_raw_nan_bearing_matrix` 把
    这件事钉成了断言（infer 帧必须等于 raw 帧、NaN 数必须还是那个数）。
    LightGBM 一类原生吸收缺失值的模型可以直接吃这帧；不能吸收的模型**必须自建 handler**
    （在自己的 infer_processors 里放 train 有界的 Fillna/归一化），不要回头改这个共享 fixture
    ——它一改，Task 6/7 的跨层比对和 Task 8 的原始值假设会一起失效。

    fit 窗口：`fit_start_time`/`fit_end_time` 今天不被任何处理器消费（infer 链为空、learn 链只有
    DropnaLabel + CSZScoreNorm，两者都不带 fit 窗口），所以它**看起来**无关紧要。它是留给
    后来者的枪口：谁往 infer/learn 链里加一个会拟合统计量的处理器，这个窗口就被 `check_transform_proc`
    原样注进那个处理器。守卫因此钉的是"喂给生产装配路径的那份窗口本身必须 train 有界"，
    而不是"当前恰好没泄漏"。
    """
    from qlib.contrib.data.handler import Alpha158

    return Alpha158(**ALPHA158_KWARGS)


@pytest.fixture(scope="session")
def alpha158_kwargs():
    """fixture 真正喂给 `Alpha158` 的那份参数字典（I1 守卫的唯一来源，别在测试里另抄一份）。"""
    return ALPHA158_KWARGS


@pytest.fixture(scope="session")
def dataset(alpha158_handler):
    """Task 7 产出、Task 8/9 复用的切分数据集（同样必须是 session 级、放 conftest）。"""
    from qlib.data.dataset import DatasetH

    return DatasetH(alpha158_handler, segments=SEGMENTS)
