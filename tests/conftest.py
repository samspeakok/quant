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
    """
    from qlib.contrib.data.handler import Alpha158

    return Alpha158(
        instruments="cn_stocks",
        start_time=HISTORY_START,
        end_time=SEGMENTS["test"][1],
        fit_start_time=HISTORY_START,
        fit_end_time=SEGMENTS["train"][1],
    )


@pytest.fixture(scope="session")
def dataset(alpha158_handler):
    """Task 7 产出、Task 8/9 复用的切分数据集（同样必须是 session 级、放 conftest）。"""
    from qlib.data.dataset import DatasetH

    return DatasetH(alpha158_handler, segments=SEGMENTS)
