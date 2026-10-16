"""Layer 2: the expression engine. Every expected value is recomputed from raw $close.

The MultiIndex level order of this store is ['instrument', 'datetime'] (Task 4 ruling), so
dropping level 0 leaves the datetime index used to align against the pandas recomputation.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import HISTORY_START, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")

CODE = "SH600519"
END = SEGMENTS["test"][1]


def _raw(code: str, fields: list[str]) -> pd.DataFrame:
    from qlib.data import D

    return D.features([code], fields, start_time=HISTORY_START, end_time=END)


def _series(code: str, expr: str) -> pd.Series:
    """One expression read back as a datetime-indexed Series over the full history window."""
    from qlib.data import D

    return D.features([code], [expr], start_time=HISTORY_START, end_time=END).iloc[:, 0].droplevel(0)


def _assert_matches_pandas(got: pd.Series, want: pd.Series, rtol: float) -> None:
    """Rows must line up first, values second.

    简报写的是 ``want.reindex(got.index).dropna()`` —— reindex 会把缺行/多行静默补成 NaN，
    再 dropna 一删，行数错位就再也看不见，只剩"比对到的都相等"。这里先钉住索引相等，
    再比数值，错位当场失败。
    """
    g, w = got.dropna(), want.dropna()
    assert g.index.equals(w.index), (
        f"row alignment broken: qlib {len(g)} 行 vs pandas {len(w)} 行; "
        f"only-qlib {g.index.difference(w.index)[:3].tolist()}, "
        f"only-pandas {w.index.difference(g.index)[:3].tolist()}"
    )
    assert np.allclose(g.to_numpy(), w.to_numpy(), rtol=rtol), (
        f"max rel err {np.max(np.abs(g.to_numpy() - w.to_numpy()) / np.maximum(np.abs(w.to_numpy()), 1e-12))}"
    )


def test_ref_shifts_exactly_one_day():
    raw = _raw(CODE, ["$close"])["$close"].droplevel(0)
    got = _series(CODE, "Ref($close, 1)")
    want = raw.shift(1)
    _assert_matches_pandas(got, want, rtol=1e-9)


def test_rolling_mean_matches_pandas():
    raw = _raw(CODE, ["$close"])["$close"].droplevel(0)
    got = _series(CODE, "Mean($close, 5)")
    # qlib/data/ops.py:752 —— Rolling 一律 min_periods=1（"expanding/rolling 与 pandas 同义"），
    # 首 N-1 天用不满的窗口出值。简报按 pandas 默认 min_periods=N 写参照，于是 qlib 侧多出
    # 4 个预热行，两条数组长度不等，np.allclose 直接 ValueError。参照必须按 qlib 的真实语义写。
    want = raw.rolling(5, min_periods=1).mean()
    _assert_matches_pandas(got, want, rtol=1e-6)


def test_std_of_returns_matches_pandas():
    raw = _raw(CODE, ["$close"])["$close"].droplevel(0)
    ret = raw / raw.shift(1) - 1
    got = _series(CODE, "Std($close/Ref($close,1)-1, 20)")
    want = ret.rolling(20, min_periods=1).std()
    _assert_matches_pandas(got, want, rtol=1e-5)


def test_rank_is_rolling_percentile_within_one_instrument():
    """pyqlib 0.9.7 的 Rank 是**时序滚动分位**，不是横截面算子。

    简报写 ``Rank($volume)`` 并按"横截面、同一天落在 (0,1]"来断言，实测抛
    ``TypeError: Rank.__init__() missing 1 required positional argument: 'N'``
    （ops.py:1133 `class Rank(Rolling)`，docstring 自己写着 "Rolling Rank (Percentile)"；
    表达式引擎里没有任何按日 groupby 的算子）。所以这里按真实性能断言：
    窗口内最后一个值必须等于该标的自己那 N 天的百分位排名。
    """
    vol = _raw(CODE, ["$volume"])["$volume"].droplevel(0)
    got = _series(CODE, "Rank($volume, 20)")
    want = vol.rolling(20, min_periods=1).rank(pct=True)
    _assert_matches_pandas(got, want, rtol=1e-5)
    v = got.dropna()
    assert v.min() > 0 and v.max() <= 1.0, f"rank out of (0, 1]: min={v.min()} max={v.max()}"


def test_rank_batched_over_instruments_stays_in_unit_interval(pool_members):
    """多标的批量读同一个滚动算子：每行都必须落在 (0, 1]。"""
    from qlib.data import D

    codes = pool_members("cn_etf")[:30]
    assert len(codes) == 30, f"cn_etf pool too small for a 30-instrument batch: {len(codes)}"
    day = END
    df = D.features(codes, ["Rank($volume, 20)"], start_time=day, end_time=day).dropna()
    assert not df.empty, f"{len(codes)} ETFs returned no rank on {day}"
    # index 层级顺序 ['instrument','datetime']：单日窗口下每个标的只剩一行
    assert set(df.index.get_level_values("datetime")) == {pd.Timestamp(day)}, "single-day read leaked days"
    v = df.iloc[:, 0]
    assert v.min() > 0 and v.max() <= 1.0, f"rank out of (0, 1]: min={v.min()} max={v.max()} ({len(v)} rows)"


def test_all_expressions_align_on_length():
    from qlib.data import D

    base = D.features([CODE], ["$close"], start_time=HISTORY_START, end_time=END)
    assert not base.empty, f"{CODE} reads back empty over {HISTORY_START}..{END}"
    for expr in ["Ref($close,1)", "Mean($close,5)", "Std($close,20)", "Max($high,10)", "Min($low,10)"]:
        got = D.features([CODE], [expr], start_time=HISTORY_START, end_time=END)
        assert len(got) == len(base), f"{expr} changed row count"
        assert got.index.equals(base.index), f"{expr} changed row identity"
