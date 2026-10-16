"""Layer 4: DatasetH segmenting.

The `dataset` fixture lives in tests/conftest.py (session scope) because pytest fixtures are
not visible across test modules and Task 8 consumes it; it wraps the shared `alpha158_handler`.
Every index access here uses level *names* — the handler frame is datetime-first (Task 6).
"""

import pytest

from config.settings import SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


def test_each_segment_is_non_empty(dataset):
    for name in SEGMENTS:
        df = dataset.prepare(name, col_set="feature")
        assert not df.empty, f"{name} is empty"


def test_segment_date_ranges_are_disjoint_and_ordered(dataset):
    def _days(seg):
        idx = dataset.prepare(seg, col_set="feature").index.get_level_values("datetime")
        return min(idx), max(idx)

    tr0, tr1 = _days("train")
    va0, va1 = _days("valid")
    te0, te1 = _days("test")
    assert tr1 < va0, "train and valid overlap"
    assert va1 < te0, "valid and test overlap"
    assert tr0 < tr1 < va0 < va1 < te0 < te1


def test_no_instrument_appears_in_two_segments_across_the_boundary(dataset):
    """切分按日期做，标的可以在多个段里重复出现，但它的任一交易日不该跨段。"""
    def _days(seg):
        return set(dataset.prepare(seg, col_set="feature").index.get_level_values("datetime"))

    assert not (_days("train") & _days("valid"))
    assert not (_days("valid") & _days("test"))


def test_feature_and_label_share_the_same_index(dataset):
    feat = dataset.prepare("train", col_set="feature")
    lab = dataset.prepare("train", col_set="label")
    assert feat.index.equals(lab.index)


def test_heldout_tail_absent_from_every_segment(dataset):
    import pandas as pd

    from config.settings import HELDOUT_START

    for name in SEGMENTS:
        days = dataset.prepare(name, col_set="feature").index.get_level_values("datetime")
        assert max(days) < pd.Timestamp(HELDOUT_START), f"{name} leaks into the held-out tail"


def test_test_segment_labels_reach_into_the_heldout_tail_by_exactly_two_rows_per_instrument(dataset):
    """I4：上一条只看**行日期**，看不见标签的前看窗口把收盘价读到留出区里去 —— 这里把它量出来并钉住。

    按设计成立的标签是 `Ref($close, -2)/Ref($close, -1) - 1`（不改，见 task-5-7 报告"偏离"）。
    `Ref` 的 `get_extended_window_size` 向右扩 2 个日历日（qlib/data/ops.py:808-826），所以 test 段
    最后两个交易日的标签用的是 HELDOUT_START **之后**的收盘价：SH600519 在 2026-09-30 的标签
    由 2026-10-08 与 2026-10-09 的收盘价算出（10-01..10-07 是国庆休市，库的日历里根本没有这两天，
    报告里"用的是 2026-10-01/02 收盘价"那句是错的）。行日期仍在 HELDOUT_START 之前，
    所以这条例外**只能靠数行数**发现：实测 660 行，恰好是尾部 330 个标的 × 2 行。

    （330 而不是 test 段全部的 626 个标的：库里 287 个标的的 bar 停在 2026-08-03、8 个停在 08-14、
    1 个停在 09-14，它们在 test 段末没有行。这是数据层事实，Task 8/9 的 test 段可打分数只覆盖这 330 只。）

    这条测试钉的是"例外被量出来"，不是"例外被消掉"：数字一变（尾部交易日数、标的数、
    或哪个标签变成 NaN）就当场失败，而不是继续沉默。
    """
    import pandas as pd

    from qlib.data import D

    from config.settings import HELDOUT_START

    label = dataset.prepare("test", col_set="label")
    assert label.shape[1] == 1, f"标签帧不该有第二列：{list(label.columns)}"

    cal = list(D.calendar())
    pos = {d: i for i, d in enumerate(cal)}
    held = pd.Timestamp(HELDOUT_START)
    days = sorted(set(label.index.get_level_values("datetime")))
    assert max(days) == pd.Timestamp(SEGMENTS["test"][1]), "test 段末行日期与 SEGMENTS['test'][1] 不一致"

    # 一个交易日行的标签要读它之后第 1、第 2 个日历日的收盘价。
    crossing = [d for d in days if any(pos[d] + k < len(cal) and cal[pos[d] + k] >= held for k in (1, 2))]
    assert len(crossing) == 2 and crossing == days[-2:], (
        f"标签跨进留出区的交易日不是预期的尾部两天，实测 {[str(d.date()) for d in crossing]}；"
        f"库日历尾部 {[str(d.date()) for d in cal[-4:]]}"
    )

    sub = label[label.index.get_level_values("datetime").isin(crossing)]
    rows = len(sub)
    live = int(sub.notna().all(axis=1).sum())
    instruments = sub.index.get_level_values("instrument").nunique()
    assert rows == live == 660, (
        f"test 段中标签依赖 HELDOUT_START({HELDOUT_START}) 之后收盘价的行数 = {rows}，"
        f"其中标签非 NaN（真的算出了留出区价格）= {live}，钉住的实测值是 660"
    )
    per_inst = sub.notna().all(axis=1).groupby(level="instrument").sum()
    assert rows == 2 * instruments == 2 * per_inst.size and set(per_inst.unique().tolist()) == {2}, (
        f"留出区依赖不是每个尾部标的恰好 2 行：{rows} 行 / {instruments} 个标的，"
        f"分布 {sorted(per_inst.value_counts().to_dict().items())}"
    )

    # 把"消费的是留出区价格"落到一个可核对的点上：SH600519 尾部两行的标签必须能由
    # 数据层 2026-09-30 / 2026-10-08 / 2026-10-09 三个收盘价重算出来。
    code = "SH600519"
    closes = D.features([code], ["$close"], start_time=crossing[-1], end_time=cal[-1])["$close"]
    closes = closes.droplevel("instrument")
    fwd = [d for d in closes.index if d >= held]
    assert [str(d.date()) for d in fwd[:2]] == ["2026-10-08", "2026-10-09"], (
        f"留出区头两个交易日变了：{[str(d.date()) for d in fwd[:3]]}"
    )
    for day, want in (
        (crossing[1], float(closes[fwd[1]] / closes[fwd[0]] - 1)),  # 2026-09-30 -> 10-09 / 10-08
        (crossing[0], float(closes[fwd[0]] / closes[crossing[1]] - 1)),  # 2026-09-29 -> 10-08 / 09-30
    ):
        got = float(label.loc[(day, code)].iloc[0])
        assert abs(got - want) <= 1e-6 * abs(want) + 1e-9, (
            f"{code} {day}: 标签 {got} vs 用留出区收盘价重算 {want}"
        )
