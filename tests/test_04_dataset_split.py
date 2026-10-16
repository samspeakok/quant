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
