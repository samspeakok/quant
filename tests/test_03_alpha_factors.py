"""Layer 3: the shipped alpha factor libraries.

`alpha158_handler` is defined in tests/conftest.py (session scope), not here: pytest fixtures
are not visible across test modules, and Task 7's `dataset` fixture consumes it.

Index caveat for everything below: the handler frame is **datetime-first**
(``qlib/data/dataset/loader.py:160`` ``swap_level: bool = True`` -> ``:226`` ``df.swaplevel()``),
which is the reverse of raw ``D.features`` (``['instrument', 'datetime']``, Task 4). All access
here therefore goes through level *names*, never through positional tuples.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from config.settings import QLIB_DIR, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


def _store_fields(one_instrument: str) -> set[str]:
    """原始字段清单：库里的 .day.bin 文件名就是 Qlib 能读到的 $field。"""
    return {p.name.split(".")[0] for p in (QLIB_DIR / "features" / one_instrument.lower()).glob("*.day.bin")}


def _dead_and_explained(feat: pd.DataFrame, config: tuple[list[str], list[str]]) -> tuple[set, set, set]:
    """返回 (全 NaN 的因子列, 由"表达式引用的原始字段不在库里"解释掉的列, 库里的字段清单)。"""
    fields, names = config
    assert not feat.empty, "factor frame came back empty"
    assert len(fields) == len(names) == feat.shape[1], (
        f"declared {len(fields)} expressions / {len(names)} names vs frame {feat.shape[1]} columns"
    )
    have = _store_fields(feat.index.get_level_values("instrument")[0])
    explained = {n for n, expr in zip(names, fields) if not set(re.findall(r"\$(\w+)", expr)) <= have}
    dead = {c for c in feat.columns if feat[c].isna().all()}
    return dead, explained, have


def test_feature_column_count_matches_declared_config(alpha158_handler):
    """用 handler 自己的配置自证，不写死 158 —— 改配置不该误报失败。"""
    feat = alpha158_handler.fetch(col_set="feature")
    _ops, names = alpha158_handler.get_feature_config()
    assert feat.shape[1] == len(names)
    assert list(feat.columns[: len(names)]) == list(names)[: len(feat.columns)]


def test_dead_feature_columns_are_all_explained_by_missing_raw_fields(alpha158_handler):
    """实测：cn_stocks 全池 158 列里恰有 1 列全 NaN —— VWAP0，且它确实只可能由缺字段造成。

    简报断言"零全 NaN 列"，实测失败：``AssertionError: 1 dead columns, e.g. ['VWAP0']``。
    根因不在因子引擎：Alpha158 默认价格组请求 VWAP（``contrib/data/loader.py:132`` 生成表达式
    ``$vwap/$close``），而本工程 dump_bin 只有 open/high/low/close/volume/amount/factor
    七个字段，库里没有 vwap.day.bin（``D.features([code], ["$vwap"])`` 实测返回空表）。
    所以断言写成"死列集合 == 缺原始字段解释掉的列集合"：任何别的死列（算子算不出来、
    日历错位、池里少标的）都会当场失败，而不是被一句"停牌嘛"咽下去。
    """
    feat = alpha158_handler.fetch(col_set="feature")
    dead, explained, have = _dead_and_explained(feat, alpha158_handler.get_feature_config())
    assert dead == explained, f"dead columns {sorted(dead)} are not explained by missing fields {sorted(explained)}"
    assert explained == {"VWAP0"}, f"库里缺的不只是 vwap：{sorted(explained)}"
    assert "vwap" not in have, "库里一旦补上 vwap，这条裁定要重新评估（见 task-5-7 报告）"


def test_price_derived_columns_match_raw_data(alpha158_handler):
    """因子矩阵里的价格衍生列必须和数据层读回来的原始值一致。

    两处照抄简报会当场崩：
    1. 没有名为 ``CLOSE/1`` 的列，也没有任何一列等于原始收盘价 —— Alpha158 的价格组一律
       除以 $close 归一，真实列名是 OPEN0/HIGH0/LOW0/VWAP0（实测 get_feature_config()）。
    2. ``row.name[1]`` 不是日期而是标的代码（handler 帧是 datetime 在前），于是
       ``D.features(..., start_time='SH600000')`` 抛
       ``pandas DateParseError: Unknown datetime string format, unable to parse: SH600000``。
    可成立的等价断言是：每一列 = 数据层同一字段的原始读数 / 同一天的原始收盘价。
    """
    from qlib.data import D

    feat = alpha158_handler.fetch(col_set="feature")
    inst = feat.index.get_level_values("instrument")[0]
    day = feat.index.get_level_values("datetime")[0]
    raw = D.features([inst], ["$open", "$high", "$low", "$close"], start_time=day, end_time=day)
    assert len(raw) == 1, f"{inst} {day}: 数据层应该只有一行，实际 {len(raw)}"
    row = feat.iloc[0]
    close = float(raw["$close"].iloc[0])
    for col, field in [("OPEN0", "$open"), ("HIGH0", "$high"), ("LOW0", "$low")]:
        want = float(raw[field].iloc[0]) / close
        got = float(row[col])
        assert abs(got - want) <= 1e-6 * abs(want) + 1e-9, f"{col} {inst} {day}: 因子矩阵 {got} vs 数据层 {want}"


def test_label_uses_the_declared_two_day_forward_return(alpha158_handler):
    """Alpha158 的标签按设计前看两天，这是预测目标不是数据泄漏，别当 bug 改。"""
    from qlib.data import D

    label = alpha158_handler.fetch(col_set="label")
    assert label.shape[1] == 1
    assert not label.isna().all().iloc[0]

    exprs, names = alpha158_handler.get_label_config()
    assert names == ["LABEL0"], f"标签列名不是 LABEL0：{names}"
    assert "Ref($close, -2)" in exprs[0] and "Ref($close, -1)" in exprs[0], (
        f"标签不再是前看两天的收益：{exprs[0]}"
    )
    # 数值侧：矩阵里那一列必须等于数据层同一表达式的读数（该表达式的算术已在 Task 5 逐点比对过）
    got = label.iloc[:, 0].dropna()
    inst = got.index.get_level_values("instrument")[0]
    day = got.index.get_level_values("datetime")[0]
    want = D.features([inst], [exprs[0]], start_time=day, end_time=day).iloc[:, 0]
    assert len(want) == 1, f"{inst} {day}: 数据层标签读数应该有 1 行，实际 {len(want)}"
    ref, val = float(want.iloc[0]), float(got.iloc[0])
    assert abs(ref - val) <= 1e-6 * abs(ref) + 1e-9, f"{inst} {day}: 矩阵标签 {val} vs 数据层 {ref}"


def test_alpha360_shape_is_time_series_windows():
    """Alpha360 = 6 个字段 x 60 个窗口的原始序列；列数由它自己的配置自证。

    两处按实测改写：
    1. ``Alpha360`` **没有** ``get_feature_config``（只有 Alpha158 有；
       ``hasattr(Alpha360, 'get_feature_config') == False``），配置在它自己 __init__ 里直接取
       ``Alpha360DL.get_feature_config()``，这里就用同一个来源。
    2. 构造时 qlib 自己的默认 infer 处理器发 ``RuntimeWarning: Mean of empty slice`` ——
       和 Alpha158 同一个根因：60 个 VWAP 列因库里没有 $vwap 而全 NaN，ZScoreNorm 对空切片
       求均值。本工程跑在 ``-W error`` 下，这正是简报第三条失败。这里把它**显式记录并逐条
       判定**（除此之外不许出现任何警告），而不是关掉 -W error 或换掉处理器。
    """
    from qlib.contrib.data.handler import Alpha360
    from qlib.contrib.data.loader import Alpha360DL
    from qlib.data.dataset.handler import DataHandlerLP

    config = Alpha360DL.get_feature_config()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = Alpha360(
            instruments="cn_etf",
            start_time=SEGMENTS["train"][0],
            end_time=SEGMENTS["train"][1],
            fit_start_time=SEGMENTS["train"][0],
            fit_end_time=SEGMENTS["train"][1],
        )
    unexpected = [
        w
        for w in caught
        if not (
            issubclass(w.category, RuntimeWarning)
            # 全 NaN 列在 numpy 里成对报警：求均值一条、求方差一条，同源于"库里没有 $vwap"。
            and any(msg in str(w.message) for msg in ("Mean of empty slice", "Degrees of freedom <= 0 for slice."))
        )
    ]
    assert not unexpected, (
        f"Alpha360 构建发出了预期外的警告：{[(w.category.__name__, str(w.message)) for w in unexpected]}"
    )
    assert caught, "预期的空切片警告没有出现：库里的 vwap 缺口可能已经变了，这条裁定要重评"

    feat = h.fetch(col_set="feature")
    _ops, names = config
    assert feat.shape[1] == len(names) and feat.shape[1] > 300
    # 原始帧（未经 Fillna）里死掉的必须正好是那批 VWAP 列
    dead, explained, have = _dead_and_explained(h.fetch(col_set="feature", data_key=DataHandlerLP.DK_R), config)
    assert "vwap" not in have
    assert dead == explained, f"dead columns {sorted(dead)} not explained by missing fields {sorted(explained)}"
    assert dead == {n for n in names if n.startswith("VWAP")} and len(dead) == 60, (
        f"Alpha360 的死列不是 60 个 VWAP 列：{sorted(dead)[:5]} ({len(dead)} 列)"
    )
    assert np.isfinite(feat.drop(columns=sorted(dead)).to_numpy(dtype="float64")).all(), (
        "Alpha360 可用列里仍有 inf/NaN：默认 ProcessInf+Fillna 之后不该有非有限值"
    )
