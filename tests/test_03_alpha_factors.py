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

from config.settings import HELDOUT_START, HISTORY_START, QLIB_DIR, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")

# 会**拟合统计量**、并且从 handler 的 fit 窗口取参的三个处理器（qlib 0.9.7 实测接口）。
FIT_WINDOW_STAT_PROCESSORS = ("MinMaxNorm", "ZScoreNorm", "RobustZScoreNorm")
# alpha158_handler 今天唯一携带的处理器：learn 链的两个标签处理器，两者都不带 fit 窗口。
EXPECTED_ALPHA158_PROCESSORS = ["DropnaLabel", "CSZScoreNorm"]
# Task 8 从同一个 fixture 继承的状态，实测于交付帧 (159208 行, 158 列)。
DELIVERED_NAN_CELLS = 306421
DELIVERED_NAN_CELLS_LIVE_COLUMNS = 147213

# 规格原文里逐字抄下来的标签表达式（docs/superpowers/plans/2026-10-15-qlib-capability-verification.md:33
# 与 .superpowers/sdd/.../task-10-brief.md:129 同一条）。A2 的关键：断言的对象必须是**规格字面量**，
# 不能是 handler 自己吐出来的 exprs —— 拿 exprs[0] 和 exprs[0] 比，把标签改成 Ref($close,-1) 也照样绿。
SPEC_LABEL_EXPR = "Ref($close, -2)/Ref($close, -1) - 1"


def _strip_ws(expr: str) -> str:
    """只剥空白，不做别的规范化：`Ref($close,-2)/Ref($close,-1)-1` 与规格写法视为同一条表达式。"""
    return re.sub(r"\s+", "", expr)


def _assert_label_expression_is_the_spec_one(exprs: list[str]) -> None:
    """把标签钉在规格字面量上（逐条、按位置，列数也必须还是 1 条）。"""
    assert [_strip_ws(e) for e in exprs] == [_strip_ws(SPEC_LABEL_EXPR)], (
        f"标签表达式不是规格里那条 {SPEC_LABEL_EXPR!r}，实际是 {exprs}。"
        "换成前看一天（`Ref($close,-1)/$close - 1` 之类）改的是预测目标本身：Task 6 的跨层数值比对、"
        "Task 8 的训练标签、Task 9 的留出尾部口径会一起失去意义。"
    )


def _store_fields(instruments: list[str]) -> set[str]:
    """原始字段清单：库里的 .day.bin 文件名就是 Qlib 能读到的 $field。

    不能拿**一个**标的的目录去推全池结论（审阅 M-3）：这里把这批标的的目录全部 glob 一遍，
    要求字段集完全一致，检查过的标的数与分组结果都出现在失败信息里。只查一个的话，
    哪天导出脚本只给部分标的补上 vwap，"死列都由缺字段解释"这条裁定会照样绿。
    """
    groups: dict[frozenset[str], list[str]] = {}
    for inst in instruments:
        d = QLIB_DIR / "features" / inst.lower()
        fields = frozenset(p.name.split(".")[0] for p in d.glob("*.day.bin"))
        assert fields, f"{inst}: 目录 {d} 里没有任何 .day.bin"
        groups.setdefault(fields, []).append(inst)
    checked = sum(len(v) for v in groups.values())
    assert checked >= 2, f"字段一致性检查只覆盖了 {checked} 个标的，不足以支撑全池结论"
    assert len(groups) == 1, (
        f"库里各标的的原始字段集不一致：检查了 {checked} 个标的，得到 {len(groups)} 种字段集；"
        + " / ".join(f"{sorted(f)} -> {len(v)} 个标的（如 {v[:3]}）" for f, v in groups.items())
    )
    return set(next(iter(groups)))


def _dead_and_explained(feat: pd.DataFrame, config: tuple[list[str], list[str]]) -> tuple[set, set, set]:
    """返回 (全 NaN 的因子列, 由"表达式引用的原始字段不在库里"解释掉的列, 库里的字段清单)。"""
    fields, names = config
    assert not feat.empty, "factor frame came back empty"
    assert len(fields) == len(names) == feat.shape[1], (
        f"declared {len(fields)} expressions / {len(names)} names vs frame {feat.shape[1]} columns"
    )
    have = _store_fields(sorted(set(feat.index.get_level_values("instrument"))))
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
    _assert_label_expression_is_the_spec_one(list(exprs))
    # 数值侧：矩阵里那一列必须等于数据层同一表达式的读数（该表达式的算术已在 Task 5 逐点比对过）
    got = label.iloc[:, 0].dropna()
    inst = got.index.get_level_values("instrument")[0]
    day = got.index.get_level_values("datetime")[0]
    want = D.features([inst], [exprs[0]], start_time=day, end_time=day).iloc[:, 0]
    assert len(want) == 1, f"{inst} {day}: 数据层标签读数应该有 1 行，实际 {len(want)}"
    ref, val = float(want.iloc[0]), float(got.iloc[0])
    assert abs(ref - val) <= 1e-6 * abs(ref) + 1e-9, f"{inst} {day}: 矩阵标签 {val} vs 数据层 {ref}"


def test_the_spec_literal_guard_rejects_the_one_day_label_forms():
    """A2 的可证伪性自证：同一条守卫作用在"前看一天"那几个写法上必须报错。

    不读数据，纯字符串，毫秒级。它证明这条断言不是拿 handler 的话复读一遍：
    handler 只要给出下面任何一条，`test_label_uses_the_declared_two_day_forward_return` 就会红。
    """
    one_day_forms = [
        "Ref($close, -1)/$close - 1",
        "Ref($close,-1)/$close-1",
        "Ref($close, -1)/Ref($close, 0) - 1",
    ]
    for wrong in one_day_forms:
        with pytest.raises(AssertionError, match="标签表达式不是规格里那条"):
            _assert_label_expression_is_the_spec_one([wrong])
    # 条数也必须钉住：多塞一条标签（哪怕第一条还对）同样不许过。
    with pytest.raises(AssertionError, match="标签表达式不是规格里那条"):
        _assert_label_expression_is_the_spec_one([SPEC_LABEL_EXPR, "Ref($close, -3)/Ref($close, -1) - 1"])
    # 空白差异不该误报：这是本条守卫唯一允许的写法自由度。
    _assert_label_expression_is_the_spec_one(["Ref($close,-2)/Ref($close,-1)-1"])


def _assert_fit_windows_are_train_bounded(processors) -> int:
    """I1 守卫本体：作用在**真实处理器对象**上，任何带 fit 窗口的处理器都必须落在 train 段内。

    返回携带 fit 窗口的处理器个数 —— 调用方必须显式裁定这个数应该是几，否则"一个都没有"
    会让整段 walk 变成不会失败的死循环（审阅 I1 的原始问题就在这）。

    上游的原文警告（qlib/data/dataset/processor.py:198-199 MinMaxNorm、:232-233 ZScoreNorm、
    :274-275 RobustZScoreNorm，三处逐字相同）：

        # NOTE: correctly set the `fit_start_time` and `fit_end_time` is very important !!!
        # `fit_end_time` **must not** include any information from the test data!!!
    """
    train_end = pd.Timestamp(SEGMENTS["train"][1])
    history_start = pd.Timestamp(HISTORY_START)
    checked = 0
    for proc in processors:
        if not (hasattr(proc, "fit_start_time") or hasattr(proc, "fit_end_time")):
            continue
        checked += 1
        name = type(proc).__name__
        start, end = proc.fit_start_time, proc.fit_end_time
        assert start is not None and end is not None, f"{name}: fit 窗口不完整 start={start!r} end={end!r}"
        assert pd.Timestamp(end) <= train_end, (
            f"{name}.fit_end_time={end} 越过 train 段末 {SEGMENTS['train'][1]}：归一化统计量将吃到 "
            "valid/test 的信息，下游 IC 与回测数字全部作废"
        )
        assert pd.Timestamp(start) >= history_start, (
            f"{name}.fit_start_time={start} 早于 HISTORY_START {HISTORY_START}：窗口伸进库里不存在数据的区间"
        )
    return checked


def test_no_processor_may_fit_on_data_beyond_the_train_window(alpha158_handler, alpha158_kwargs):
    """I1：把"不泄漏"这条性质钉在真实对象上，而不是钉在 fixture 的源码文本上。

    今天这个 fixture 是安全的，但它安全的原因一个都没被断言过：`fit_end_time` 取 train 段末、
    `Alpha158` 的 `infer_processors` 默认为空（`qlib/contrib/data/handler.py:105`）。
    两者任一被改掉（把 fit 窗口推到 valid 段末、或往 infer 链里塞一个归一化），
    71 条测试会全部照绿，而每个 IC/回测数字都失去意义。三条腿分别钉住：

    1. **对象 walk**：handler 携带的每个处理器，只要暴露 fit_start_time/fit_end_time，
       窗口必须 train 有界（走 `_assert_fit_windows_are_train_bounded`）。
    2. **infer 链必须还是空的**：这就是"fixture 保持 raw"的承诺本身。会拟合统计量的三类处理器
       —— MinMaxNorm / ZScoreNorm / RobustZScoreNorm —— 一旦被加进 infer_processors，
       `check_transform_proc`（`qlib/contrib/data/handler.py:12-34`）会把 fixture 声明的
       `fit_start_time`/`fit_end_time` **原样注入**它们，上游警告随之生效：
       "`fit_end_time` **must not** include any information from the test data!!!"
       加处理器的人必须同时把这条守卫改成"窗口有界的处理器 + 更新交付形态断言"，不能悄悄过。
    3. **装配路径本身**：把 fixture 真正声明的那份窗口喂进生产的 `check_transform_proc`，
       造出"加归一化时就会拿到的那个处理器对象"再过一遍守卫 —— 这一腿才是
       "有人把 fit_end_time 推到 valid 段末"唯一的暴露点：此刻没有任何处理器消费那个窗口，
       光靠对象 walk 看不见它。
    """
    from qlib.contrib.data.handler import check_transform_proc
    from qlib.data.dataset import processor as processor_module
    from qlib.data.dataset.processor import Processor
    from qlib.utils import init_instance_by_config

    procs = alpha158_handler.get_all_processors()
    # 腿 1：walk 真实对象 —— 现在一个带窗口的处理器都不该有，但这条分支必须能报错
    # （报错分支的活性由 test_the_fit_window_guard_rejects_windows_reaching_valid_test_or_heldout 自证）。
    checked = _assert_fit_windows_are_train_bounded(procs)
    assert checked == 0, (
        f"对象 walk 发现 {checked} 个带 fit 窗口的处理器：共享 fixture 已不再 raw，"
        "需要归一化的模型应自建 handler，而不是改这个 fixture"
    )
    assert [type(p).__name__ for p in procs] == EXPECTED_ALPHA158_PROCESSORS, (
        f"alpha158_handler 携带的处理器已经不是 {EXPECTED_ALPHA158_PROCESSORS}："
        f"{[type(p).__name__ for p in procs]}。共享 fixture 的交付形态被改了，"
        "必须同时更新 test_handler_delivers_raw_nan_bearing_matrix 与 conftest 里的文档。"
    )
    # 腿 2：infer 链必须还是空的 —— "fixture 保持 raw"这条承诺本身。
    assert alpha158_handler.infer_processors == [], (
        f"infer_processors 不再为空：{[type(p).__name__ for p in alpha158_handler.infer_processors]}。"
        "Task 8 拿到的矩阵不再是原始值，跨层比对与原始值断言全部失效。"
    )
    assert alpha158_handler.shared_processors == [], (
        f"shared_processors 不再为空：{[type(p).__name__ for p in alpha158_handler.shared_processors]}"
    )
    # 腿 3：装配路径本身 —— 把 fixture 真正声明的那份窗口喂进生产的 check_transform_proc。
    assert alpha158_kwargs["fit_end_time"] is not None and alpha158_kwargs["fit_start_time"] is not None, (
        "fixture 没有声明 fit 窗口：check_transform_proc 会在注入归一化时报错，但这条守卫也失去了意义"
    )
    injected = []
    for cls in FIT_WINDOW_STAT_PROCESSORS:
        cfg = check_transform_proc(
            [{"class": cls, "kwargs": {}}],
            alpha158_kwargs["fit_start_time"],
            alpha158_kwargs["fit_end_time"],
        )[0]
        proc = init_instance_by_config(cfg, processor_module, accept_types=Processor)
        assert hasattr(proc, "fit_end_time") and hasattr(proc, "fit_start_time"), (
            f"{cls} 不再暴露 fit_start_time/fit_end_time：qlib 接口变了，守卫的 walk 要按新接口重写"
        )
        injected.append(proc)
    assert _assert_fit_windows_are_train_bounded(injected) == len(FIT_WINDOW_STAT_PROCESSORS), (
        "生产装配路径没有把 fixture 声明的窗口注入这三个处理器：这条腿已经看不到真实继承值"
    )


def test_the_fit_window_guard_rejects_windows_reaching_valid_test_or_heldout():
    """I1 的可证伪性自证：同一个守卫作用在窗口越界的处理器上必须报错。

    上一条测试里的 walk 今天一个处理器都不检查（`checked == 0`），所以它的分支必须是
    可证明活着的，否则那条测试只是恒真。这里只构造处理器对象（不读数据，毫秒级），
    两条路径各证一次：直接构造、以及走生产的 `check_transform_proc` 装配。
    """
    from qlib.contrib.data.handler import check_transform_proc
    from qlib.data.dataset import processor as processor_module
    from qlib.data.dataset.processor import Processor
    from qlib.utils import init_instance_by_config

    bad_ends = [SEGMENTS["valid"][1], SEGMENTS["test"][1], HELDOUT_START]
    for cls in FIT_WINDOW_STAT_PROCESSORS:
        for end in bad_ends:
            proc = init_instance_by_config(
                {"class": cls, "kwargs": {"fit_start_time": HISTORY_START, "fit_end_time": end}},
                processor_module,
                accept_types=Processor,
            )
            with pytest.raises(AssertionError, match="越过 train 段末"):
                _assert_fit_windows_are_train_bounded([proc])

            # fixture 的 fit_end_time 一旦被推到 valid 段末，被注入归一化的就是这份越界窗口。
            cfg = check_transform_proc([{"class": cls, "kwargs": {}}], HISTORY_START, end)[0]
            wired = init_instance_by_config(cfg, processor_module, accept_types=Processor)
            assert wired.fit_end_time == end, f"{cls}: check_transform_proc 没有注入窗口，这条腿形同虚设"
            with pytest.raises(AssertionError, match="越过 train 段末"):
                _assert_fit_windows_are_train_bounded([wired])

    # 起点越界同样要报错：窗口整体往前挪出 HISTORY_START 时，只查 fit_end_time 看不见这条。
    early_start = init_instance_by_config(
        {
            "class": "RobustZScoreNorm",
            "kwargs": {"fit_start_time": "2020-01-01", "fit_end_time": SEGMENTS["train"][1]},
        },
        processor_module,
        accept_types=Processor,
    )
    with pytest.raises(AssertionError, match="早于 HISTORY_START"):
        _assert_fit_windows_are_train_bounded([early_start])


def test_handler_delivers_raw_nan_bearing_matrix(alpha158_handler):
    """I2：钉住 Task 8 实际继承的矩阵形态 —— 原始、未归一化、含 NaN，**不是**"已经 Fillna 填成 0"。

    简报/报告里"features 已被 Fillna 填 0"那句描述的是 Alpha360 的默认处理器链
    （ProcessInf/ZScoreNorm/Fillna，`qlib/contrib/data/handler.py:41-45`）；这个 fixture 的
    `infer_processors` 是空的（`Alpha158` 默认 `[]`，`handler.py:105`），所以交付的是 raw 帧本身：
    NaN 单元格 306421 个、159208 行**每行至少一个 NaN**（其中 `VWAP0` 一列整列 NaN —— 源字段
    `$vwap` 在本库里不存在），另有 147213 个 NaN 落在可用列上（滚动窗口的预热期与停牌/缺行）。
    LightGBM 一类能原生吸收缺失值的模型可以直接吃；不能吸收的模型必须自建 handler
    （train 有界的 infer_processors），别回头改这个共享 fixture。

    这条测试的存在意义是"谁把这个 fixture 归一化/填零了，就得先在这里解释"：infer 帧一旦不等于
    raw 帧，或 NaN 数量被抹平，Task 6 的跨层原始值比对与 Task 8 的原始值假设会一起失效。
    """
    from qlib.data import D
    from qlib.data.dataset.handler import DataHandlerLP

    infer = alpha158_handler.fetch(col_set="feature")
    raw = alpha158_handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    assert infer.shape == (159208, 158), f"交付帧形状变了：{infer.shape}"
    assert infer.equals(raw), (
        "infer 帧不再逐元素等于 raw 帧：有人给这个共享 fixture 加了 infer_processors（归一化/Fillna）。"
        "Task 8 继承的矩阵不再是原始值 —— 需要归一化的模型请自建 handler。"
    )

    dead = sorted(infer.columns[infer.isna().all()].tolist())
    assert dead == ["VWAP0"], f"整列 NaN 的列不再是 VWAP0 一列：{dead}"
    nan_cells = int(infer.isna().sum().sum())
    assert nan_cells == DELIVERED_NAN_CELLS, (
        f"NaN 单元格总数 {nan_cells} != 实测交付值 {DELIVERED_NAN_CELLS}："
        "缺失值被填掉了（或库里 vwap 缺口变了），Task 8 的缺失值前提要重评"
    )
    live_nan_cells = int(infer.drop(columns=dead).isna().sum().sum())
    assert live_nan_cells == DELIVERED_NAN_CELLS_LIVE_COLUMNS, (
        f"可用列里的 NaN 单元格 {live_nan_cells} != 实测交付值 {DELIVERED_NAN_CELLS_LIVE_COLUMNS}"
    )
    rows_with_nan = int(infer.isna().any(axis=1).sum())
    assert rows_with_nan == len(infer), (
        f"{rows_with_nan}/{len(infer)} 行含 NaN：交付帧不再是『每一行都带缺失值』的原始形态"
    )

    # 原始值本身也必须在：取 test 段最后一天那一行，价格衍生列仍等于数据层的原始读数除以原始收盘价。
    inst = infer.index.get_level_values("instrument")[-1]
    day = infer.index.get_level_values("datetime")[-1]
    layer = D.features([inst], ["$open", "$close"], start_time=day, end_time=day)
    assert len(layer) == 1, f"{inst} {day}: 数据层应该有 1 行，实际 {len(layer)}"
    got = float(infer.loc[(day, inst), "OPEN0"])
    want = float(layer["$open"].iloc[0]) / float(layer["$close"].iloc[0])
    assert abs(got - want) <= 1e-6 * abs(want) + 1e-9, (
        f"OPEN0 {inst} {day}: 因子矩阵 {got} vs 数据层原始值 {want} —— 帧已被归一化"
    )


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
