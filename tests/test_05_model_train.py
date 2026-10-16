"""Layer 5: LightGBM training and prediction sanity.

The `trained_model` / `predictions` fixtures live in tests/conftest.py (session scope):
pytest fixtures are not visible across test modules and Task 9 consumes `predictions`, and a
second LightGBM fit in one session would double the (minutes-long) cost. Every index access
here uses level *names* — the handler/dataset frame is datetime-first (Task 6).

Inherited state (task-5-7 report + tests/test_03_alpha_factors.py): the feature matrix handed
to LightGBM is **raw, unnormalized and NaN-bearing** — `VWAP0` is entirely NaN. LightGBM
absorbs missing values natively, so nothing here imputes, fills or normalizes anything;
`test_fit_consumes_the_learn_frame_of_the_shared_dataset` checks that it really stays raw.

Measured outcome of this layer (full numbers in task-8-report.md) — read it before "fixing"
anything below: early stopping keeps **iteration 2**, so the shipped model is 2 trees over 32
ever-split features, and it emits only **30 distinct score values**. Pooled Spearman IC is
-0.005445 (valid) and +0.004031 (test), both inside one null standard deviation
(0.006050 / 0.003346); the mean *daily* rank IC is +0.0009 / +0.0017 with t = 0.06 / 0.16.
The model layer therefore runs end to end with honest diagnostics, but carries **no detectable
ranking power** on this 8-month training window. That is this layer's finding, not a defect to
hide: the "beats random" guard below is the brief's single seeded null draw — a weak criterion
(a pure-noise score clears it about half the time), so nobody may read it as significance,
and no number here may be loosened to manufacture one.
"""

import numpy as np
import pandas as pd
import pytest

from config.settings import SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")

PREDICTED_SEGMENTS = ("valid", "test")

# 下面这些数字全部实测于本库 + 这套模型参数。它们变了就说明库、切分或模型层变了，
# 正确反应是重出报告重审结论，而不是把常量改成新数字。
PREPARED_ROWS = {"train": 38313, "valid": 27388, "test": 90008}
PREDICTED_INSTRUMENTS = {"valid": 335, "test": 626}
LABEL_NAN_ROWS = {"train": 53, "valid": 71, "test": 688}
LEARN_TRAIN_ROWS = 38260
LEARN_TRAIN_FEATURE_NAN_CELLS = 95206
# test 段末两个交易日：660 行、330 只标的有价，且这 330 只标签全可算
# （与 test_04 钉住的 660 = 330 x 2 是同一个事实）。
TAIL_ROWS = 660
TAIL_LIVE_INSTRUMENTS = 330
IC_BAND_UPPER = 0.5


def _score_label_join(dataset, predictions, seg):
    """预测与本段**原始**标签（DK_I，没走过 learn 链的 CSZScoreNorm）按索引对齐后去掉 NaN。"""
    labels = dataset.prepare(seg, col_set="label").iloc[:, 0]
    scores = predictions[seg].reindex(labels.index)
    missing = int(scores.isna().sum())
    assert missing == 0, f"{seg}: 预测与标签索引不齐，reindex 后出现 {missing} 个空位"
    joined = pd.concat([scores.rename("score"), labels.rename("label")], axis=1).dropna()
    n_label_nan = int(labels.isna().sum())
    assert len(joined) == len(labels) - n_label_nan, (
        f"{seg}: joined {len(joined)} 行 != 标签 {len(labels)} 行 - NaN {n_label_nan} 行"
    )
    return joined


def _rank_ic(joined):
    """( pooled Spearman IC, 同一批标签上的随机基线 |IC| )；基线用固定种子，可复现。"""
    from scipy.stats import spearmanr

    ic = spearmanr(joined["score"], joined["label"]).statistic
    rng = np.random.default_rng(7)
    baseline = abs(spearmanr(rng.normal(size=len(joined)), joined["label"]).statistic)
    return ic, baseline


def test_fit_consumes_the_learn_frame_of_the_shared_dataset(dataset):
    """fit 的输入契约（`qlib/contrib/model/gbdt.py:37` 取 DK_L）必须是真的。

    标签侧：DropnaLabel 砍掉 NaN 标签行、CSZScoreNorm 把它标准化到方差 1；
    特征侧：158 列仍是**原始、未归一化、含 NaN** 的那一帧，VWAP0 整列 NaN 没被填掉。
    这条既挡住"偷偷把喂给模型的数据洗过一遍"，也是 LightGBM 能原生吸收缺失值的前提。
    """
    from qlib.data.dataset.handler import DataHandlerLP

    n_prepared = len(dataset.prepare("train", col_set="feature"))
    assert n_prepared == PREPARED_ROWS["train"], f"train 段 prepared 行数变了：{n_prepared}"

    learn = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    x, y = learn["feature"], learn["label"].iloc[:, 0]
    assert len(learn) == x.shape[0] == LEARN_TRAIN_ROWS, f"learn 帧行数不是 {LEARN_TRAIN_ROWS}：{len(learn)}"
    assert n_prepared - len(learn) == LABEL_NAN_ROWS["train"], (
        f"DropnaLabel 砍掉的行数是 {n_prepared - len(learn)}，不是实测的 {LABEL_NAN_ROWS['train']}"
    )
    assert x.shape[1] == 158, f"特征列数不是 158：{x.shape[1]}"
    assert y.notna().all(), "DropnaLabel 之后 learn 标签里仍有 NaN"
    # 原始两日前向收益的 std 是 1e-2 量级；≈1 才说明 CSZScoreNorm 真的作用在标签上。
    assert abs(y.std() - 1.0) < 0.01, f"learn 标签 std={y.std()} 没被横截面标准化"
    n_nan = int(x.isna().sum().sum())
    assert n_nan == LEARN_TRAIN_FEATURE_NAN_CELLS, (
        f"learn 特征帧的 NaN 单元格数从 {LEARN_TRAIN_FEATURE_NAN_CELLS} 变成 {n_nan}：交付形态被改了"
    )
    assert x["VWAP0"].isna().all(), "VWAP0 整列 NaN 是交付形态，模型层不该把它填掉"
    assert x["KMID"].notna().any(), "learn 特征帧全是 NaN"


def test_fit_produced_a_real_boosting_model(trained_model):
    """fit 必须真的产出一个 booster，而且早停确实生效（不是跑满 1000 轮的空转）。"""
    booster = trained_model.model
    assert booster is not None, "LGBModel.fit 之后 model 仍是 None"
    assert booster.best_iteration >= 1, f"早停迭代号异常：{booster.best_iteration}"
    n_trees = booster.num_trees()
    assert 1 <= n_trees < trained_model.num_boost_round, f"树数 {n_trees} 说明早停没起作用"


def test_prediction_row_count_matches_input(dataset, predictions):
    """打分不能悄悄丢行：每段预测行数必须等于该段 prepared 的特征样本行数，且索引逐行相同。"""
    for seg in PREDICTED_SEGMENTS:
        feat = dataset.prepare(seg, col_set="feature")
        n_in, n_out = len(feat), len(predictions[seg])
        assert n_in == PREPARED_ROWS[seg], f"{seg}: prepared 行数从 {PREPARED_ROWS[seg]} 变成了 {n_in}"
        assert n_out == n_in, f"{seg}: prediction count {n_out} != input sample count {n_in}"
        assert predictions[seg].index.equals(feat.index), f"{seg}: 预测索引与特征帧索引不一致"
        assert list(predictions[seg].index.names) == ["datetime", "instrument"], predictions[seg].index.names
        n_inst = predictions[seg].index.get_level_values("instrument").nunique()
        assert n_inst == PREDICTED_INSTRUMENTS[seg], f"{seg}: 打分标的数 {n_inst} != 预期 {PREDICTED_INSTRUMENTS[seg]}"


def test_predictions_are_not_a_constant_column(predictions):
    """常数预测 = 假收敛，模型其实什么都没学到。"""
    for seg, s in predictions.items():
        n_nan = int(s.isna().sum())
        assert n_nan == 0, f"{seg}: 预测里出现了 {n_nan} 个 NaN"
        assert np.isfinite(s).all(), f"{seg}: 预测里出现了 inf"
        assert s.nunique() > 1, f"{seg}: 预测只有一个取值（{s.iloc[0]}）"
        std = float(s.std())
        assert std > 1e-9, f"{seg}: model predicts a constant (std={std})"


def test_valid_ic_is_in_a_plausible_band_and_beats_random(dataset, predictions):
    """IC 双向判据。下限排除纯噪声；上限同样重要 —— IC 异常大几乎一定是数据泄漏
    （用了未来函数），那种情况该报警而不是算通过。

    valid 同时是 fit 的早停验证集（`gbdt.py:36-37` 把 train/valid 一起塞进 valid_sets），
    它的数字天然偏乐观，样本外裁定只能看下一条。
    """
    joined = _score_label_join(dataset, predictions, "valid")
    assert len(joined) == PREPARED_ROWS["valid"] - LABEL_NAN_ROWS["valid"], f"valid joined 行数变了：{len(joined)}"
    ic, baseline = _rank_ic(joined)
    assert not np.isnan(ic), "valid IC 是 NaN（分数退化成常数？）"
    assert 0 < abs(ic) < IC_BAND_UPPER, (
        f"valid |IC|={abs(ic):.6f} 不在 (0, {IC_BAND_UPPER}) 内 —— 上限意味着前视泄漏，下限意味着零信息"
    )
    assert abs(ic) > baseline, f"model IC {ic:.4f} no better than random {baseline:.4f}"


def test_test_ic_is_in_a_plausible_band_and_beats_random(dataset, predictions):
    """同一条双向判据必须作用在真正样本外的 test 段上 —— Task 9 打分的正是这一段。"""
    joined = _score_label_join(dataset, predictions, "test")
    assert len(joined) == PREPARED_ROWS["test"] - LABEL_NAN_ROWS["test"], f"test joined 行数变了：{len(joined)}"
    ic, baseline = _rank_ic(joined)
    assert not np.isnan(ic), "test IC 是 NaN（分数退化成常数？）"
    assert 0 < abs(ic) < IC_BAND_UPPER, (
        f"test |IC|={abs(ic):.6f} 不在 (0, {IC_BAND_UPPER}) 内 —— 上限意味着前视泄漏，下限意味着零信息"
    )
    assert abs(ic) > baseline, f"model IC {ic:.4f} no better than random {baseline:.4f}"


def test_only_the_tail_330_instruments_are_tradable_at_the_end_of_the_test_segment(dataset, predictions):
    """test 段打分宇宙是 626 只，但段末只剩 330 只有价 —— 90008 行不等于"626 只跑到底"。

    库里 287 只标的的 bar 停在 2026-08-03、8 只停在 08-14、1 只停在 09-14（task-5-7 报告
    "顺手量到的新事实"），所以段末两个交易日只有 330 只标的有行、有可算标签，与 test_04
    钉住的 660 行 = 330 x 2 是同一件事。Task 9 的换手只能发生在这批标的里。
    """
    labels = dataset.prepare("test", col_set="label").iloc[:, 0]
    days = sorted(set(labels.index.get_level_values("datetime")))
    tail_days = days[-2:]
    assert tail_days[-1] == pd.Timestamp(SEGMENTS["test"][1]), f"test 段末日不是 SEGMENTS['test'][1]：{tail_days[-1]}"

    tail = labels[labels.index.get_level_values("datetime").isin(tail_days)]
    tail_instruments = tail.index.get_level_values("instrument").nunique()
    assert len(tail) == 2 * tail_instruments, f"段末两天不是每只标的一条：{len(tail)} 行 / {tail_instruments} 只"
    live = tail.dropna()
    assert len(live) == TAIL_ROWS, (
        f"段末两天非 NaN 标签行数从 {TAIL_ROWS} 变成了 {len(live)}（那两天有行的标的是 {tail_instruments} 只）"
    )
    n_live = live.index.get_level_values("instrument").nunique()
    assert n_live == TAIL_LIVE_INSTRUMENTS, (
        f"段末两天有可用标签的标的数从 {TAIL_LIVE_INSTRUMENTS} 变成了 {n_live}"
        f"（test 段打分宇宙是 {PREDICTED_INSTRUMENTS['test']} 只，两者不相等）"
    )
