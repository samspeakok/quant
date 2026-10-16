import os
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


def _use_temporary_mlflow_store(path: Path) -> None:
    """把 qlib 的实验跟踪指到一个临时的本地目录，绕开 mlflow 3.x 对文件后端的硬拒绝。

    这不是"顺手加的配置"，是 `LGBModel.fit` 在本环境里**跑不完**的直接后果，实测链条：
    `qlib/contrib/model/gbdt.py:90` 在 fit 结尾无条件 `R.log_metrics(...)`
    -> `qlib/workflow/__init__.py:590` -> `qlib/workflow/expm.py:327` `MlflowClient(tracking_uri=None)`
    -> mlflow 回落默认 `./mlruns` -> `mlflow/store/tracking/file_store.py:224`（开关判定）/`:225`（抛出）
    `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode ...`
    （装的 mlflow 是 3.17.1，官方只留了一个开关 `MLFLOW_ALLOW_FILE_STORE=true`）。
    因为文件后端被拒，qlib 连实验都建不出来，fit 会带着**已经训好的树**整段抛异常
    （RED 里 [2] 轮就已 early stop，异常发生在之后的 metrics 落盘上）。

    开关按官方要求打开；目录走 **qlib 自己的** `R.set_uri()`（写进 `C.exp_manager["kwargs"]["uri"]`），
    不能用 `mlflow.set_tracking_uri()` —— 实测 `qlib/workflow/recorder.py:337` 在起 recorder 时会
    `mlflow.set_tracking_uri(self.uri)`，而 `self.uri` 就是 ExpManager 的 default_uri（今天默认 None），
    于是 mlflow 的全局值被 qlib 原样覆盖回 `./mlruns`（第一版这么写过，仓库根因此多出 mlruns/ 实验目录，
    已实测确认并改掉）。跟踪目录用 pytest 的 session 临时目录：模型层要验的是 fit/predict，
    跟踪日志是纯副产物，既不往仓库根扔东西，也不占 `artifacts/`（那是研究产出目录）。
    """
    from qlib.workflow import R

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    R.set_uri(str(path))


@pytest.fixture(scope="session")
def trained_model(dataset, tmp_path_factory):
    """Task 8 产出、Task 9 复用的 LightGBM 模型；简报参数表 = qlib baseline YAML 里的那套。

    fit 用的是 learn 帧（`gbdt.py:37` `data_key=DataHandlerLP.DK_L`）：Alpha158 的 learn 链是
    DropnaLabel + CSZScoreNorm(fields_group="label")，所以**标签**被按日横截面标准化
    （实测 std 0.9979）、标签为 NaN 的行被丢掉（实测 train 段 38313 行里有 53 行 NaN 标签，
    进 LightGBM 的是 38260 行），而 158 列**特征仍是原始、未归一化、含 NaN** 的那一帧
    （实测 95206 个 NaN 单元格，VWAP0 整列仍全 NaN），由 LightGBM 自己原生吸收缺失值。
    早停的验证集是 **valid**（`_prepare_data` 把 train/valid 都塞进 valid_sets），所以 valid
    的数字天然偏乐观，样本外只能看 test。实测早停停在第 2 轮 -> 模型只有 2 棵树。
    """
    from qlib.contrib.model.gbdt import LGBModel

    _use_temporary_mlflow_store(tmp_path_factory.mktemp("mlflow-runs"))
    model = LGBModel(
        loss="mse",
        colsample_bytree=0.8879,
        learning_rate=0.2,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=210,
        num_threads=8,
    )
    model.fit(dataset)
    # fit 通过 R.log_metrics 留下了一个活跃实验/运行；`qlib/workflow/utils.py:28` 注册了
    # `atexit.register(R.end_exp, ...)`，于是 qlib 会在**解释器退出时**（pytest 早已关掉输出捕获）
    # 再写一条 timer 日志，表现为 session 结束后一串 "--- Logging error --- ValueError:
    # I/O operation on closed file"。这里在 fixture 内当场把实验收尾，atexit 钩子就不会再触发
    # （qlib 自己的注释："will not take effect if experiment ends"）。
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    R.end_exp(Recorder.STATUS_FI)
    return model


@pytest.fixture(scope="session")
def predictions(trained_model, dataset):
    """Task 8 交付给 Task 9 的打分帧：`{段名: pd.Series}`，段为 valid / test。

    **真实返回类型是 pd.Series，不是 DataFrame**：`qlib/contrib/model/gbdt.py:92-96`
    `predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test")` 里面是
    `pd.Series(self.model.predict(x_test.values), index=x_test.index)`，没有列名。
    所以简报里那行 `predictions[seg].iloc[:, 0]` 会当场 IndexError，Task 9 应直接
    `signal=predictions["test"]`（qlib 自己接受 Series 信号：`qlib/backtest/signal.py:102`
    `isinstance(obj, (pd.DataFrame, pd.Series))`），要帧就 `.to_frame("score")`，别再写 `.iloc[:, 0]`。
    本 fixture 不替它包装成 DataFrame —— 保持模型层的原生产出，索引是 `['datetime','instrument']`。

    Task 9 必须知道的实测形态：2 棵树的模型在 27388 / 90008 行上只给出 **30 个不同的分值**
    （值域 -0.0243..0.0727，因为训练目标是按日横截面标准化过的标签，分数就在 z 分尺度上）。
    TopkDropoutStrategy(topk=30) 因此是在**大量并列**里挑标的，选中谁很大程度上取决于
    qlib 的排序稳定性，而不是模型意见 —— 回测结论要把这件事写清楚。
    """
    return {seg: trained_model.predict(dataset, segment=seg) for seg in ("valid", "test")}
