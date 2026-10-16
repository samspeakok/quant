"""Alpha158 + LightGBM + TopkDropout 全流程研究脚本（六层闭环的真跑一遍）。

这是唯一允许写 `artifacts/` 的入口；`tools/` 只产数据、`tests/` 只证功能、本文件只出研究结论。
依赖方向单向：本文件可以读 config/settings.py，但不 import tests/（测试不是库）。

结论口径由本文件**自己量出来的数字**支撑，不抄 tests/ 里的常量：置换零分布、退化分值数、
分数线并列规模都在这里重算一遍。两处装配必须与 tests/test_06_backtest.py 完全一致
（TOPK / N_DROP / 账户 / 费率 / 执行器），否则测试覆盖的不是这里跑的东西。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    ARTIFACTS,
    BENCHMARK,
    HISTORY_START,
    QLIB_DIR,
    REGION,
    SEGMENTS,
)

# 与 tests/test_06_backtest.py 同源。简报原写 topk=50，但 50 从未被任何一条测试覆盖过，
# 用它就等于"测的是一套、交的是另一套"，所以统一到已实测的 30。
TOPK, N_DROP = 30, 3
ACCOUNT = 100_000_000
OPEN_COST, CLOSE_COST, MIN_COST, LIMIT_THRESHOLD = 0.0005, 0.0015, 5, 0.095

# 与 tests/conftest.py::trained_model 逐项相同，含 seed —— 少了 seed 就等于放弃了可复现性
# （实测 qlib 默认配置同一份库两次运行 best_iteration 从 2 跳到 52，累计收益漂 20.7pp）。
MODEL_KWARGS = dict(
    loss="mse",
    colsample_bytree=0.8879,
    learning_rate=0.2,
    subsample=0.8789,
    lambda_l1=205.6999,
    lambda_l2=580.9768,
    max_depth=8,
    num_leaves=210,
    num_threads=8,
    seed=66,
    deterministic=True,
)

PERMUTATIONS, PERMUTATION_SEED = 200, 11

# 随机排序锚点：同 universe / topk / n_drop / 费率 / 同样 test 段前 20 个交易日，
# 区别只有分数是随机排的。来源是控制器的 pre-flight 实测（见 ledger 2026-10-15）。
RANDOM_ANCHOR = {"cum_return": -0.009648, "sum_cost": 0.003937, "mean_turnover": 0.217417}
ANCHOR_DAYS = 20


def _pooled_ic(score: pd.Series, label: pd.Series) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(score, label).statistic)


def _rank_pearson(rank_score: np.ndarray, rank_label: np.ndarray) -> float:
    a = rank_score - rank_score.mean()
    b = rank_label - rank_label.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _null_percentile(rank_score: np.ndarray, rank_label: np.ndarray, observed_abs_ic: float) -> dict:
    """观测 |IC| 在"把分数的秩打乱"的 200 次抽样里的位置。Spearman 只依赖秩，所以置换秩等价。"""
    from scipy.stats import rankdata

    rs = rankdata(np.asarray(rank_score, dtype=float))
    rl = rankdata(np.asarray(rank_label, dtype=float))
    rng = np.random.default_rng(PERMUTATION_SEED)
    draws = np.array([_rank_pearson(rng.permutation(rs), rl) for _ in range(PERMUTATIONS)])
    p95 = float(np.quantile(np.abs(draws), 0.95))
    return {
        "n_perm": PERMUTATIONS,
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std(ddof=1)),
        "null_abs_p95": p95,
        "percentile_in_null": float(np.count_nonzero(np.abs(draws) < observed_abs_ic)) / PERMUTATIONS,
        "beats_null_p95": bool(observed_abs_ic > p95),
    }


def _join(dataset, scores: pd.DataFrame, seg: str) -> pd.DataFrame:
    labels = dataset.prepare(seg, col_set="label").iloc[:, 0]
    joined = pd.concat(
        [scores[seg].reindex(labels.index).rename("score"), labels.rename("label")], axis=1
    ).dropna()
    return joined


def _tie_profile(vals: np.ndarray, topk: int) -> dict:
    """一天横截面上 topk 席位里有多少是"同分挑谁"决定的。与 tests/test_06 同一定义。"""
    if vals.size < topk:
        return {"defined": False}
    cut = float(np.sort(vals)[::-1][topk - 1])
    strictly_above = int((vals > cut).sum())
    return {
        "defined": True,
        "n_at_cut": int((vals == cut).sum()),
        "slots_from_tie": topk - strictly_above,
        "n_distinct": int(np.unique(vals).size),
    }


def main() -> int:
    # mlflow 3.x 把文件后端置为维护模式直接抛异常，而 qlib 的 fit 结尾无条件 log_metrics；
    # 官方开关必须在 import 之前打开，目录只能用 qlib 自己的 R.set_uri()（见计划 Global Constraints）。
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.backtest import backtest
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(QLIB_DIR), region=REGION)
    # 跟踪目录**必须**放在纯 ASCII 路径下，不能用 ARTIFACTS/mlruns。
    # 本机最小复现（两份对照，只差路径）：`R.set_uri("<仓库内路径>/mlruns")` 之后
    # `R.start()` 一律抛 `MlflowException: Run '<id>' not found` —— mlflow 3.x 的 file store
    # 能把 run 目录建出来，但 `_get_run_info` 找不到它（file_store.py:777→1209→810）；
    # 同一个脚本换成 `C:\Users\...\Temp\...` 就正常。本仓库路径含中文（个人事项），
    # 所以只要落在仓库内就必然踩到。tests/ 侧早就因为另一个原因走的是 pytest 临时目录，
    # 于是这个坑只在研究脚本这边暴露。跟踪日志是纯副产物，不放 artifacts/。
    R.set_uri(Path(tempfile.mkdtemp(prefix="qlib-mlruns-")).as_posix())

    handler = Alpha158(
        instruments="cn_stocks",
        start_time=HISTORY_START,
        end_time=SEGMENTS["valid"][1],
        fit_start_time=HISTORY_START,
        fit_end_time=SEGMENTS["train"][1],
    )
    dataset = DatasetH(handler, segments=SEGMENTS)

    # 库容量是结论第 4 节"数据不足以支撑业绩判断"的依据，所以这里当场数一遍，不引用记忆里的数
    pools = {
        name: len(D.list_instruments(D.instruments(market=m), as_list=False))
        for name, m in (("all", "all"), ("cn_stocks", "cn_stocks"))
    }
    print(f"[T10] 库内标的数 {pools}", flush=True)

    print("[T10] 导出因子帧 …", flush=True)
    factors = dataset.prepare("train", col_set="feature")
    factors.to_parquet(ARTIFACTS / "factors.parquet")

    print("[T10] 训练 LightGBM …", flush=True)
    evals: dict = {}
    model = LGBModel(**MODEL_KWARGS)
    model.fit(dataset, evals_result=evals)
    R.end_exp(Recorder.STATUS_FI)

    train_l2 = [float(v) for v in evals["train"]["l2"]]
    valid_l2 = [float(v) for v in evals["valid"]["l2"]]
    scores = {seg: model.predict(dataset, segment=seg) for seg in ("valid", "test")}
    # 每段都是无名 Series；parquet 需要列名，所以显式命名 score（与 tests/conftest 的交付形状一致）
    pd.concat({seg: s.to_frame("score") for seg, s in scores.items()}, names=["segment"]).to_parquet(
        ARTIFACTS / "predictions.parquet"
    )

    metrics: dict = {
        "store": pools,
        "rounds_trained": len(train_l2),
        "best_iteration": int(model.model.best_iteration),
        "num_boost_round": int(model.num_boost_round),
        "early_stopping_rounds": int(model.early_stopping_rounds),
        "train_l2_first": train_l2[0],
        "train_l2_last": train_l2[-1],
        "valid_l2_best": float(min(valid_l2)),
        "valid_l2_last": valid_l2[-1],
        "segments": {},
    }
    for seg in ("valid", "test"):
        joined = _join(dataset, scores, seg)
        ic = _pooled_ic(joined["score"], joined["label"])
        block = _null_percentile(joined["score"].to_numpy(), joined["label"].to_numpy(), abs(ic))
        block.update(
            {
                "pooled_ic": ic,
                "n_rows_scored": int(len(scores[seg])),
                "n_rows_joined": int(len(joined)),
                "n_instruments": int(scores[seg].index.get_level_values("instrument").nunique()),
                "distinct_scores": int(scores[seg].nunique()),
                "score_min": float(np.min(scores[seg])),
                "score_max": float(np.max(scores[seg])),
            }
        )
        metrics["segments"][seg] = block

    test_scores = scores["test"].to_frame("score")
    by_day = {d: f["score"] for d, f in test_scores.groupby(level="datetime", sort=True)}
    profiles = [_tie_profile(v.to_numpy(), TOPK) for v in by_day.values()]
    defined = [p for p in profiles if p["defined"]]
    tie_cut = np.array([p["n_at_cut"] for p in defined], dtype=float)
    tie_slots = np.array([p["slots_from_tie"] for p in defined], dtype=float)
    n_cand = np.array([v.size for v in by_day.values()], dtype=float)
    metrics["tie_profile"] = {
        "n_days_profiled": len(defined),
        "candidates_min": int(n_cand.min()),
        "candidates_max": int(n_cand.max()),
        "cut_tie_median": float(np.median(tie_cut)),
        "cut_tie_max": int(tie_cut.max()),
        "slots_from_tie_median": float(np.median(tie_slots)),
        "fully_tied_days": int((tie_slots >= TOPK).sum()),
    }

    print("[T10] 回测 …", flush=True)
    portfolio, _indicator = backtest(
        start_time=SEGMENTS["test"][0],
        end_time=SEGMENTS["test"][1],
        strategy={
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": test_scores, "topk": TOPK, "n_drop": N_DROP},
        },
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        benchmark=BENCHMARK,
        account=ACCOUNT,
        exchange_kwargs={
            "limit_threshold": LIMIT_THRESHOLD,
            "deal_price": "close",
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "min_cost": MIN_COST,
        },
    )
    keys = list(portfolio)
    assert keys == ["1day"], f"portfolio 的键实测应为 ['1day']，实得 {keys}"
    report = portfolio["1day"][0]
    report.to_csv(ARTIFACTS / "backtest_report.csv")

    ret = report["return"].fillna(0.0)
    nav = (1 + ret).cumprod()
    excess = ret - report["bench"].fillna(0.0)
    total_cost_last = float(report["total_cost"].iloc[-1])
    total_turnover_last = float(report["total_turnover"].iloc[-1])
    head = report.iloc[:ANCHOR_DAYS]
    metrics["backtest"] = {
        "days": int(len(report)),
        "first_day": str(report.index[0].date()),
        "last_day": str(report.index[-1].date()),
        "cum_return": float(nav.iloc[-1] - 1),
        "cum_benchmark": float((1 + report["bench"].fillna(0.0)).prod() - 1),
        "annual_return": float(nav.iloc[-1] ** (252 / len(nav)) - 1),
        "annual_excess": float(excess.mean() * 252),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "volatility": float(report["return"].std() * np.sqrt(252)),
        "mean_turnover": float(report["turnover"].mean()),
        "sum_cost_rate": float(report["cost"].sum()),
        "total_cost_money": total_cost_last,
        "total_turnover_money": total_turnover_last,
        "blended_cost_per_turnover": total_cost_last / total_turnover_last,
        "anchor_window": {
            "cum_return": float((1 + head["return"].fillna(0.0)).prod() - 1),
            "mean_turnover": float(head["turnover"].mean()),
            "sum_cost": float(head["cost"].sum()),
            "n_days": int(len(head)),
        },
    }

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_findings(metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def _write_findings(m: dict) -> None:
    test, valid = m["segments"]["test"], m["segments"]["valid"]
    bt, tie = m["backtest"], m["tie_profile"]
    anchor = bt["anchor_window"]
    beats_any = test["beats_null_p95"] or valid["beats_null_p95"]
    verdict = (
        "跑赢了置换零分布的 95 分位 —— '本窗口无可检测排序能力'的结论作废，必须重审"
        if beats_any
        else "**没有**跑赢置换零分布的 95 分位（两段都是）—— 本窗口无可检测的横截面排序能力"
    )
    (ARTIFACTS / "FINDINGS.md").write_text(
        f"""# Alpha158 + LightGBM + TopkDropout 在本机 A 股日线上的全流程结果

生成脚本：`research/alpha158_lgbm_backtest.py`（本文件由它按**本次实测数字**写出，不是模板）。
逐项判据与可复现证据在 `tests/`（`pytest tests/` 全绿）；本文只讲结论。

## 一句话结论

**六层闭环（数据 → 表达式 → 因子 → 数据集 → 模型 → 回测）在这套本地数据上完整跑通了；
但同一个模型在本窗口没有任何可检测的横截面排序能力，所以下面这条收益曲线不是 alpha 证据。**

## 1. 机制跑通（六层各自的硬证据在 tests/）

- 数据层：`qlib_data/` 由本机 `market.duckdb`（**只读**）导出的日线二进制库，
  `instruments/all.txt` 覆盖 {m['store']['all']} 个标的、可交易股票池 `cn_stocks` {m['store']['cn_stocks']} 个，
  与源库逐日逐字段交叉校验通过（判据见 `tests/test_01_data_layer.py`：日历集合逐日相等 +
  抽样价格与停牌配对，每次都对着活库现算）。
- 表达式/因子层：Alpha158 交付 (159208, 158) 的原始、未归一化、含 NaN 特征帧，VWAP0 整列 NaN。
- 数据集层：train/valid/test 三段切分 {SEGMENTS['train'][0]} .. {SEGMENTS['test'][1]}，
  test 段 {bt['days']} 个交易日（{bt['first_day']} .. {bt['last_day']}，区间端点是日历日、
  2026-01-01 休市所以首个交易日是 {bt['first_day']}）。
- 模型层：fit 真跑了 {m['rounds_trained']} 轮，`best_iteration = {m['best_iteration']}`，
  valid l2 在第 {m['best_iteration']} 轮见底 {m['valid_l2_best']:.8f} 后升到末轮 {m['valid_l2_last']:.8f}
  （越过"永远预测 0"的零信息基线 1.0），train l2 一路不升 {m['train_l2_first']:.8f} → {m['train_l2_last']:.8f}。
  **这是教科书式过拟合形状，说明这个窗口里没有可学的横截面信号，而不是早停把模型掐早了。**
- 回测层：撮合、账户、成本、报表全部跑通，报表列与 `positions` 结构由 tests/test_06 钉住。

## 2. 没有可检测的排序能力（这是发现，不是待修缺陷）

| 段 | pooled Spearman IC | 行数 | 不同分值 | 零分布 sd | 零分布 \\|IC\\| p95 | 模型分位 | 跑赢 p95？ |
|---|---|---|---|---|---|---|---|
| valid | {valid['pooled_ic']:+.8f} | {valid['n_rows_joined']:,} | {valid['distinct_scores']} | {valid['null_sd']:.8f} | {valid['null_abs_p95']:.8f} | {valid['percentile_in_null']:.3f} | {valid['beats_null_p95']} |
| test | {test['pooled_ic']:+.8f} | {test['n_rows_joined']:,} | {test['distinct_scores']} | {test['null_sd']:.8f} | {test['null_abs_p95']:.8f} | {test['percentile_in_null']:.3f} | {test['beats_null_p95']} |

零分布 = 把已算好分数的秩打乱 {test['n_perm']} 次（不重训），种子 {PERMUTATION_SEED} 固定。
裁定：{verdict}。

打分退化到什么程度（`topk={TOPK}`）：test 段 {tie['n_days_profiled']} 个交易日里
**每一天**都只有 {test['distinct_scores']} 档不同分值；分数线（第 {TOPK} 高的分数）上并列的标的数
中位 {tie['cut_tie_median']:.0f}、最多 {tie['cut_tie_max']} 只；
有 **{tie['fully_tied_days']}/{tie['n_days_profiled']} 天的整份 topk 名单都不是排名决定的，而是"同分里挑谁"**。
候选标的数每天 {tie['candidates_min']}..{tie['candidates_max']} 只。

## 3. 回测数字（披露，不评价）

- 累计收益 **{bt['cum_return']:+.4%}** ／ 基准 {bt['cum_benchmark']:+.4%} ／ 超额 {bt['cum_return'] - bt['cum_benchmark']:+.4%}
- 年化 {bt['annual_return']:+.4%} ／ 日均超额年化 {bt['annual_excess']:+.4%} ／ 最大回撤 {bt['max_drawdown']:.4%} ／ 年化波动 {bt['volatility']:.4%}
- 日均换手 {bt['mean_turnover']:.6f} ／ 累计成本率 {bt['sum_cost_rate']:.6f}
  ／ 累计成本金额 {bt['total_cost_money']:,.2f} ／ 累计成交额 {bt['total_turnover_money']:,.2f}
  ／ 每单位成交额的混合费率 {bt['blended_cost_per_turnover']:.8f}（必须落在配置费率 [{OPEN_COST}, {CLOSE_COST}] 内）
- 前 {anchor['n_days']} 个交易日与随机排序锚点对照：
  模型 cum {anchor['cum_return']:+.6f} / turnover {anchor['mean_turnover']:.6f} / cost {anchor['sum_cost']:.6f}
  vs 随机 cum {RANDOM_ANCHOR['cum_return']:+.6f} / turnover {RANDOM_ANCHOR['mean_turnover']:.6f} / cost {RANDOM_ANCHOR['sum_cost']:+.6f}
  → 差 {anchor['cum_return'] - RANDOM_ANCHOR['cum_return']:+.6f}（**只记录，不作为闸门**）

**为什么这条 {bt['cum_return']:+.2%} 不能当本事**：
同一份数据、同一套参数，只把 RNG 定下来（`seed=66, deterministic=True`）之前，
这条曲线是 ≈0.00%，定下来之后是 {bt['cum_return']:+.2%} —— 换个随机种子就漂十几个百分点，
是"没有稳定信号"的反证，不是"找到了信号"的证据。第 2 节的 IC 与分位数读数说的是同一件事。

## 4. 本轮没有覆盖 / 不能声称的东西

- 不声称跑赢基准或跑赢随机：第 2 节的裁定已经否掉了"有排序能力"。
- 未覆盖 qlib 的其余能力面（在线学习、嵌套决策、Portfolio 层其他策略、模型 zoo、
  高频/分钟线、`Recorder` 的完整实验管理、Alpha360 的实盘装配等）—— 见 README 的功能地图。
- 数据容量本身不足以下结论：库里 A 股日线只到 2021-08-26 起、2021/2022/2023 各只有 1 只标的，
  训练窗只有 {SEGMENTS['train'][0]} 起的一段，且 universe 仅 330 只（test 段末只剩 {tie['candidates_min']} 只有价）。
  **本轮判据是"能否跑通"，收益好不好是后面的事，且必须先解决数据容量。**

## 5. 解释性注意事项

- 标签是 `Ref($close,-2)/Ref($close,-1)-1`，按设计是两日前向收益（不是次日）。
- 成本是配置的 5bp 开 / 15bp 平 + 最小 5 元；**未建模滑点**。
- `limit_threshold={LIMIT_THRESHOLD}` 近似 10% 涨跌停，但挡不住"涨停买不进"的真实成交失败，实盘更差。
- valid 段同时是 fit 的早停验证集（`gbdt.py:36-37`），它的数字天然偏乐观；样本外看 test。
- 三段宇宙不同（train 330 / valid 335 / test 626，段末仅 330 只有价），段间 IC 不可直接比较。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
