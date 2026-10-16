"""Layer 6: matching engine, account, costs and the report frame.

Consumes the session fixtures `qlib_ready` / `predictions` from tests/conftest.py (Task 8 delivers
one **single-column DataFrame named `score`** per segment, MultiIndex level names
`["datetime", "instrument"]`); produces `make_strategy(signal)` and the `backtest_result` fixture
for Task 10. Every index access below uses level *names*, never positions.

SHAPES MEASURED ON THIS STORE (two of them cost the controller a failed pre-flight run each; do not
re-derive them):
  * `backtest(...)` returns `(portfolio, indicator)`, both keyed `"1day"`.
  * `portfolio["1day"][0]` is the report frame with columns
    `account return total_turnover turnover total_cost cost value cash bench` —
    the benchmark column is **`bench`**, NOT `benchmark` (the shipped plan text KeyError'd on that).
  * `portfolio["1day"][1]` is a **dict keyed by date**, values are `Position` objects. Iterating the
    dict yields its *keys*, so `[len(p) for p in positions]` measures dates instead of portfolios and
    the "holdings <= topk" criterion would pass forever. The holdings count here comes from
    `.values()` plus `get_stock_list()`, and `test_positions_never_exceed_topk` also pins the value
    type so that trap cannot be re-introduced.
  * `Position` has **no `__len__`** (measured: `hasattr(Position(...), "__len__") is False`), so the
    brief's `hasattr(p, "__len__")` filter would drop every day and `assert held` would fail for the
    wrong reason.

INHERITED STATE THAT DECIDES WHAT THIS CURVE MEANS (tests/test_05_model_train.py + conftest,
both re-measured on the **seeded** fit): the trained model is a **1-round** LightGBM whose
test-segment scores take only **5 distinct values** across 626 instruments, while `topk=30` —
and not one trading day of the 181 is an exception. The portfolio is therefore chosen out of
enormous ties: measured, the tier straddling the 30th slot holds up to **249 tied instruments**
(candidates 30..269), and on **107 of 181 days the entire top-30 slate is decided by ties rather
than by ranking**. The controller ruled (with measurement: reordering tied rows produced
byte-identical reports) that tie-break *sensitivity* is not measurable here, because
`TopkDropoutStrategy` resolves ties deterministically. What is measurable, and what this layer
therefore reports as numbers instead of prose, is (R1) how degenerate the ranking is and
(R2) whether the whole thing reproduces.

`test_backtest_is_reproducible` (R2) is the falsifiable half: two independent full-segment backtests
on the same signal must produce identical report frames. If qlib's internal ordering ever stops
being deterministic, that test goes red — which is exactly what we would need to know. Nothing here
asserts a tie-break *spread*, because the measurement says the spread is zero by construction.

THE NUMBERS ARE DISCLOSED, NOT JUDGED. This layer verifies *Qlib mechanics*. The pinned measurements
are the equity curve this store actually produces; they are **not** alpha evidence (the model layer
measured pooled |IC| = 0.0041 on the test segment, inside its permutation null, with
`beats_null_p95=False`). The strongest evidence for that stance is this file's own history: the
curve is pinned at **cum_return = +12.98%** (benchmark -5.88%, excess +18.86%) under the seeded
model, while the *unseeded* run of the same data and same config produced ≈ 0.00%. A 13-percentage-point
swing from changing the RNG, with nothing else touched, is the opposite of a stable signal — which is
why nothing here gates on "the strategy beats random". The controller's random anchor on this universe
with the same topk/n_drop/costs over the same first 20 trading days gives cum_return -0.96% / total
cost 0.39% / mean daily turnover 21.7%; `test_signal_is_not_indistinguishable_from_random_selection`
measures the model's first-20-days numbers against that anchor and **prints the gap without gating on
it** — a no-signal model landing near the anchor is the expected outcome, not a failure, and gating on
it would re-create the coin-flip gate the model layer removed.

Runtime (measured, and it corrected a wrong estimate): the **whole 181-day backtest takes ~8s**
(the pre-flight's "60s per 20 days" had counted store/exchange warm-up, not per-day cost), so this
file costs 241s end to end — three backtests plus one shared model fit. Do not budget 10 minutes
per segment.
"""

import os

import numpy as np
import pandas as pd
import pytest

from config.settings import BENCHMARK, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")

TOPK = 30
N_DROP = 3
ACCOUNT = 100_000_000

# 换手上的成本上界完全由这三个费率决定，所以"每日成本在合理范围内"这条判据能从配置直接推出来，
# 不用凭感觉扣一个 5% 的大帽子（简报那条保留了，但它比下面这条松两个量级）。
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
MIN_COST = 5
LIMIT_THRESHOLD = 0.095

# 实测列名，按列表相等钉住（qlib 改名会当场红，而不是悄悄绕过"有没有 return 列"这种弱判据）。
REPORT_COLUMNS = [
    "account",
    "return",
    "total_turnover",
    "turnover",
    "total_cost",
    "cost",
    "value",
    "cash",
    "bench",
]

# --- 以下数字全部实测于本库 + 本模型信号（**定种子 seed=66** 那一次 fit 的打分）+ 这套回测配置。
# --- 它们变了就说明信号、配置或库变了，正确反应是重出报告重审结论，而不是把常量改成新数字
# --- 或把带宽放宽。带宽与定种子前完全一致（未放宽），只有被测量随配置变更重取。
TEST_TRADING_DAYS = 181  # len(D.calendar(SEGMENTS["test"]))；报表实测 2026-01-05..2026-09-30
RANDOM_ANCHOR = {  # 控制器的随机排序锚点：同 universe / topk / n_drop / 费率 / 同样前 20 个交易日
    "cum_return": -0.009648,
    "sum_cost": 0.003937,
    "mean_turnover": 0.217417,
}
ANCHOR_DAYS = 20

DISTINCT_SCORES_CEILING = 30  # == TOPK；与 test_05 的 DISTINCT_SCORES_CEILING 同一事实（实测 5）
DAILY_DISTINCT_SCORES = 5  # 181 个交易日里每一天的不同分值数，实测恒定
MIN_DAILY_CANDIDATES = 330  # test 段候选最少的那一天（段末只剩 330 只有价）
TOP_TIE_MEDIAN = 36  # 典型那一天共享最高分的标的数（实测 min 6 / max 129）
TOP_TIE_BAND = 5
CUT_TIE_MEDIAN = 47  # 典型那一天并列在 topk 分数线上的标的数（实测 min 7 / max 249）
CUT_TIE_BAND = 10
SLOTS_FROM_TIE_MEDIAN = 30  # topk 席位里由"同分挑谁"而非排名决定的个数（实测 min 1 / max 30）
SLOTS_FROM_TIE_BAND = 3
# 整份 topk 名单都由并列决定的天数（slots_from_tie >= TOPK）：181 天里 107 天。
FULLY_TIED_DAYS = 107
FULLY_TIED_DAYS_BAND = 3
HOLDINGS_AT_CUT_MEDIAN = 15  # 真交出去的持仓里分数正好压在决策日分数线上的只数（mean 11.65 / max 28）
HOLDINGS_AT_CUT_BAND = 3
HOLDINGS_AT_CUT_MEAN = 11.65
HOLDINGS_AT_CUT_MEAN_BAND = 1.0

MAX_HOLDINGS = 33  # == TOPK + N_DROP；持仓数实测 distinct=[0, 30, 31, 33]（0 是段首第一天还没建仓）
MEAN_HOLDINGS = 30.5028
MEAN_HOLDINGS_BAND = 1.0
CUM_RETURN = 0.12982849
CUM_RETURN_BAND = 2e-4
MEAN_TURNOVER = 0.15319714
MEAN_TURNOVER_BAND = 2e-3
SUM_COST = 0.02724323
SUM_COST_BAND = 5e-4
MAX_DAILY_COST = 0.00047459
MAX_DAILY_COST_BAND = 5e-4
COST_RATE_SLACK = 1e-6
N_RETURN_NAN = 0


def make_executor():
    return {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }


def make_exchange_kwargs():
    return {
        "limit_threshold": LIMIT_THRESHOLD,
        "deal_price": "close",
        "open_cost": OPEN_COST,
        "close_cost": CLOSE_COST,
        "min_cost": MIN_COST,
    }


def make_strategy(signal):
    """Task 9 交给 Task 10 的装配点：`backtest(strategy=...)` 直接吃这个 dict（init_instance_by_config）。"""
    return {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {"signal": signal, "topk": TOPK, "n_drop": N_DROP},
    }


# T9_DEV_WINDOW_DAYS 只给开发期用（未设/0 = 整段 test，即交付形态）。
_DEV_WINDOW = int(os.environ.get("T9_DEV_WINDOW_DAYS", "0"))


def _window():
    """回测窗口 = SEGMENTS["test"] 整段。"""
    if not _DEV_WINDOW:
        return SEGMENTS["test"][0], SEGMENTS["test"][1]
    from qlib.data import D

    days = [d.strftime("%Y-%m-%d") for d in D.calendar(start_time=SEGMENTS["test"][0], end_time=SEGMENTS["test"][1])]
    assert _DEV_WINDOW <= len(days), f"T9_DEV_WINDOW_DAYS={_DEV_WINDOW} 超过 test 段的 {len(days)} 个交易日"
    return days[0], days[_DEV_WINDOW - 1]


def _run(predictions):
    from qlib.backtest import backtest

    start, end = _window()
    return backtest(
        start_time=start,
        end_time=end,
        strategy=make_strategy(predictions["test"]),
        executor=make_executor(),
        benchmark=BENCHMARK,
        account=ACCOUNT,
        exchange_kwargs=make_exchange_kwargs(),
    )


@pytest.fixture(scope="session")
def backtest_result(predictions):
    """整段 test 的一次真回测 —— 本层判据全部复用这一份，只跑一次（实测约 10 分钟）。"""
    return _run(predictions)


def _unwrap(portfolio):
    """`(report, positions)`：外层键是执行频率（实测只有 `"1day"` 一个）。"""
    assert portfolio, "empty portfolio metrics"
    keys = list(portfolio)
    assert keys == ["1day"], f"portfolio 的键实测应为 ['1day']，实得 {keys}"
    return portfolio["1day"]


def _as_timestamp(value):
    """positions 的键与报表索引都归一到 Timestamp（层级与类型只按名字取，不猜位置）。"""
    return value if isinstance(value, pd.Timestamp) else pd.Timestamp(str(value))


def _scores_by_day(signal):
    """`{Timestamp: Series(instrument -> score)}`，只按层级名取。"""
    return {_as_timestamp(day): frame["score"].droplevel("datetime") for day, frame in signal.groupby(level="datetime")}


def _tie_profile(vals):
    """一天横截面分数 -> topk 个席位里有多少是并列分决定的。

    cut = 第 TOPK 高的分数（进/出的分数线）。strictly_above 是分数**严格高于**线的标的数，它们靠排名
    必进；剩下 `TOPK - strictly_above` 个席位只能从 `n_at_cut` 个同分标的里挑，挑谁与分数无关。
    """
    vals = np.asarray(vals, dtype=float)
    assert vals.size >= TOPK, f"当天候选只有 {vals.size} 个，少于 topk={TOPK}，分数线无从定义"
    cut = float(np.sort(vals)[::-1][TOPK - 1])
    strictly_above = int((vals > cut).sum())
    return {
        "cut": cut,
        "n_at_cut": int((vals == cut).sum()),
        "strictly_above": strictly_above,
        "slots_from_tie": TOPK - strictly_above,
        "n_top_tie": int((vals == vals.max()).sum()),
        "n_distinct": int(np.unique(vals).size),
    }


def _median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def _cum_return(frame):
    return float((1 + frame["return"].fillna(0.0)).prod() - 1)


def _cum_bench(frame):
    return float((1 + frame["bench"].fillna(0.0)).prod() - 1)


def test_report_frame_has_return_and_risk_columns(backtest_result):
    report, _positions = _unwrap(backtest_result[0])
    nan_counts = {col: int(report[col].isna().sum()) for col in REPORT_COLUMNS}
    print(f"[T9] report rows={len(report)} columns={list(report.columns)} "
          f"index.name={report.index.name!r} index.type={type(report.index).__name__} nan={nan_counts}")
    assert list(report.columns) == REPORT_COLUMNS, (
        f"回测报表列名与实测不符：实得 {list(report.columns)}，预期 {REPORT_COLUMNS}"
    )
    for col in ("return", "bench", "cost", "turnover"):
        assert col in report.columns, f"missing {col}: {list(report.columns)}"
    assert report.index.name == "datetime", f"报表索引名实测是 datetime，实得 {report.index.name!r}"
    assert len(report) == TEST_TRADING_DAYS, (
        f"test 段回测覆盖 {len(report)} 个交易日，实测应为 {TEST_TRADING_DAYS}（SEGMENTS['test'] 整段）"
    )
    # return 列有空位时，下面所有 cum 乘积都会被 fillna(0) 静默抹平，所以先把它钉住。
    assert nan_counts["return"] == N_RETURN_NAN, (
        f"报表 return 列有 {nan_counts['return']} 个 NaN（实测 {N_RETURN_NAN}），"
        f"累计收益会被 fillna(0) 静默低估：其余列的 NaN 数 = {nan_counts}"
    )


def test_backtest_covers_the_whole_test_segment(backtest_result):
    """判据不许靠缩短窗口变便宜：报表必须铺满 SEGMENTS['test'] 里的**每一个交易日**。"""
    from qlib.data import D

    report, _positions = _unwrap(backtest_result[0])
    first, last = _as_timestamp(report.index[0]), _as_timestamp(report.index[-1])
    print(f"[T9] window actually backtested: {first.date()} .. {last.date()} ({len(report)} days)")
    # SEGMENTS 的两个端点是**日历日**而不是交易日：2026-01-01 是元旦休市，实测报表首日为
    # 2026-01-05。所以正确判据是"等于该区间内的首个/末个交易日"，覆盖率仍需全量成立，
    # 缩短窗口依旧会被 len(report) 那条抓住。
    seg_days = D.calendar(start_time=SEGMENTS["test"][0], end_time=SEGMENTS["test"][1])
    # D.calendar 返回 numpy 数组，`assert seg_days` 会当场 ValueError: truth value ... ambiguous
    # （本机实测），所以按长度判空。
    assert len(seg_days) > 0, "test 段在 Qlib 日历上没有任何交易日，配置或库出了问题"
    assert first == _as_timestamp(min(seg_days)), (
        f"回测起点 {first} != test 段首个交易日 {min(seg_days)}"
        f"（区间端点 {SEGMENTS['test'][0]} 可能本身是休市日）"
    )
    assert last == _as_timestamp(max(seg_days)), f"回测终点 {last} != test 段末个交易日 {max(seg_days)}"
    assert len(report) == len(seg_days), f"报表 {len(report)} 天 != test 段交易日 {len(seg_days)} 天"


def test_positions_never_exceed_topk(backtest_result):
    _report, positions = _unwrap(backtest_result[0])
    assert isinstance(positions, dict), (
        f"positions 实测是按日期索引的 dict（date -> Position），实得 {type(positions).__name__}"
    )
    # 值必须真是 Position：这条就是"迭代 dict 拿到日期串"那个坑的结构性防线（日期串没有 get_stock_list）。
    assert all(callable(getattr(p, "get_stock_list", None)) for p in positions.values()), (
        "positions 的值不都是 Position 对象 —— 又踩了迭代 dict 的坑，持仓数会变成 len(日期)"
    )
    held = [len(p.get_stock_list()) for p in positions.values()]
    print(f"[T9] holdings/day: n_days={len(held)} max={max(held)} min={min(held)} "
          f"mean={float(np.mean(held)):.4f} distinct={sorted(set(held))}")
    assert held, "no positions recorded"
    # 实测最大持仓 **33 > topk=30**，且 33 == topk + n_drop。这不是 bug 而是 TopkDropout 的
    # 语义：它每天最多卖 n_drop 只、再把仓位补到 topk，因此在"卖不掉（停牌/跌停）+ 继续补仓"
    # 的日子里持仓会短暂超过 topk。原来写 `<= TOPK` 是拿直觉当判据，实测必然失败。
    assert max(held) <= TOPK + N_DROP, f"held {max(held)} names > topk+n_drop={TOPK + N_DROP}"
    # 假日判据自检：若每天持仓数都恰好等于 len(日期字符串)，说明又踩了迭代 dict 的坑
    assert set(held) != {10}, "positions look like date strings, not portfolios"
    assert max(held) == MAX_HOLDINGS, (
        f"最多持仓数从实测的 {MAX_HOLDINGS} 变成 {max(held)}（topk={TOPK}, n_drop={N_DROP}）—— "
        f"要么策略的补仓算式变了，要么涨跌停挡住卖出把持仓顶高了"
    )
    mean_held = float(np.mean(held))
    assert abs(mean_held - MEAN_HOLDINGS) <= MEAN_HOLDINGS_BAND, (
        f"日均持仓从实测 {MEAN_HOLDINGS:.4f} 变成 {mean_held:.4f}（带宽 {MEAN_HOLDINGS_BAND}）"
    )


def test_turnover_and_cost_are_in_a_sane_range(backtest_result):
    """成本上界由配置反推：单日换手 T 的买卖，成本最多 T*max(open,close)_cost*2 + min_cost 的绝对额。"""
    report, _positions = _unwrap(backtest_result[0])
    mean_turnover = float(report["turnover"].mean())
    sum_cost = float(report["cost"].sum())
    max_daily_cost = float(report["cost"].max())
    print(f"[T9] cum_return={_cum_return(report):+.8f} mean_turnover={mean_turnover:.8f} "
          f"max_turnover={float(report['turnover'].max()):.8f} sum_cost={sum_cost:.8f} "
          f"max_daily_cost={max_daily_cost:.8f} total_cost_last={float(report['total_cost'].iloc[-1]):.8f} "
          f"total_turnover_last={float(report['total_turnover'].iloc[-1]):.8f}")
    assert (report["cost"].abs() < 0.05).all(), "daily cost > 5% — cost config or matching is wrong"
    upper = report["turnover"].abs() * max(OPEN_COST, CLOSE_COST) * 2 + MIN_COST / ACCOUNT
    over = report.loc[report["cost"] > upper + COST_RATE_SLACK]
    assert over.empty, (
        f"{len(over)} 天的成本高于费率配置允许的上界（open={OPEN_COST} close={CLOSE_COST} "
        f"min_cost={MIN_COST}）：\n{over[['turnover', 'cost']].head()}"
    )
    # 单位必须分清。qlib report.py:81-84 的字段注释就是权威定义：
    #   turnovers      = "turnover for each trade time"      → 报表 `turnover`（**率**）
    #   costs           = "trade cost rate for each time"    → 报表 `cost`（**率**）
    #   total_turnovers = "total turnover for each trade time" → 报表 `total_turnover`（**金额**）
    #   total_costs     = "total trade cost for each trade time" → 报表 `total_cost`（**金额**）
    # 所以两个 total_* 都是金额、两个非 total 的都是率；`total_cost[-1] == sum(cost)` 这类
    # 累加关系从定义上就不成立（实测末值 3.12e6 / 3.17e9 对累加率 0.027 / 27.7，差六个量级）。
    # 原先这里写过"total_turnover 是按率累计的、与 sum(turnover) 相等"—— 那条是**错的**，
    # 它在这一次运行里当场变红才暴露（3167638507 != 27.728681）。换成下面这条单位自洽的比率判据。
    total_cost_last = float(report["total_cost"].iloc[-1])
    total_turnover_last = float(report["total_turnover"].iloc[-1])
    assert total_cost_last > 1e4, (
        f"total_cost 末值 {total_cost_last} 不像累计金额，成本口径或账户规模变了"
    )
    assert total_turnover_last > 1e8, (
        f"total_turnover 末值 {total_turnover_last} 不像累计成交金额（账户 {ACCOUNT:,}）：口径变了"
    )
    assert (report["total_cost"].diff().dropna() >= -1e-9).all(), "total_cost 非单调：累计金额不该回退"
    assert (report["total_turnover"].diff().dropna() >= -1e-9).all(), "total_turnover 非单调：累计成交额不该回退"
    assert float(report["cost"].abs().max()) < 0.05, "单日 cost 费率 > 5%，与 total_cost 的金额量级混淆"
    assert (report["turnover"] >= 0.0).all(), "出现负换手：撮合或报表符号错了"
    blended = total_cost_last / total_turnover_last
    assert OPEN_COST <= blended <= CLOSE_COST, (
        f"累计成本 / 累计成交额 = {blended:.8f}，落在配置费率 [{OPEN_COST}, {CLOSE_COST}] 之外 —— "
        f"要么成本按金额累计的口径变了，要么成交额与成本对不上"
        f"（total_cost={total_cost_last:.2f}, total_turnover={total_turnover_last:.2f}）"
    )
    assert abs(mean_turnover - MEAN_TURNOVER) <= MEAN_TURNOVER_BAND, (
        f"日均换手从实测 {MEAN_TURNOVER:.8f} 变成 {mean_turnover:.8f}（带宽 {MEAN_TURNOVER_BAND}）"
    )
    assert abs(sum_cost - SUM_COST) <= SUM_COST_BAND, (
        f"累计成本从实测 {SUM_COST:.8f} 变成 {sum_cost:.8f}（带宽 {SUM_COST_BAND}）"
    )
    assert abs(max_daily_cost - MAX_DAILY_COST) <= MAX_DAILY_COST_BAND, (
        f"单日最大成本从实测 {MAX_DAILY_COST:.8f} 变成 {max_daily_cost:.8f}（带宽 {MAX_DAILY_COST_BAND}）"
    )


def test_returns_are_not_identical_to_the_benchmark(backtest_result):
    """完全相等说明回测根本没在做决策。"""
    report, _positions = _unwrap(backtest_result[0])
    diff = (report["return"].fillna(0.0) - report["bench"].fillna(0.0)).abs()
    n_identical = int((report["return"].fillna(0.0) == report["bench"].fillna(0.0)).sum())
    print(f"[T9] max|return-bench|={float(diff.max()):.8f} n_days_identical={n_identical}/{len(report)} "
          f"cum_bench={_cum_bench(report):+.8f}")
    assert float(diff.max()) > 1e-6, "strategy return == benchmark return everywhere: no real trading"
    assert n_identical < len(report), f"每一天都与基准逐位相等（{n_identical}/{len(report)}）：账户没在交易"


def test_equity_curve_is_pinned_and_reported(backtest_result):
    """把这条曲线本身钉住：它是"机制跑通"的证据，不是 alpha 证据（理由见模块 docstring）。"""
    report, _positions = _unwrap(backtest_result[0])
    cum, cum_bench = _cum_return(report), _cum_bench(report)
    print(f"[T9] 整段 test：days={len(report)} cum_return={cum:+.8f} cum_bench={cum_bench:+.8f} "
          f"excess={cum - cum_bench:+.8f} value_last={float(report['value'].iloc[-1]):.2f} "
          f"account_last={float(report['account'].iloc[-1]):.2f} cash_frac_min="
          f"{float((report['cash'] / report['account']).min()):.6f}")
    assert abs(cum - CUM_RETURN) <= CUM_RETURN_BAND, (
        f"累计收益从实测 {CUM_RETURN:+.8f} 变成 {cum:+.8f}（带宽 {CUM_RETURN_BAND}）。这里钉的是"
        f"“这个信号在这套装配下的实际曲线”：数值漂了说明信号、配置或库变了，要重出报告重审结论，"
        f"而不是把常量改成新数字。"
    )
    assert abs(cum - cum_bench) > CUM_RETURN_BAND, (
        f"策略累计收益 {cum:+.8f} 与基准累计收益 {cum_bench:+.8f} 落在同一条带宽里 —— 组合实际上没在做决策"
    )


def test_degeneracy_of_the_ranking_is_measured_in_numbers(predictions):
    """R1（1/2）：并列退化度必须是数字。这条只吃信号本身，不需要回测。

    cut = 第 topk 高的分数；slots_from_tie = 有多少席位不是由排名决定、而是从 n_at_cut 个同分标的里
    挑谁决定。定种子后全段只有 **5** 个不同分值而 topk=30，分数线上永远站着一大片同分标的，
    选出来的名单基本不携带排序信息 —— 下面这几个数字就是这条结论的证据。
    """
    scores = predictions["test"]["score"]
    by_day = _scores_by_day(predictions["test"])
    profiles = [_tie_profile(s.to_numpy()) for s in by_day.values()]
    n_cand = np.array([s.size for s in by_day.values()], dtype=float)
    n_distinct = np.array([p["n_distinct"] for p in profiles], dtype=float)
    top_tie = np.array([p["n_top_tie"] for p in profiles], dtype=float)
    cut_tie = np.array([p["n_at_cut"] for p in profiles], dtype=float)
    slots = np.array([p["slots_from_tie"] for p in profiles], dtype=float)
    fully_tied_days = int((slots >= TOPK).sum())
    n_distinct_all = int(scores.nunique())
    print(f"[T9][R1] test 段：rows={len(scores)} 不同分值={n_distinct_all} 打分天数={len(profiles)} | "
          f"候选数/天 min={n_cand.min():.0f} median={_median(n_cand):.0f} max={n_cand.max():.0f} | "
          f"单日不同分值 min={n_distinct.min():.0f} median={_median(n_distinct):.0f} max={n_distinct.max():.0f} | "
          f"共享当日最高分的标的数 median={_median(top_tie):.0f} mean={top_tie.mean():.1f} "
          f"min={top_tie.min():.0f} max={top_tie.max():.0f} | "
          f"并列在分数线上的标的数 median={_median(cut_tie):.0f} min={cut_tie.min():.0f} max={cut_tie.max():.0f} | "
          f"由并列决定的席位 median={_median(slots):.0f} min={slots.min():.0f} max={slots.max():.0f} | "
          f"整份 topk 名单都由并列决定的天数={fully_tied_days}/{len(profiles)}")
    assert n_distinct_all <= DISTINCT_SCORES_CEILING, (
        f"test 段不同分值数 {n_distinct_all} 超过了 topk 上界 {DISTINCT_SCORES_CEILING}（模型层退化事实变了，"
        f"见 tests/test_05_model_train.py::DISTINCT_SCORES_CEILING）"
    )
    assert int(n_cand.min()) == MIN_DAILY_CANDIDATES, (
        f"test 段候选最少那一天的打分标的数从实测 {MIN_DAILY_CANDIDATES} 变成 {int(n_cand.min())}"
    )
    assert n_cand.min() >= TOPK, f"有日子候选不足 topk={TOPK}（最少 {int(n_cand.min())} 只），分数线定义不成立"
    # 退化到"每天只剩 5 档分值"是这条结论最直白的形态：181 天里没有一天例外。
    assert int(n_distinct.min()) == int(n_distinct.max()) == DAILY_DISTINCT_SCORES, (
        f"每日不同分值从实测的恒定 {DAILY_DISTINCT_SCORES} 变成 "
        f"[{int(n_distinct.min())}, {int(n_distinct.max())}]（中位 {_median(n_distinct):.0f}）—— "
        f"打分一旦能切出更多档，本层与模型层的退化结论都要重审"
    )
    assert _median(top_tie) == pytest.approx(TOP_TIE_MEDIAN, abs=TOP_TIE_BAND), (
        f"典型交易日共享最高分的标的数从实测 {TOP_TIE_MEDIAN} 变成 {_median(top_tie)}（带宽 {TOP_TIE_BAND}）"
    )
    assert _median(cut_tie) == pytest.approx(CUT_TIE_MEDIAN, abs=CUT_TIE_BAND), (
        f"典型交易日并列在 topk 分数线上的标的数从实测 {CUT_TIE_MEDIAN} 变成 {_median(cut_tie)}"
        f"（带宽 {CUT_TIE_BAND}）"
    )
    assert _median(slots) == pytest.approx(SLOTS_FROM_TIE_MEDIAN, abs=SLOTS_FROM_TIE_BAND), (
        f"由同分而非排名决定的席位数从实测 {SLOTS_FROM_TIE_MEDIAN} 变成 {_median(slots)}"
        f"（带宽 {SLOTS_FROM_TIE_BAND}）；这个数回到 0 才说明模型给出了可分辨的排序 —— 那是另一个结论，"
        f"要重出报告而不是改这里的常量"
    )
    assert slots.min() > 0, (
        f"存在分数线以下毫无并列的日子（min slots_from_tie={slots.min()}）—— 与“分数只有 "
        f"不同取值不超过 topk={TOPK}”这个退化事实矛盾，重审信号来源"
    )
    # 最直白的一条：有多少天的**整份** topk 名单都不是排名决定的。
    assert abs(fully_tied_days - FULLY_TIED_DAYS) <= FULLY_TIED_DAYS_BAND, (
        f"整份 topk 名单完全由并列决定的天数从实测 {FULLY_TIED_DAYS} 变成 {fully_tied_days}"
        f"/{len(profiles)} 天（带宽 {FULLY_TIED_DAYS_BAND}）。这个数掉到 0 才说明模型给出了"
        f"有意义的排序 —— 那是另一个结论，要重出报告而不是改这里的常量"
    )


def test_delivered_holdings_are_chosen_from_ties(predictions, backtest_result):
    """R1（2/2）：真交出去的持仓里有多少只是同分决定的。

    TopkDropoutStrategy 在第 i 天用的是**第 i-1 天**的分数（`generate_trade_decision` 里
    `get_step_time(trade_step, shift=1)`），所以分数线按决策日算，不是按成交日算。
    """
    _report, positions = _unwrap(backtest_result[0])
    by_day = _scores_by_day(predictions["test"])
    days = sorted(by_day)
    decision_of = {days[i + 1]: days[i] for i in range(len(days) - 1)}
    at_cut = []
    n_all_tie_days = 0
    n_no_decision = 0
    for day, pos in positions.items():
        scores = by_day.get(decision_of.get(_as_timestamp(day)))
        if scores is None:
            n_no_decision += 1
            continue
        held = pos.get_stock_list()
        if not held:
            continue
        prof = _tie_profile(scores.to_numpy())
        n_at = sum(1 for code in held if code in scores.index and float(scores.loc[code]) == prof["cut"])
        at_cut.append(n_at)
        if n_at == len(held):
            n_all_tie_days += 1
    assert at_cut, "没有任何一天同时有持仓和决策日分数，度量没跑起来"
    med = _median(at_cut)
    print(f"[T9][R1] 按决策日分数线统计真持仓：n_days={len(at_cut)} 同分决定的持仓数 "
          f"median={med:.0f} mean={float(np.mean(at_cut)):.2f} min={min(at_cut)} max={max(at_cut)}；"
          f"整份名单全由同分决定的天数={n_all_tie_days}/{len(at_cut)}；段首无决策日可对齐的步数={n_no_decision}")
    assert med == pytest.approx(HOLDINGS_AT_CUT_MEDIAN, abs=HOLDINGS_AT_CUT_BAND), (
        f"由同分（而非排名）决定的持仓数从实测 {HOLDINGS_AT_CUT_MEDIAN} 变成 {med}（带宽 {HOLDINGS_AT_CUT_BAND}）"
    )
    mean_at_cut = float(np.mean(at_cut))
    assert abs(mean_at_cut - HOLDINGS_AT_CUT_MEAN) <= HOLDINGS_AT_CUT_MEAN_BAND, (
        f"同分决定的持仓数均值从实测 {HOLDINGS_AT_CUT_MEAN:.2f} 变成 {mean_at_cut:.2f}"
        f"（带宽 {HOLDINGS_AT_CUT_MEAN_BAND}）"
    )
    assert max(at_cut) > 0, (
        f"没有任何一天存在分数正好压在决策日分数线上的持仓（max={max(at_cut)}）—— "
        f"与“分数线上站着大片同分标的”矛盾，重审信号来源"
    )
    # 记录但不主张"整份名单都由同分决定"：实测 0/180 天。原因是 TopkDropout 会把前一天的仓位
    # 带过来，真持仓是"今天分数线"与"昨天遗留"的混合，所以名单不会**全部**落在那条线上；
    # "整份名单由并列决定"这件事是在名单侧量的（见 FULLY_TIED_DAYS = 107/181 天）。
    print(f"[T9][R1] 名单侧整份由并列决定的天数见 test_degeneracy（{FULLY_TIED_DAYS}/181）；"
          f"持仓侧全由同分决定的天数={n_all_tie_days}/{len(at_cut)}（实测 0，因持仓跨日携带，仅记录）")


def test_signal_is_not_indistinguishable_from_random_selection(backtest_result):
    """与控制器锚点对照：同 universe、同 topk/n_drop/费率、同样前 20 个交易日，
    区别只有分数来自随机排序（同样只有 30 个不同取值）还是来自模型。

    只度量、只上报、不当闸门（理由见模块 docstring）。能钉住的只有"这 20 天里确实发生了交易"。
    """
    report, _positions = _unwrap(backtest_result[0])
    assert len(report) >= ANCHOR_DAYS, f"报表只有 {len(report)} 天，不足锚点的 {ANCHOR_DAYS} 天窗口"
    head = report.iloc[:ANCHOR_DAYS]
    cum, mean_turnover, sum_cost = _cum_return(head), float(head["turnover"].mean()), float(head["cost"].sum())
    verdict = "不高于锚点" if cum <= RANDOM_ANCHOR["cum_return"] + 1e-6 else "高于锚点"
    print(f"[T9][anchor] 前 {ANCHOR_DAYS} 个交易日 模型信号 cum_return={cum:+.6f} "
          f"mean_turnover={mean_turnover:.6f} sum_cost={sum_cost:.6f} ｜ 随机排序锚点 "
          f"cum_return={RANDOM_ANCHOR['cum_return']:+.6f} mean_turnover={RANDOM_ANCHOR['mean_turnover']:.6f} "
          f"sum_cost={RANDOM_ANCHOR['sum_cost']:+.6f} ｜ 差 cum_return={cum - RANDOM_ANCHOR['cum_return']:+.6f} "
          f"mean_turnover={mean_turnover - RANDOM_ANCHOR['mean_turnover']:+.6f} "
          f"sum_cost={sum_cost - RANDOM_ANCHOR['sum_cost']:+.6f}（{verdict}；记录，不断言）")
    assert mean_turnover > 0.0, "前 20 个交易日零换手：策略根本没下单，锚点对照无意义"


def test_backtest_is_reproducible(predictions):
    """R2：同一份信号跑两遍回测，报表必须逐位相同。

    这条可失败，且钉住的正是"并列分数的打破顺序是确定性的"这件事：控制器实测"只改行序结果逐位
    相同"，所以这里不问 tie-break 散度（测不出来），只问同输入两次是否一致。qlib 内部排序一旦不再
    确定，这条当场红 —— 那正是需要知道的。代价：再跑一遍回测（整段实测约 10 分钟）。
    """
    # `backtest()` 返回的是 `(portfolio, indicator)` 二元组，`_unwrap` 吃的是其中的 portfolio
    # —— 少写 `[0]` 会把整个元组传进去，`list(portfolio)` 就变成了 [{'1day': ...}, Indicator]。
    first_report, _ = _unwrap(_run(predictions)[0])
    second_report, _ = _unwrap(_run(predictions)[0])
    pd.testing.assert_frame_equal(first_report, second_report)
    print(f"[T9][R2] 两次回测报表逐位相同：rows={len(first_report)} "
          f"cum_return={_cum_return(first_report):+.8f} 两次 cum 差="
          f"{abs(_cum_return(first_report) - _cum_return(second_report)):.3e}")
