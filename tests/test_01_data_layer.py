"""Layer 1: does qlib read back the exact numbers that are in the source database?

from_qlib_code below is a DELIBERATE duplicate of tools/build_qlib_data.py's mapping.
If we reused the exporter's function, a reversed mapping would be wrong on both sides
and this whole cross-check would silently validate itself.
"""

from __future__ import annotations

import random
from functools import lru_cache

import pandas as pd
import pytest

from config.settings import BENCHMARK, HISTORY_START, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


def from_qlib_code(qlib_code: str) -> str:
    return f"{qlib_code[2:]}.{qlib_code[:2]}"


def test_calendar_contains_no_foreign_trading_days(db):
    """CN 日历里的每一个交易日都必须在源库的 CN 数据中出现。
    非 CN 标的混入会带来幽灵交易日，让所有 A 股在该日全为 NaN。"""
    from qlib.data import D

    # 必须先留**有序列表**再派生 set：layer-1 判据明文要求"日历非空且严格递增"，而上一版直接
    # 写成集合推导式，把重复日/乱序全吸收了，cal == src 照样通过。Qlib 不会替我们清理：
    # qlib/data/data.py:675 是 `[pd.Timestamp(x) for x in backend_obj]`（不 np.unique、不 sort），
    # :174 的索引表是 `{x: i for i, x in enumerate(...)}`（重复日静默保留最后一个下标）。
    cal_list = list(D.calendar())
    assert cal_list, "Qlib 日历为空：calendars/day.txt 读不出任何交易日"
    assert cal_list == sorted(set(cal_list)), (
        f"day.txt 必须非空、无重复日且严格递增：{len(cal_list)} 项，"
        f"{len(cal_list) - len(set(cal_list))} 个重复，"
        f"乱序位 {[i for i in range(1, len(cal_list)) if cal_list[i] <= cal_list[i - 1]][:5]}"
    )
    cal = {d.strftime("%Y-%m-%d") for d in cal_list}
    src = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT strftime(CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE), '%Y-%m-%d') FROM bars "
            "WHERE frequency='1d' AND adjust='qfq' AND market='CN' "
            "AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) >= ?",
            [HISTORY_START],
        ).fetchall()
    }
    ghost = cal - src
    assert not ghost, f"{len(ghost)} phantom trading days, e.g. {sorted(ghost)[:5]}"
    assert cal == src, f"calendar does not cover the exported range: missing {sorted(src - cal)[:5]}"


def test_benchmark_loads(db):
    from qlib.data import D

    df = D.features([BENCHMARK], ["$close"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
    # 简报里的阈值是 `notna().sum() > 2000`，写在 progress.md 的 Ruling 2（导出窗口收窄到
    # HISTORY_START=2024-08-05、全库只剩 529 个交易日）之前。收窄前基准指数从 1990 年堆起、
    # 有 8000+ 天，2000 只是"基本覆盖完整"的粗略说法；现在这一段**最多 524 行**，实测
    # `assert (not False and np.int64(524) > 2000)` 恒假 —— 阈值不可能通过，属计划文本遗留。
    # 按 Ruling 2 对 Task 3 日历断言的同一条裁定改成"按窗口精确判定"：基准指数必须在区间内
    # 每一个交易日都有非 NaN 收盘价。这是原意的严格版，不是放宽。
    n_window_days = len(D.calendar(start_time=HISTORY_START, end_time=SEGMENTS["test"][1]))
    n_covered = int(df["$close"].notna().sum())
    assert not df.empty, f"{BENCHMARK} reads back empty"
    assert n_covered == n_window_days, f"{BENCHMARK}: only {n_covered}/{n_window_days} 交易日有收盘价"


@lru_cache(maxsize=1)
def _qlib_spans() -> dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
    """标的名 -> instruments/all.txt 里的 (start, end) 覆盖区间，经 D.list_instruments 读出，
    不自己解析文件 —— 这里要的就是「Qlib 自己认为这个标的哪些天该有数据」。
    缓存一次：D_calendar 会被调用 600+ 次，每次都重建整张 698 项字典要 5 秒以上。"""
    from qlib.data import D

    # market="all" 在这里**只**用于按标的查它自己的起止区间；它绝不是可交易宇宙，
    # 不可复制到任何选股/回测代码里（三个池 cn_stocks/cn_etf/cn_index 绝不合并）。
    return {code: tuple(spans) for code, spans in D.list_instruments(D.instruments(market="all"), as_list=False).items()}


def D_calendar(top_symbol: str) -> list[pd.Timestamp]:
    """该标的的交易日历。

    简报写的是 `D.calendar(top_symbol=top_symbol)`，但 pyqlib 0.9.7 与 vendor/qlib 上游
    都没有这个参数，实测 `TypeError: BaseProvider.calendar() got an unexpected keyword
    argument 'top_symbol'`（三处 calendar 定义签名均为
    `(start_time, end_time, freq, future)`）。旧版 top_symbol 的语义就是「用该标的在
    instruments 文件里的起止切全局日历」，这里按同义实现，取的仍是 Qlib 自己的起止数据。
    """
    from qlib.data import D

    spans = _qlib_spans()[top_symbol]
    return [d for d in D.calendar() if any(s <= d <= e for s, e in spans)]


def _pick_samples(pool_members) -> tuple[list[tuple[str, str]], list[str]]:
    """每个池取 min(20, 池大小) —— cn_index 只有 10 个，写死 20 会让测试直接失败。
    样本在测试体内自取，不用 @pytest.mark.parametrize：parametrize 在收集阶段就要拿到
    fixture 的值，而那时 qlib 还没 init。

    返回 (picked, thin)。thin 是「被抽中但 Qlib 自己认为它不足 10 个交易日」因而整只丢掉
    5 个样本的成员名单：这一档丢弃过去是静默的，抽样分母会无声缩小，现在必须一并上报。"""
    random.seed(20261015)
    picked: list[tuple[str, str]] = []
    thin: list[str] = []
    for pool in ("cn_stocks", "cn_etf", "cn_index"):
        members = pool_members(pool)
        for code in random.sample(members, min(20, len(members))):
            days = D_calendar(code)
            if len(days) < 10:
                thin.append(f"{pool}:{code}={len(days)}d")
                continue
            for d in [days[0], days[-1], days[len(days) // 3], days[len(days) // 2], days[-3]]:
                picked.append((code, d.strftime("%Y-%m-%d")))
    return picked, thin


def test_close_matches_source_database(db, pool_members):
    """核心断言：Qlib 读出的收盘价 == 源库里那一天的收盘价。

    通过条件不是「比对到的都相等」，而是**每一个抽样对都必须走到数值比对**。
    任何一条 skip 路径都等于本层只验了样本的一部分 —— 而那正是这层要防的失败模式，
    所以宁可当场失败并打印原因明细，也不「跳过然后全绿」。"""
    from qlib.data import D

    picked, thin = _pick_samples(pool_members)
    assert picked, "no samples selected"
    assert len(picked) >= 50, f"sampling is broken: only {len(picked)} pairs picked"
    assert not thin, f"{len(thin)} sampled instruments skipped for <10 Qlib trading days: {thin[:10]}"

    checked = 0
    # 控制器 round-2 裁定：**已验证的 NaN 配对**（Qlib=NaN ⟺ 源库 0 行）是一次**比对通过**，
    # 不是 skip。它计入 checked，并单列 nan_verified 供报错原文展示；否则一个合法的停牌日
    # 一旦被抽中，就会在完全正确的数据上把 `checked == len(picked)` 打成假报警。
    nan_verified: list[str] = []
    # 剩下的 skip 档只有空表。每一档都留 offending 样本供最终断言的报错原文使用。
    skipped: dict[str, list[str]] = {"qlib_empty_frame": []}
    for code, day in picked:
        got = D.features([code], ["$close"], start_time=day, end_time=day)["$close"]
        if got.empty:
            # 这个 day 就是从该标的**自己的** Qlib 覆盖区间里取出来的，返回空表本身就是
            # span-vs-data 错位信号（Qlib 说这天该有行却一行都没有），不能咽下去。
            skipped["qlib_empty_frame"].append(f"{code} {day}")
            continue
        want = db.execute(
            "SELECT close FROM bars WHERE symbol=? AND frequency='1d' AND adjust='qfq' AND market='CN' "
            "AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) = CAST(? AS DATE)",
            [from_qlib_code(code), day],
        ).fetchall()
        qlib_v = got.iloc[0]
        # 简报在这里只有一句 `if got.empty: continue`，假定 Qlib 用「不返回该行」表示停牌。
        # 实测相反：停牌日**会**返回一行，值是 NaN（SH601995 的 2025-11-20 实测
        # rows=1 value=nan，而同一天源库 0 行）。按简报原样往下走，`assert want` 会在
        # 完全正确的数据上假报警。停牌日因此单独判定，而且两个方向都钉死：
        #   qlib=NaN  ⟺  源库当天必须有 0 行（否则是丢数据）
        #   qlib=数值 ⟹  源库当天必须有一行且数值相等（原断言，容差 1e-6 未动）
        if pd.isna(qlib_v):
            assert not want, f"{code} {day}: qlib reads NaN but the source database has a bar"
            # 到这里即 Qlib=NaN 且源库 0 行 —— 停牌日双向对齐**验证通过**：计入 checked，
            # 记进可见的 nan_verified，绝不再算 skip。
            nan_verified.append(f"{code} {day}")
            checked += 1
            continue
        assert want, f"{code} {day}: qlib has a bar but the source database has none"
        # (symbol, day) 必须唯一确定一根 bar：实测今日 0 例重复。多行意味着源库同一天有
        # 重复 bar（或时区取日把两天并进了一天），此时 want[0] 取到哪一行是碰运气。
        assert len(want) == 1, f"{code} {day}: source has {len(want)} bars for one (symbol, day): {want}"
        src_close = want[0][0]
        # 显式判定 NULL，而不是让 float(None) 抛一个和真实问题无关的 TypeError。
        assert src_close is not None, f"{code} {day}: source close is NULL while qlib reads {float(qlib_v)}"
        src_v = float(src_close)
        assert abs(float(qlib_v) - src_v) <= abs(src_v) * 1e-6 + 1e-9, f"{code} {day}: qlib={float(qlib_v)} source={src_v}"
        checked += 1

    breakdown = "; ".join(f"{reason}={len(hits)}{hits[:5] if hits else ''}" for reason, hits in skipped.items())
    assert checked == len(picked), (
        f"picked={len(picked)} compared={checked} (其中 nan_verified={len(nan_verified)}) "
        f"skipped={len(picked) - checked} -> {breakdown}. "
        f"每个抽样对都必须完成数值比对或已验证的 NaN 停牌配对；skip 不是「正常停牌」的通行证，而是本层没验完。"
    )


def test_gapped_symbol_alignment(db, pool_members):
    """停牌/中途上市的标的最容易暴露日历错位，专门挑一个来测。"""
    from qlib.data import D

    candidates = []
    for code in pool_members("cn_stocks"):
        span = D_calendar(code)
        if len(span) < 100:
            continue
        n_src = db.execute(
            "SELECT count(*) FROM bars WHERE symbol=? AND frequency='1d' AND adjust='qfq' AND market='CN'",
            [from_qlib_code(code)],
        ).fetchone()[0]
        if 0 < n_src < len(span):
            candidates.append((code, n_src, len(span)))
    assert candidates, "no partially-covered instrument found; check the export filter"

    code, n_src, _ = max(candidates, key=lambda c: c[2] - c[1])
    df = D.features([code], ["$close"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1]).dropna()
    # 两侧口径必须一致，"相等"才是可判定的比较：Qlib 侧被 start/end 夹在窗口里，源库侧原来
    # 只有下界，于是窗口外尾部的那几天（实测 SH601995 多 2026-10-08/09/12 三天：351 vs 348）
    # 会把上界方向变成假报警。这里补 <= SEGMENTS["test"][1] 把源库夹进同一个窗口。
    src_days = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT strftime(CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE), '%Y-%m-%d') FROM bars "
            "WHERE symbol=? AND frequency='1d' AND adjust='qfq' AND market='CN' "
            "AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) >= ? "
            "AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) <= CAST(? AS DATE)",
            [from_qlib_code(code), HISTORY_START, SEGMENTS["test"][1]],
        ).fetchall()
    }
    # get_level_values 是「整个 MultiIndex」的方法，不是单行元组的方法
    qlib_days = {
        d.strftime("%Y-%m-%d") for d in df.index.get_level_values("datetime").unique()
    }
    assert qlib_days, f"{code}: qlib reads zero non-NaN trading days inside the test window"
    # 双向断言。上一版只写 `qlib_days <= src_days`，那只证明「Qlib 不凭空造日」，
    # 证明不了「Qlib 不丢源库已有的交易日」—— 后者正是本层要抓的静默丢数据；而唯一能
    # 抓到它的 D3 NaN 分支在本轮种子下触发 0 次，等于这条方向只有纸面守卫。
    assert qlib_days == src_days, (
        f"{code}: qlib {len(qlib_days)} 天 vs source {len(src_days)} 天 "
        f"(窗口 {HISTORY_START}..{SEGMENTS['test'][1]}); "
        f"only in qlib {sorted(qlib_days - src_days)[:5]}; "
        f"only in source {sorted(src_days - qlib_days)[:5]}"
    )
