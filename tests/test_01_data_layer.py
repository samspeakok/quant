"""Layer 1: does qlib read back the exact numbers that are in the source database?

from_qlib_code below is a DELIBERATE duplicate of tools/build_qlib_data.py's mapping.
If we reused the exporter's function, a reversed mapping would be wrong on both sides
and this whole cross-check would silently validate itself.
"""

from __future__ import annotations

import random
from datetime import date as _date
from functools import lru_cache

import pandas as pd
import pytest

from config.settings import BENCHMARK, HISTORY_START, QLIB_DIR, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


def from_qlib_code(qlib_code: str) -> str:
    return f"{qlib_code[2:]}.{qlib_code[:2]}"


def test_calendar_contains_no_foreign_trading_days(db):
    """CN 日历里的每一个交易日都必须在源库的 CN 数据中出现。
    非 CN 标的混入会带来幽灵交易日，让所有 A 股在该日全为 NaN。"""
    from qlib.data import D

    cal = {d.strftime("%Y-%m-%d") for d in D.calendar()}
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


def _pick_samples(pool_members) -> list[tuple[str, str]]:
    """每个池取 min(20, 池大小) —— cn_index 只有 10 个，写死 20 会让测试直接失败。
    样本在测试体内自取，不用 @pytest.mark.parametrize：parametrize 在收集阶段就要拿到
    fixture 的值，而那时 qlib 还没 init。"""
    random.seed(20261015)
    picked: list[tuple[str, str]] = []
    for pool in ("cn_stocks", "cn_etf", "cn_index"):
        members = pool_members(pool)
        for code in random.sample(members, min(20, len(members))):
            days = D_calendar(code)
            if len(days) < 10:
                continue
            for d in [days[0], days[-1], days[len(days) // 3], days[len(days) // 2], days[-3]]:
                picked.append((code, d.strftime("%Y-%m-%d")))
    return picked


def test_close_matches_source_database(db, pool_members):
    """核心断言：Qlib 读出的收盘价 == 源库里那一天的收盘价。"""
    from qlib.data import D

    picked = _pick_samples(pool_members)
    assert picked, "no samples selected"

    checked = 0
    for code, day in picked:
        got = D.features([code], ["$close"], start_time=day, end_time=day)["$close"]
        if got.empty:
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
            continue
        assert want, f"{code} {day}: qlib has a bar but the source database has none"
        src_v = float(want[0][0])
        assert abs(float(qlib_v) - src_v) <= abs(src_v) * 1e-6 + 1e-9, f"{code} {day}: qlib={float(qlib_v)} source={src_v}"
        checked += 1

    assert checked >= 50, f"only {checked} pairs compared; sampling is broken"


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
    src_days = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT strftime(CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE), '%Y-%m-%d') FROM bars "
            "WHERE symbol=? AND frequency='1d' AND adjust='qfq' AND market='CN' "
            "AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) >= ?",
            [from_qlib_code(code), HISTORY_START],
        ).fetchall()
    }
    # get_level_values 是「整个 MultiIndex」的方法，不是单行元组的方法
    qlib_days = {
        d.strftime("%Y-%m-%d") for d in df.index.get_level_values("datetime").unique()
    }
    assert qlib_days <= src_days, f"qlib reports days absent from source: {sorted(qlib_days - src_days)[:5]}"
