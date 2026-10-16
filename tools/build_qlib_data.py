"""Build the Qlib binary store from exported CSVs, then write per-asset instrument pools."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    EXPORT_CSV_DIR,
    FREQ,
    POOLS,
    QLIB_DIR,
    VENDOR_DUMP_BIN,
    classify_symbol,
)

INSTRUMENTS_SEP = "\t"


def run_dump() -> None:
    """Invoke dump_bin.py's fire CLI as a subprocess.

    不用 importlib 在进程内加载：DumpDataAll._dump_features 用 ProcessPoolExecutor，
    Windows 的 spawn 子进程按 "模块名.限定名" 反序列化被提交的绑定方法。作为子进程
    运行时 dump_bin.py 本身就是 __main__，类以 __main__.DumpDataAll 序列化，子进程
    可直接重建；而 importlib 加载的 "dump_bin" 不在子进程的导入路径上，会在耗时数分
    钟的 CSV 加载之后才在池内抛 PicklingError / ModuleNotFoundError。
    """
    cmd = [
        sys.executable,
        str(VENDOR_DUMP_BIN),
        "dump_all",
        f"--data_path={EXPORT_CSV_DIR}",
        f"--qlib_dir={QLIB_DIR}",
        f"--freq={FREQ}",
        "--date_field_name=date",
        "--symbol_field_name=symbol",
        # date 列本身不该成为一个特征字段
        "--exclude_fields=date",
        "--max_workers=8",
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"dump_bin.py exited with {proc.returncode}; store is not trustworthy")


def verify_store(qlib_dir: Path = QLIB_DIR, csv_dir: Path = EXPORT_CSV_DIR) -> None:
    """dump_bin 的失败可能只落在日志里；写池之前先绝对地证实库真的建起来了。

    为什么不能只比较 n_feat 与 n_all：两者都由同一次 dump 产出，相等只说明"这次 dump 自洽"，
    不说明"该有的标的都在"。`--limit_nums=3` 重跑一遍就会得到 3 == 3 的平凡通过，随后
    write_pools 写出 3 成员的池、main() 返回 0 —— 闸门形同虚设。

    锚点取闸门自身不产出的东西：tools/export_bars.py 保证一个标的一张 CSV（写之前清空
    陈旧 *.csv，写完断言 written == symbols），所以 instruments/all.txt 的行数必须等于
    导出目录里的 CSV 张数。features/ 还要逐个目录核对 close.{FREQ}.bin，并把 all.txt 的
    标的名与实到目录名做集合比对 —— 纯计数相等也可能被"缺一个又多一个"蒙过去。
    """
    cal_path = qlib_dir / "calendars" / f"{FREQ}.txt"
    all_path = qlib_dir / "instruments" / "all.txt"
    feats_dir = qlib_dir / "features"
    close_bin = f"close.{FREQ}.bin"

    n_cal = len(cal_path.read_text(encoding="utf-8").split()) if cal_path.exists() else 0
    if n_cal == 0:
        raise RuntimeError(f"{cal_path} missing or empty; feature dump failed")
    if not all_path.exists():
        raise RuntimeError(f"{all_path} missing; instruments dump failed")

    codes = [ln.split(INSTRUMENTS_SEP)[0] for ln in all_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n_all = len(codes)

    n_csv = len(list(csv_dir.glob("*.csv"))) if csv_dir.exists() else 0
    if n_csv == 0:
        raise RuntimeError(
            f"{csv_dir} holds 0 CSVs, so nothing anchors {n_all} instruments in {all_path}; "
            "run tools/export_bars.py first"
        )
    if n_all != n_csv:
        raise RuntimeError(
            f"{all_path} lists {n_all} instruments but {csv_dir} holds {n_csv} CSVs; "
            f"dump truncated or export stale — refusing to write pools over {n_all} of {n_csv} symbols"
        )

    complete = (
        {d.name for d in feats_dir.iterdir() if d.is_dir() and (d / close_bin).exists()}
        if feats_dir.exists()
        else set()
    )
    expected = {code.lower() for code in codes}
    missing = sorted(expected - complete)
    unexpected = sorted(complete - expected)
    if len(complete) != n_all or missing or unexpected:
        raise RuntimeError(
            f"features/ holds {len(complete)} dirs with {close_bin} but {all_path} lists {n_all} "
            f"instruments (missing {missing}, unexpected {unexpected}); dump incomplete"
        )

    print(
        f"store: {n_cal} calendar days, {n_all} instruments == {n_csv} CSVs, "
        f"{len(complete)} dirs with {close_bin}"
    )


def _from_qlib_code(qlib_code: str) -> str:
    """SH600519 -> 600519.SH，用于把池成员送回分类函数。"""
    return f"{qlib_code[2:]}.{qlib_code[:2]}"


def write_pools(qlib_dir: Path) -> dict[str, int]:
    all_path = qlib_dir / "instruments" / "all.txt"
    rows = []
    for line in all_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        code, start, end = line.split(INSTRUMENTS_SEP)[:3]
        rows.append({"code": code, "start": start, "end": end, "asset": classify_symbol(_from_qlib_code(code))})
    if not rows:
        raise RuntimeError(f"{all_path} holds no instruments")
    df = pd.DataFrame(rows)
    dup = df[df.duplicated("code", keep=False)]
    if len(dup):
        raise RuntimeError(f"duplicate symbols in {all_path}: {sorted(dup['code'].unique())}")

    counts = {}
    for pool, asset in POOLS.items():
        sub = df[df["asset"] == asset]
        out = qlib_dir / "instruments" / f"{pool}.txt"
        out.write_text(
            "".join(f"{r.code}{INSTRUMENTS_SEP}{r.start}{INSTRUMENTS_SEP}{r.end}\n" for r in sub.itertuples()),
            encoding="utf-8",
        )
        counts[pool] = len(sub)
    unmapped = df[df["asset"].isna()]
    if len(unmapped):
        print(f"[warn] {len(unmapped)} instruments unclassified and excluded from all pools")
    return counts


def main() -> int:
    if not EXPORT_CSV_DIR.exists():
        print("[error] no CSV export; run tools/export_bars.py first", file=sys.stderr)
        return 1
    QLIB_DIR.mkdir(parents=True, exist_ok=True)
    run_dump()
    verify_store(QLIB_DIR, EXPORT_CSV_DIR)
    print("pools:", write_pools(QLIB_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
