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


def verify_store() -> None:
    """dump_bin 的失败可能只落在日志里；写池之前先证实库真的建起来了。"""
    cal_path = QLIB_DIR / "calendars" / f"{FREQ}.txt"
    all_path = QLIB_DIR / "instruments" / "all.txt"
    n_cal = len(cal_path.read_text(encoding="utf-8").split()) if cal_path.exists() else 0
    if n_cal == 0:
        raise RuntimeError(f"{cal_path} missing or empty; feature dump failed")
    n_all = len([ln for ln in all_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
    close_bin = f"close.{FREQ}.bin"
    n_feat = sum(
        1
        for d in (QLIB_DIR / "features").iterdir()
        if d.is_dir() and (d / close_bin).exists()
    )
    print(f"store: {n_cal} calendar days, {n_all} instruments in all.txt, {n_feat} dirs with {close_bin}")
    if n_feat != n_all:
        raise RuntimeError(
            f"features/ holds {n_feat} complete instruments but all.txt lists {n_all}; dump incomplete"
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
    verify_store()
    print("pools:", write_pools(QLIB_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
