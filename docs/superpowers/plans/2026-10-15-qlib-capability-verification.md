# Qlib 能力验证工程 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把本机 duckdb 里的 31.5 万行真实 A 股日线（698 个标的）接入 Microsoft Qlib，并逐层验证「数据 → 表达式 → 因子 → 数据集 → 模型 → 回测」六层闭环是否真的可用。

**Architecture:** 单向依赖的三层目录。`tools/` 是唯一的造数据层（duckdb 只读导出 → 官方 dump_bin → `.bin`），产出 `qlib_data/`；`tests/` 按层断言功能可用且不产出任何文件；`research/` 消费已验证的数据跑出真实回测产物。三者互不触发。

**Tech Stack:** Python 3.12.10、pyqlib 0.9.7（官方 cp312-win_amd64 wheel）、duckdb 1.5.5（只读）、pandas 3.0.6、numpy 2.5.4、LightGBM 4.7.0、pytest。

**Spec:** `docs/superpowers/specs/2026-10-15-qlib-capability-verification-design.md`

## Global Constraints

以下每条都是实测结论，不是推测。实现时不得与之冲突，也不得"顺手改"。

- 工作目录：`D:\Desktop\个人事项\quant-l\agent space`，git 分支 `main`。
- **本机未配置 git 身份，用户明确决定暂不配置。** 因此每个任务的最后一步只做 `git add <具体文件>` 暂存，**不执行 `git commit`**。全部任务完成后由用户给出身份再统一提交。
- Python 一律用 `./.venv/Scripts/python.exe`（Windows + Git Bash）。系统 Python 3.12.10 不带 qlib。
- **`market.duckdb` 正被主系统持续写入，同目录有 `housekeeping.lock`。所有连接必须 `read_only=True`，只执行 SELECT。** 任何写操作、`CHECKPOINT`、`VACUUM` 都禁止。产物只落在本目录。
- 数据过滤条件固定：`frequency = '1d' AND adjust = 'qfq' AND market = 'CN'`，再加时间下界 `ts >= HISTORY_START`。
- **实测基数（收窄后）**：194,738 行 / 698 个标的 / **529 个交易日**。不加时间下界时是 314,676 行但日历被指数的 1990 年数据撑到 **8,741 天**，而 A 股只覆盖其中约 500 天 —— 每个 `.bin` 数组都会被 NaN 填满。收窄后标的数不变（698），NaN 填充降一个数量级。
- Qlib 区域固定 `region='cn'`，频率固定 `freq='day'`。
- 符号格式：duckdb 侧 `600519.SH`，Qlib 侧 `SH600519`。
- 标的池按**代码前缀规则**分类，**不使用 `asset_type` 列** —— 实测 `000001.SH`（上证指数）与 `000905.SH`（中证500）同时被标成 `stock` 和 `index`，按标签分池会把指数当股票买入，而指数没有可成交价，会凭空抬高回测收益。
- 三个 instrument 池 `cn_stocks` / `cn_etf` / `cn_index` 绝不合并。
- 日期区间（2026-10-15 依实测数据重定，见下方"数据现实"）：train 2025-01-01~2025-08-31、valid 2025-09-01~2025-12-31、test 2026-01-01~2026-09-30，`HISTORY_START = 2024-08-05`。**每个区间右端都留出的尾部数据（2026-10-01~2026-10-14）不进任何区间**，保持一段真正的样本外参照。
- **数据现实（实测，推翻了初版计划的区间设定）**：本库 A 股日线只从 2021-08-26 开始，且 2021/2022/2023 各只有 **1** 只标的，2024 年 55 只，2025 年 347 只，2026 年 642 只。ETF 有 2007→2026 的长历史但每年仅 8~10 只；指数 1990→2026 共 10 只且不可交易。因此**任何跨多年的股票训练窗口在这份数据上都不存在**，回测性能数字不具备可信度，本轮六层的判据是"机制是否真的跑通"，不是"策略是否赚钱"。补历史数据是后续独立工程。
- 已核实的真实 API 参数名（不要按网络资料改回旧写法）：`dump_bin.py` 用 **`--data_path`**（不是 `--csv_path`）；`qlib.backtest.backtest(start_time, end_time, strategy, executor, benchmark, account, exchange_kwargs, pos_type)`；`DatasetH.prepare(segments, col_set=..., ...)`；`Alpha158.get_feature_config()` 返回 `([ops], [names])` 二元组。
- Alpha158 的标签是 `Ref($close, -2)/Ref($close, -1) - 1`，**设计上就前看两天**。看到"用了未来数据"不要当 bug 修掉，它是预测目标本身。

## 文件结构

```
agent space/
├─ config/settings.py            路径、过滤条件、符号分类规则、日期区间 —— 唯一事实源
├─ tools/
│   ├─ export_bars.py            duckdb 只读 → 每标的一张 CSV
│   └─ build_qlib_data.py        调 vendor dump_bin 建 .bin + 写三个 instrument 池
├─ tests/
│   ├─ conftest.py               缺 qlib_data 则整体 skip；qlib.init；duckdb 只读连接
│   ├─ test_00_symbol_rule.py    符号分类规则（唯一不需要数据的测试）
│   ├─ test_01_data_layer.py     数据层 + 与 duckdb 的收盘价交叉校验
│   ├─ test_02_expression_ops.py 表达式引擎
│   ├─ test_03_alpha_factors.py  Alpha158 / Alpha360
│   ├─ test_04_dataset_split.py  DatasetH 切分
│   ├─ test_05_model_train.py    LightGBM 训练与预测
│   └─ test_06_backtest.py       TopkDropout 撮合与报告
├─ research/alpha158_lgbm_backtest.py
├─ vendor/qlib/                  已 clone（gitignored）
├─ qlib_data/                    生成物（gitignored）
├─ artifacts/                    生成物（gitignored）
└─ docs/superpowers/specs|plans/
```

---

### Task 1: 配置中心与符号分类规则

这是唯一不依赖数据就能完整测试的部分，而它恰好是整条链路里最容易出错、错了又最不明显的逻辑，所以放在最前面并测透。

**Files:**
- Create: `config/__init__.py`（空文件）
- Create: `config/settings.py`
- Create: `tests/__init__.py`（空文件）
- Create: `tests/test_00_symbol_rule.py`
- Modify: 无
- 依赖：安装 `pytest`

**Interfaces:**
- Consumes: 无
- Produces:
  - `settings.ROOT: Path`, `settings.DUCKDB_PATH: Path`, `settings.VENDOR_DUMP_BIN: Path`, `settings.EXPORT_CSV_DIR: Path`, `settings.QLIB_DIR: Path`, `settings.ARTIFACTS: Path`
  - `settings.SQL_FILTER: str`
  - `settings.classify_symbol(symbol: str) -> str | None` —— 返回 `"stock"` / `"etf"` / `"index"` / `None`
  - `settings.to_qlib_code(symbol: str) -> str` —— `600519.SH` → `SH600519`
  - `settings.POOLS: dict[str, str]` —— `{"cn_stocks": "stock", "cn_etf": "etf", "cn_index": "index"}`
  - `settings.SEGMENTS: dict[str, tuple[str, str]]`、`settings.HISTORY_START: str`、`settings.BENCHMARK: str`

- [ ] **Step 1: 安装 pytest**

```bash
cd "D:/Desktop/个人事项/quant-l/agent space"
./.venv/Scripts/python.exe -m pip install pytest
```

- [ ] **Step 2: 写 `config/settings.py`**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DUCKDB_PATH = Path("D:/Desktop/个人事项/codex/量化分析/data/market.duckdb")
VENDOR_DUMP_BIN = ROOT / "vendor" / "qlib" / "scripts" / "dump_bin.py"
EXPORT_CSV_DIR = ROOT / "tools" / "_export" / "csv"
QLIB_DIR = ROOT / "qlib_data"
ARTIFACTS = ROOT / "artifacts"

REGION = "cn"
FREQ = "day"

# 非 CN 市场的 10 个标的（US/HK/KR/GLOBAL 共 45,413 行）会把它们各自的
# 交易日并进 calendars/day.txt，制造出数千个所有 A 股全为 NaN 的幽灵交易日。
SQL_FILTER = "frequency = '1d' AND adjust = 'qfq' AND market = 'CN'"

_SH_STOCK = ("600", "601", "603", "605", "688", "689")
_SH_ETF = ("51", "52", "56", "58")
_SZ_STOCK = ("000", "001", "002", "003", "300", "301", "302")
_SZ_ETF = ("15", "16")
# 899xxx 是北证系列指数（899001 即北证50），必须先于股票段判定，否则指数会被
# 分进可交易池 —— 指数没有可成交价，回测会凭空抬高收益。
_BJ_INDEX = ("899",)
_BJ_STOCK = ("43", "83", "87", "88", "92")


def classify_symbol(symbol: str) -> str | None:
    """按交易所与代码段判定资产类型。不使用 asset_type —— 实测该列自相矛盾。"""
    code, _, exch = symbol.partition(".")
    exch = exch.upper()
    if exch == "SH":
        if code.startswith("000"):
            return "index"
        if code.startswith(_SH_STOCK):
            return "stock"
        if code.startswith(_SH_ETF):
            return "etf"
    elif exch == "SZ":
        if code.startswith("399"):
            return "index"
        if code.startswith(_SZ_STOCK):
            return "stock"
        if code.startswith(_SZ_ETF):
            return "etf"
    elif exch == "BJ":
        if code.startswith(_BJ_INDEX):
            return "index"
        if code.startswith(_BJ_STOCK):
            return "stock"
    return None


def to_qlib_code(symbol: str) -> str:
    code, _, exch = symbol.partition(".")
    return f"{exch.upper()}{code}"


POOLS = {"cn_stocks": "stock", "cn_etf": "etf", "cn_index": "index"}

BENCHMARK = "SH000300"
# 股票数据实际从 2021-08-26 才开始，且 2021-2023 各只有 1 只标的；
# 2024-08-05 起才有成批股票（当年 55 只、2025 年 347 只、2026 年 642 只）。
# HISTORY_START 取 2024-08-05，给 Alpha158 的滚动窗口留出预热期。
HISTORY_START = "2024-08-05"

SEGMENTS = {
    "train": ("2025-01-01", "2025-08-31"),
    "valid": ("2025-09-01", "2025-12-31"),
    "test": ("2026-01-01", "2026-09-30"),
}

# 数据到今天 2026-10-14。这段尾部不进任何区间，作为真正的样本外参照。
HELDOUT_START = "2026-10-01"
```

- [ ] **Step 3: 写失败的测试 `tests/test_00_symbol_rule.py`**

分类规则的每一条分支都要有用例，且必须包含实测发现的那几个坑。

```python
import pytest

from config.settings import POOLS, SEGMENTS, classify_symbol, to_qlib_code


@pytest.mark.parametrize(
    "symbol, expected",
    [
        # 沪市主板与科创板
        ("600519.SH", "stock"),
        ("601318.SH", "stock"),
        ("603288.SH", "stock"),
        ("605499.SH", "stock"),
        ("688256.SH", "stock"),
        # 深市主板与创业板
        ("000001.SZ", "stock"),
        ("002594.SZ", "stock"),
        ("300750.SZ", "stock"),
        ("301158.SZ", "stock"),
        ("302132.SZ", "stock"),
        # 北交所
        ("832004.BJ", "stock"),
        ("430139.BJ", "stock"),
        ("889117.BJ", "stock"),
        # 北证系列指数：899xxx 必须是 index，不能落进可交易股票池
        ("899001.BJ", "index"),
        # ETF：15/16 开头在深市，51/56/58 与 52 开头在上海
        ("159611.SZ", "etf"),
        ("510300.SH", "etf"),
        ("520880.SH", "etf"),
        ("562500.SH", "etf"),
        ("588000.SH", "etf"),
        # 指数：000 开头在上海是指数，399 开头在深市是指数
        ("000300.SH", "index"),
        ("000001.SH", "index"),
        ("000905.SH", "index"),
        ("399006.SZ", "index"),
        # 非 CN 市场一律拒绝，绝不能进导出
        (".DJI.US", None),
        ("^HSI.HK", None),
        ("DXY.GLOBAL", None),
        ("KOSPI.KR", None),
        # 无法识别
        ("BOGUS", None),
    ],
)
def test_classify_symbol(symbol, expected):
    assert classify_symbol(symbol) == expected


def test_shanghai_000_prefix_is_index_not_stock():
    """000 开头在深圳是股票、在上海是指数。这是 asset_type 标签出错的确切位置。"""
    assert classify_symbol("000001.SH") == "index"
    assert classify_symbol("000001.SZ") == "stock"


def test_to_qlib_code():
    assert to_qlib_code("600519.SH") == "SH600519"
    assert to_qlib_code("000001.SZ") == "SZ000001"
    assert to_qlib_code("159611.SZ") == "SZ159611"


def test_benchmark_is_a_classified_index():
    from config.settings import BENCHMARK

    assert classify_symbol("000300.SH") == "index"
    assert to_qlib_code("000300.SH") == BENCHMARK


def test_segments_are_ordered_and_disjoint():
    assert SEGMENTS["train"][1] < SEGMENTS["valid"][0]
    assert SEGMENTS["valid"][1] < SEGMENTS["test"][0]


def test_segments_are_the_exact_measured_windows():
    """钉住四个端点。只断言顺序的话，整段窗口平移一年也照样全绿。"""
    assert SEGMENTS == {
        "train": ("2025-01-01", "2025-08-31"),
        "valid": ("2025-09-01", "2025-12-31"),
        "test": ("2026-01-01", "2026-09-30"),
    }


def test_heldout_tail_is_excluded_from_every_segment():
    """数据到 2026-10-14，这段尾部不进任何区间，留作样本外参照。"""
    for name, (_start, end) in SEGMENTS.items():
        assert end < HELDOUT_START, f"{name} leaks into the held-out tail"


def test_pools_map_to_the_three_classifiers():
    assert set(POOLS.values()) == {"stock", "etf", "index"}
```

- [ ] **Step 4: 运行测试，确认失败原因是有意义的**

```bash
cd "D:/Desktop/个人事项/quant-l/agent space"
./.venv/Scripts/python.exe -m pytest tests/test_00_symbol_rule.py -v
```

Expected: 若 Step 2 已正确写入则直接全绿。若出现 `ModuleNotFoundError: No module named 'config'`，补 `tests/conftest.py` 把项目根加入 `sys.path`：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 5: 暂存**

```bash
git add config/__init__.py config/settings.py tests/__init__.py tests/test_00_symbol_rule.py tests/conftest.py
```

---

### Task 2: duckdb 只读导出器

把 `bars` 表按过滤条件导出成 dump_bin 认得的形态：**每个标的一张 CSV**，文件名即 Qlib 代码，列为 `date,open,high,low,close,volume,amount,factor`。

**Files:**
- Create: `tools/export_bars.py`
- Create: `tools/__init__.py`（空文件）
- Create: `tests/test_export_bars.py`
- Modify: `tests/conftest.py`（追加两个 fixture）

**Interfaces:**
- Consumes: `settings.DUCKDB_PATH`、`settings.SQL_FILTER`、`settings.classify_symbol`、`settings.to_qlib_code`、`settings.EXPORT_CSV_DIR`
- Produces:
  - `export_bars.connect_readonly() -> duckdb.DuckDBPyConnection`
  - `export_bars.load_bars(con) -> tuple[pandas.DataFrame, int]` —— 列 `symbol, date, open, high, low, close, volume, amount, asset`，`date` 为 `YYYY-MM-DD` 字符串，已按 classify 过滤掉 `None`；第二个返回值是 **classify 丢弃之前** 的行数。若只返回帧而把 `rows_in` 在丢行后才测，`dropped_by_classify` 会永远是 0 —— 一个会说谎的体检计数器，正是本层约束禁止的
  - `export_bars.HygieneReport` dataclass：`rows_in, rows_out, symbols, dropped_by_classify, dup_rows, bad_price_rows`；须满足守恒 `rows_in - dup_rows - bad_price_rows == rows_out`
  - `export_bars.write_csvs(df, out_dir, rows_in: int) -> HygieneReport` —— 写之前先删除 `out_dir` 下旧的 `*.csv`（否则会留下上一轮的全历史文件被 dump_bin 当现况吞掉），写完断言文件数等于标的数
  - `export_bars.main() -> int`

- [ ] **Step 1: 先写"只读约束"的失败测试**

这条测试保护的是外部系统的真实数据，优先级高于一切功能测试。

```python
import pytest
import duckdb

from config.settings import DUCKDB_PATH, EXPORT_CSV_DIR
from tools.export_bars import load_bars, connect_readonly, HygieneReport


def test_writing_to_the_source_database_is_rejected(tmp_path):
    """主系统正在写这个库。我们的连接必须是只读的，且这一约束要有测试兜底。"""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    with pytest.raises(Exception):
        con.execute("CREATE TABLE _qoder_probe (x INTEGER)")
    con.close()


def test_export_is_reproducible_and_market_filtered():
    con = connect_readonly()
    df = load_bars(con)
    con.close()
    # 实测：CN + 1d + qfq 共 698 个标的
    assert len(set(df["symbol"])) == 698
    assert df["asset"].isin(["stock", "etf", "index"]).all()
    assert df["symbol"].str.contains(r"\.(SH|SZ|BJ)$").all()
    assert not df["symbol"].str.endswith((".US", ".HK", ".KR", ".GLOBAL")).any()
```

- [ ] **Step 2: 运行确认失败**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_export_bars.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.export_bars'`。

- [ ] **Step 3: 写 `tools/export_bars.py`**

```python
"""Read-only export of daily bars from the main system's duckdb into CSV per symbol."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (  # noqa: E402
    DUCKDB_PATH,
    EXPORT_CSV_DIR,
    HISTORY_START,
    SQL_FILTER,
    classify_symbol,
    to_qlib_code,
)

FIELDS = ["open", "high", "low", "close", "volume", "amount"]

# `ts` 是 TIMESTAMPTZ。直接 CAST 成 DATE 会按 UTC 取日，与 strftime 的本地取日不一致，
# 差 8 小时就是差一个交易日 —— 那正是本项目要用交叉校验去抓的那类静默错位。
# 因此在这里显式按 Asia/Shanghai 落到本地日历，且不依赖宿主机时区。
QUERY = f"""
    SELECT symbol, ts AT TIME ZONE 'Asia/Shanghai' AS ts, {", ".join(FIELDS)}
    FROM bars
    WHERE {SQL_FILTER}
      AND CAST(ts AT TIME ZONE 'Asia/Shanghai' AS DATE) >= '{HISTORY_START}'
"""


def connect_readonly() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


@dataclass
class HygieneReport:
    rows_in: int = 0
    rows_out: int = 0
    symbols: int = 0
    dropped_by_classify: int = 0
    dup_rows: int = 0
    bad_price_rows: int = 0

    def as_text(self) -> str:
        return (
            f"rows_in            {self.rows_in:>7d}\n"
            f"rows_out           {self.rows_out:>7d}\n"
            f"symbols            {self.symbols:>7d}\n"
            f"dropped_by_classify {self.dropped_by_classify:>7d}\n"
            f"dup_rows            {self.dup_rows:>7d}\n"
            f"bad_price_rows      {self.bad_price_rows:>7d}"
        )


def load_bars(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(QUERY).fetch_df()
    df["date"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d")
    df["asset"] = df["symbol"].map(classify_symbol)
    unmapped = int(df["asset"].isna().sum())
    if unmapped:
        print(f"[warn] {unmapped} rows dropped: symbol not classified")
    df = df.dropna(subset=["asset"])
    keep = ["symbol", "date", "asset"] + FIELDS
    return df.loc[:, keep]


def write_csvs(df: pd.DataFrame, out_dir: Path) -> HygieneReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = HygieneReport(rows_in=len(df))
    # dump_bin 会静默按日期去重；重复行必须在导出前暴露出来而不是交给它处理
    dup = df.duplicated(subset=["symbol", "date"]).sum()
    if dup:
        print(f"[warn] {dup} duplicate (symbol, date) rows removed")
        df = df.drop_duplicates(subset=["symbol", "date"])
    rep.dup_rows = int(dup)

    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    if bad:
        print(f"[warn] {bad} rows with non-positive price removed")
        df = df[~(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    rep.bad_price_rows = int(bad)

    df = df.sort_values(["symbol", "date"])
    rep.rows_out = len(df)
    rep.symbols = df["symbol"].nunique()

    for symbol, sub in df.groupby("symbol", sort=True):
        path = out_dir / f"{to_qlib_code(symbol)}.csv"
        sub = sub.drop(columns=["symbol", "asset"]).copy()
        # 数据本身已是前复权，factor=1.0 表示 Qlib 侧不再做二次复权
        sub["factor"] = 1.0
        sub.to_csv(path, index=False)
    return rep


def main() -> int:
    con = connect_readonly()
    try:
        df = load_bars(con)
    finally:
        con.close()
    report = write_csvs(df, EXPORT_CSV_DIR)
    print(report.as_text())
    # 实测（CN + 1d + qfq + ts>=HISTORY_START）= 194,738 行 / 698 标的 / 529 交易日
    # 低于 90% 说明过滤条件或源库状态变了，必须硬失败
    if report.rows_out < 194_738 * 0.9:
        print("[error] exported row count far below expectation", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_export_bars.py -v
```

Expected: 2 passed。

- [ ] **Step 5: 实际导出并检查产物**

```bash
./.venv/Scripts/python.exe tools/export_bars.py
ls tools/_export/csv | wc -l
head -3 tools/_export/csv/SH600519.csv
```

Expected: 打印体检报告；约 698 个 CSV；表头为 `date,open,high,low,close,volume,amount,factor`。

- [ ] **Step 6: 确认没污染主系统目录**

对比文件名清单，**不要比 mtime/size** —— 主系统在持续写入，那些本来就一直变。

```bash
ls "D:/Desktop/个人事项/codex/量化分析/data" | sort > /tmp/after.txt
diff <(grep -v '^\.' /tmp/after.txt) <(grep -v '^\.' /tmp/before.txt 2>/dev/null || echo SKIP) || true
```

执行前先跑 `ls "D:/Desktop/个人事项/codex/量化分析/data" | sort > /tmp/before.txt` 建立基线。Expected: 无新增文件，尤其不得出现 `.wal` / `.tmp`。

- [ ] **Step 7: 暂存**

```bash
git add tools/__init__.py tools/export_bars.py tests/test_export_bars.py
```

---

### Task 3: 建库与 instrument 分组

**已核实的关键事实：`dump_bin.py` 的 `dump()` 只产出 `calendars/day.txt`、`features/*` 和 `instruments/all.txt`，不会创建任何按市场划分的标的池。** 三个池必须由我们自己写。

**与 spec §4.5 的一处偏离**：spec 写「在 `cn_index.txt` 注释里标注 hs300_pit 的位置」。实现不了 —— Qlib 的 instrument 文件是 `\t` 分隔的三列纯文本，读取方 `_read_instruments` 用 `pd.read_csv(sep="\t", names=[...])` 且不接受注释行，加一行说明会让它多出一个畸形标的。hs300_pit 的位置改记在本计划的 Task 11 README 第 6 节里。

**与 Task 4 的关系**：本任务的 `_from_qlib_code` 与 `tests/test_01_data_layer.py` 里的 `from_qlib_code` 是同一逻辑的两份独立实现。这是**故意的**，不要合并、不要让测试 import tools 里的版本 —— 否则映射写反时两侧一起错，交叉校验就失效了（详见 spec §5.2）。

**Files:**
- Create: `tools/build_qlib_data.py`
- Create: `tests/test_build_qlib_data.py`
- Modify: 无

**Interfaces:**
- Consumes: `settings.VENDOR_DUMP_BIN`、`settings.EXPORT_CSV_DIR`、`settings.QLIB_DIR`、`settings.POOLS`、`settings.classify_symbol`、`settings.FREQ`
- Produces:
  - `build_qlib_data.load_dump_bin() -> module` —— 用 `importlib` 从 vendor 路径加载
  - `build_qlib_data.run_dump(module) -> None`
  - `build_qlib_data.write_pools(qlib_dir: Path) -> dict[str, int]` —— 读 `instruments/all.txt`，按分类写三个池，返回 `{pool_name: 标的名}`
  - `build_qlib_data.main() -> int`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

import pytest

from config.settings import QLIB_DIR


pytestmark = pytest.mark.skipif(
    not (QLIB_DIR / "calendars" / "day.txt").exists(),
    reason="qlib_data not built; run: python tools/export_bars.py && python tools/build_qlib_data.py",
)


def test_calendar_is_strictly_increasing_and_unique():
    dates = Path(QLIB_DIR / "calendars" / "day.txt").read_text(encoding="utf-8").split()
    assert dates == sorted(set(dates))
    # 收窄导出后日历应约 529 个交易日；若这里出现数千，说明时间下界没生效，
    # 指数的 1990 年数据又把 NaN 填充撑回来了
    assert 400 <= len(dates) <= 600, f"calendar has {len(dates)} days, expected ~529"


def test_three_pool_files_exist_with_expected_sizes():
    """实测 CN 日线前复权：股票 626 / ETF 62 / 指数 10。"""
    expected = {"cn_stocks": 626, "cn_etf": 62, "cn_index": 10}
    for name, count in expected.items():
        path = QLIB_DIR / "instruments" / f"{name}.txt"
        assert path.exists(), f"missing {path}"
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == count, f"{name}: got {len(lines)} expected {count}"


def test_no_index_leaks_into_the_tradable_stock_pool():
    """指数混进可交易池会凭空抬高回测收益 —— 这是本项目最不可接受的一类错误。"""
    def members(name):
        return {
            ln.split("\t")[0]
            for ln in (QLIB_DIR / "instruments" / f"{name}.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    stocks, index = members("cn_stocks"), members("cn_index")
    assert not (stocks & index)
    assert "SH000001" not in stocks   # 上证指数
    assert "SH000905" not in stocks   # 中证500
    assert "SH000300" in index


def test_benchmark_instrument_has_features():
    bench = QLIB_DIR / "features" / "sh000300"
    assert bench.exists() and (bench / "close.day.bin").exists()
```

- [ ] **Step 2: 运行确认失败**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_qlib_data.py -v
```

Expected: 全部 SKIP（`qlib_data` 还不存在），此时 SKIP 是正确的失败形态。

- [ ] **Step 3: 写 `tools/build_qlib_data.py`**

```python
"""Build the Qlib binary store from exported CSVs, then write per-asset instrument pools."""

from __future__ import annotations

import importlib.util
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
    to_qlib_code,
)

INSTRUMENTS_SEP = "\t"


def load_dump_bin():
    spec = importlib.util.spec_from_file_location("dump_bin", VENDOR_DUMP_BIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_dump(module) -> None:
    dumper = module.DumpDataAll(
        data_path=str(EXPORT_CSV_DIR),
        qlib_dir=str(QLIB_DIR),
        freq=FREQ,
        date_field_name="date",
        symbol_field_name="symbol",
        # date 列本身不该成为一个特征字段
        exclude_fields="date",
        max_workers=8,
    )
    dumper.dump()


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
    df = pd.DataFrame(rows)

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
    run_dump(load_dump_bin())
    print("pools:", write_pools(QLIB_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 执行建库**

```bash
./.venv/Scripts/python.exe tools/build_qlib_data.py
```

Expected: dump_bin 跑完（314,676 行、698 个 CSV 文件，约数分钟），打印 `pools: {'cn_stocks': 626, 'cn_etf': 62, 'cn_index': 10}`。若某池数字不同，**先查是数据变了还是分类规则错了，不许直接改测试里的期望值。**

- [ ] **Step 5: 运行测试确认通过**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_build_qlib_data.py -v
```

Expected: 4 passed。若 `cn_stocks` 数量对不上，用 `tools/build_qlib_data.py` 里的 `classify_symbol` 逐个排查，别改前缀表来凑数。

- [ ] **Step 6: 暂存**

```bash
git add tools/build_qlib_data.py tests/test_build_qlib_data.py
```

---

### Task 4: 数据层交叉校验

整个工程最重要的一层。`dump_bin` 的日历错位**不报任何错**：Qlib 会正常读出数据，但每根 K 线整体偏移若干天，导致因子偷用未来数据、回测收益虚高、结论全错。只测"能读出来"抓不到它。

**Files:**
- Modify: `tests/conftest.py`（追加 fixture）
- Create: `tests/test_01_data_layer.py`

**Interfaces:**
- Consumes: `settings.QLIB_DIR`、`settings.DUCKDB_PATH`、`settings.SEGMENTS`、`settings.HISTORY_START`、`settings.BENCHMARK`
- Produces（供 Task 5~9 复用，名字必须一致）:
  - fixture `qlib_ready: bool` —— session 级，`qlib.init` 成功则 True
  - fixture `db: duckdb.DuckDBPyConnection` —— session 级只读连接
  - fixture `pool_members: Callable[[str], list[str]]` —— `"cn_stocks"` → `["SH600519", ...]`
  - `tests.test_01_data_layer.from_qlib_code(qlib_code: str) -> str`

- [ ] **Step 1: 扩展 `tests/conftest.py`**

```python
import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DUCKDB_PATH, QLIB_DIR, REGION  # noqa: E402

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
```

- [ ] **Step 2: 写 `tests/test_01_data_layer.py`**

```python
"""Layer 1: does qlib read back the exact numbers that are in the source database?

from_qlib_code below is a DELIBERATE duplicate of tools/build_qlib_data.py's mapping.
If we reused the exporter's function, a reversed mapping would be wrong on both sides
and this whole cross-check would silently validate itself.
"""

from __future__ import annotations

import random
from datetime import date as _date

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

    cal_list = list(D.calendar())
    # 必须先按有序列表断言，再降为集合：qlib data.py:675 是原样返回文件内容，
    # 既不 np.unique 也不排序，:174 的索引字典还会让重复日静默保留最后一个索引。
    # 若一开始就收成 set，重复日与乱序日会被吸收而 cal == src 照样通过。
    assert cal_list and cal_list == sorted(set(cal_list)), "calendar not strictly increasing"
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
    # 实测：pyqlib 0.9.7 与 vendor 上游的 D.calendar 签名都是 (start_time, end_time, freq, future)，
    # 没有 top_symbol 参数（传了会 TypeError）。旧 top_symbol 的语义 = 用该标的在 instruments
    # 文件里的起止去切全局日历，按同义实现，且取的仍是 Qlib 自己认为的起止：
    window_days = len(D.calendar(start_time=HISTORY_START, end_time=SEGMENTS["test"][1]))
    # 原计划的 `> 2000` 在收窄导出窗口后物理上不可能成立（上限只有 524 天），
    # 改为要求窗口内交易日 100% 覆盖 —— 比原断言严格得多
    covered = int(df["$close"].notna().sum())
    assert covered == window_days, f"benchmark covers {covered}/{window_days} in-window days"


@lru_cache(maxsize=1)
def _qlib_spans() -> dict[str, tuple[tuple[pd.Timestamp, pd.Timestamp], ...]]:
    """标的名 -> instruments 里的 (start, end) 覆盖区间，经 D.list_instruments 读出，不自己解析
    文件 —— 这里要的就是「Qlib 自己认为这个标的哪些天该有数据」。
    **注意**：`market="all"` 只用于按标的查区间，绝不可当可交易 universe 复制使用（三个池绝不合并）。
    缓存是必需的：D_calendar 会被调用 600+ 次，每次重建 698 项字典要 5 秒以上。"""
    from qlib.data import D

    return {
        code: tuple(spans)
        for code, spans in D.list_instruments(D.instruments(market="all"), as_list=False).items()
    }


def D_calendar(top_symbol: str) -> list[pd.Timestamp]:
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
        assert want, f"{code} {day}: qlib has a bar but the source database has none"
        qlib_v, src_v = float(got.iloc[0]), float(want[0][0])
        assert abs(qlib_v - src_v) <= abs(src_v) * 1e-6 + 1e-9, f"{code} {day}: qlib={qlib_v} source={src_v}"
        checked += 1

    assert checked == len(picked), f"only {checked}/{len(picked)} compared; skipped={skipped}"


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
    # 双向：只断言 <= 就永远证明不了"Qlib 不会丢掉源库有的日子"（静默丢数据那一类）。
    # 源侧查询必须同样加 <= SEGMENTS["test"][1] 上界 —— 实测不加就是 351 vs 348，
    # 多出的 2026-10-08/09/12 落在留出尾部，不是数据缺失。
    assert qlib_days == src_days, (
        f"only-in-qlib={sorted(qlib_days - src_days)[:5]} only-in-source={sorted(src_days - qlib_days)[:5]}"
    )
```

- [ ] **Step 3: 运行确认通过**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_01_data_layer.py -v
```

Expected: 4 passed。**这是整个工程的验收咽喉**：`test_close_matches_source_database` 通过意味着 numpy 2.5 / pandas 3.0 下 Qlib 二进制数据层真实可用，spec 里那条"已知未验证风险"就此销账。

- [ ] **Step 4: 若数值全部对不上且偏移固定天数**

不要改成"放宽容差"。固定偏移说明是 dump_bin 与源库时区差异：`ts` 是 `TIMESTAMP WITH TIME ZONE`，`CAST(ts AS DATE)` 按 UTC 取日，而 CSV 里用 `.dt.strftime` 按本地时区取日。修法是统一在一处转日期，两边都用同一个字符串。用下面这条快速判定：

```bash
./.venv/Scripts/python.exe -c "
import duckdb
con=duckdb.connect('D:/Desktop/个人事项/codex/量化分析/data/market.duckdb',read_only=True)
print(con.execute(\"select ts, CAST(ts AS DATE), strftime(ts,'%Y-%m-%d'), strftime(CAST(ts AS TIMESTAMP), '%Y-%m-%d') from bars where symbol='600519.SH' limit 3\").fetchall())"
```

- [ ] **Step 5: 暂存**

```bash
git add tests/conftest.py tests/test_01_data_layer.py
```

---

### Task 5: 表达式层

**Files:**
- Create: `tests/test_02_expression_ops.py`

**Interfaces:**
- Consumes: fixtures `qlib_ready`、`pool_members`
- Produces: 无（叶子测试）

- [ ] **Step 1: 写测试**

```python
"""Layer 2: the expression engine. Every expected value is recomputed from raw $close."""

import numpy as np
import pandas as pd
import pytest

from config.settings import HISTORY_START, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


def _raw(code: str) -> pd.Series:
    from qlib.data import D

    df = D.features([code], ["$close", "$volume"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
    return df["$close"].droplevel(0)


def test_ref_shifts_exactly_one_day():
    from qlib.data import D

    code = "SH600519"
    raw = _raw(code)
    ref = D.features([code], ["Ref($close, 1)"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
    got = ref.iloc[:, 0].droplevel(0).dropna()
    want = raw.shift(1).reindex(got.index).dropna()
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_rolling_mean_matches_pandas():
    code = "SH600519"
    raw = _raw(code)
    from qlib.data import D

    got = (
        D.features([code], ["Mean($close, 5)"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
        .iloc[:, 0]
        .droplevel(0)
        .dropna()
    )
    want = raw.rolling(5).mean().reindex(got.index).dropna()
    assert np.allclose(got.to_numpy(), want.to_numpy(), rtol=1e-6)


def test_std_of_returns_matches_pandas():
    code = "SH600519"
    raw = _raw(code)
    ret = raw / raw.shift(1) - 1
    from qlib.data import D

    got = (
        D.features([code], ["Std($close/Ref($close,1)-1, 20)"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
        .iloc[:, 0]
        .droplevel(0)
        .dropna()
    )
    want = ret.rolling(20).std().reindex(got.index).dropna()
    assert np.allclose(got.to_numpy(), want.to_numpy(), rtol=1e-5)


def test_rank_is_cross_sectional_within_one_day(pool_members):
    """Rank 是横截面算子，同一天的值必须落在 (0, 1] 且与手动排名一致。"""
    from qlib.data import D

    codes = pool_members("cn_etf")[:30]
    day = SEGMENTS["test"][1]
    df = D.features(codes, ["Rank($volume)"], start_time=day, end_time=day).dropna()
    assert not df.empty
    v = df.iloc[:, 0]
    assert v.min() > 0 and v.max() <= 1.0


def test_all_expressions_align_on_length():
    from qlib.data import D

    code = "SH600519"
    base = D.features([code], ["$close"], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
    for expr in ["Ref($close,1)", "Mean($close,5)", "Std($close,20)", "Max($high,10)", "Min($low,10)"]:
        got = D.features([code], [expr], start_time=HISTORY_START, end_time=SEGMENTS["test"][1])
        assert len(got) == len(base), f"{expr} changed row count"
```

- [ ] **Step 2: 运行**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_02_expression_ops.py -v
```

Expected: 5 passed。

- [ ] **Step 3: 暂存**

```bash
git add tests/test_02_expression_ops.py
```

---

### Task 6: 因子层 Alpha158 / Alpha360

**Files:**
- Create: `tests/test_03_alpha_factors.py`

**Interfaces:**
- Consumes: fixture `qlib_ready`、`settings.SEGMENTS`、`settings.HISTORY_START`
- Produces: fixture `alpha158_handler`（session 级，Task 7 复用，避免重复构建）

- [ ] **Step 1: 写测试**

```python
"""Layer 3: the shipped alpha factor libraries."""

import pytest

from config.settings import HISTORY_START, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


@pytest.fixture(scope="session")
def alpha158_handler():
    from qlib.contrib.data.handler import Alpha158

    return Alpha158(
        instruments="cn_stocks",
        start_time=HISTORY_START,
        end_time=SEGMENTS["valid"][1],
        fit_start_time=HISTORY_START,
        fit_end_time=SEGMENTS["train"][1],
    )


def test_feature_column_count_matches_declared_config(alpha158_handler):
    """用 handler 自己的配置自证，不写死 158 —— 改配置不该误报失败。"""
    feat = alpha158_handler.fetch(col_set="feature")
    _ops, names = alpha158_handler.get_feature_config()
    assert feat.shape[1] == len(names)
    assert list(feat.columns[: len(names)]) == list(names)[: len(feat.columns)]


def test_no_entirely_nan_feature_column(alpha158_handler):
    feat = alpha158_handler.fetch(col_set="feature")
    all_nan = feat.columns[feat.isna().all()].tolist()
    assert not all_nan, f"{len(all_nan)} dead columns, e.g. {all_nan[:8]}"


def test_close_derived_column_matches_raw_data(alpha158_handler):
    """因子矩阵里的价格衍生列必须和数据层读回来的原始值一致。"""
    from qlib.data import D

    feat = alpha158_handler.fetch(col_set="feature")
    code = feat.index.get_level_values("instrument")[0]
    row = feat.iloc[0]
    raw = D.features([code], ["$close"], start_time=row.name[1], end_time=row.name[1])["$close"].iloc[0]
    # CLOSE/1 是原始收盘价本身
    assert abs(float(row["CLOSE/1"]) - float(raw)) < 1e-6 * abs(float(raw)) + 1e-9


def test_label_uses_the_declared_two_day_forward_return(alpha158_handler):
    """Alpha158 的标签按设计前看两天，这是预测目标不是数据泄漏，别当 bug 改。"""
    label = alpha158_handler.fetch(col_set="label")
    assert label.shape[1] == 1
    assert not label.isna().all().iloc[0]


def test_alpha360_shape_is_time_series_windows():
    from qlib.contrib.data.handler import Alpha360

    h = Alpha360(
        instruments="cn_etf",
        start_time=SEGMENTS["train"][0],
        end_time=SEGMENTS["train"][1],
        fit_start_time=SEGMENTS["train"][0],
        fit_end_time=SEGMENTS["train"][1],
    )
    feat = h.fetch(col_set="feature")
    _ops, names = h.get_feature_config()
    assert feat.shape[1] == len(names) and feat.shape[1] > 300
```

- [ ] **Step 2: 运行**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_03_alpha_factors.py -v
```

Expected: 5 passed。Alpha158 全池构建耗时数分钟，这是本工程最慢的一条。若 `test_close_derived_column_matches_raw_data` 找不到列名 `CLOSE/1`，用 `print(feat.columns.tolist()[:12])` 确认真实列名后按实际名称引用 —— **不要**改用 `in feat.columns` 这种"存在就算过"的弱断言。

- [ ] **Step 3: 暂存**

```bash
git add tests/test_03_alpha_factors.py
```

---

### Task 7: 数据集切分层

**Files:**
- Create: `tests/test_04_dataset_split.py`

**Interfaces:**
- Consumes: fixture `qlib_ready`、`alpha158_handler`（Task 6 产出）
- Produces: fixture `dataset`（session 级，Task 8 复用）

- [ ] **Step 1: 写测试**

```python
"""Layer 4: DatasetH segmenting."""

import pytest

from config.settings import SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


@pytest.fixture(scope="session")
def dataset(alpha158_handler):
    from qlib.data.dataset import DatasetH

    return DatasetH(alpha158_handler, segments=SEGMENTS)


def test_each_segment_is_non_empty(dataset):
    for name in SEGMENTS:
        df = dataset.prepare(name, col_set="feature")
        assert not df.empty, f"{name} is empty"


def test_segment_date_ranges_are_disjoint_and_ordered(dataset):
    def _days(seg):
        idx = dataset.prepare(seg, col_set="feature").index.get_level_values("datetime")
        return min(idx), max(idx)

    tr0, tr1 = _days("train")
    va0, va1 = _days("valid")
    te0, te1 = _days("test")
    assert tr1 < va0, "train and valid overlap"
    assert va1 < te0, "valid and test overlap"
    assert tr0 < tr1 < va0 < va1 < te0 < te1


def test_no_instrument_appears_in_two_segments_across_the_boundary(dataset):
    """切分按日期做，标的可以在多个段里重复出现，但它的任一交易日不该跨段。"""
    def _days(seg):
        return set(dataset.prepare(seg, col_set="feature").index.get_level_values("datetime"))

    assert not (_days("train") & _days("valid"))
    assert not (_days("valid") & _days("test"))


def test_feature_and_label_share_the_same_index(dataset):
    feat = dataset.prepare("train", col_set="feature")
    lab = dataset.prepare("train", col_set="label")
    assert feat.index.equals(lab.index)


def test_heldout_tail_absent_from_every_segment(dataset):
    import pandas as pd

    from config.settings import HELDOUT_START

    for name in SEGMENTS:
        days = dataset.prepare(name, col_set="feature").index.get_level_values("datetime")
        assert max(days) < pd.Timestamp(HELDOUT_START), f"{name} leaks into the held-out tail"
```

- [ ] **Step 2: 运行**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_04_dataset_split.py -v
```

Expected: 5 passed。

- [ ] **Step 3: 暂存**

```bash
git add tests/test_04_dataset_split.py
```

---

### Task 8: 模型层 LightGBM

**Files:**
- Create: `tests/test_05_model_train.py`

**Interfaces:**
- Consumes: fixture `qlib_ready`、`dataset`
- Produces: fixture `trained_model`（session 级，Task 9 复用其预测）

- [ ] **Step 1: 写测试**

```python
"""Layer 5: LightGBM training and prediction sanity."""

import numpy as np
import pytest

from config.settings import SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")


@pytest.fixture(scope="session")
def predictions(dataset):
    from qlib.contrib.model.gbdt import LGBModel

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
    return {seg: model.predict(dataset, segment=seg) for seg in ("valid", "test")}


def test_prediction_row_count_matches_input(dataset, predictions):
    for seg in ("valid", "test"):
        n_in = len(dataset.prepare(seg, col_set="feature"))
        assert len(predictions[seg]) == n_in, f"{seg}: prediction count != input sample count"


def test_predictions_are_not_a_constant_column(predictions):
    """常数预测 = 假收敛，模型其实什么都没学到。"""
    for seg, s in predictions.items():
        v = s.iloc[:, 0].dropna()
        assert v.std() > 1e-9, f"{seg}: model predicts a constant"


def test_valid_ic_is_in_a_plausible_band_and_beats_random(dataset, predictions):
    """IC 双向判据。下限排除纯噪声；上限同样重要 —— IC 异常大几乎一定是数据泄漏
    （用了未来函数），那种情况该报警而不是算通过。"""
    import pandas as pd
    from scipy.stats import spearmanr

    labels = dataset.prepare("valid", col_set="label").iloc[:, 0]
    scores = predictions["valid"].iloc[:, 0].reindex(labels.index)
    joined = pd.concat([scores.rename("score"), labels.rename("label")], axis=1).dropna()
    assert len(joined) > 100

    ic = spearmanr(joined["score"], joined["label"]).statistic
    assert not np.isnan(ic)
    assert abs(ic) < 0.5, f"IC={ic} implausibly high — suspect look-ahead leakage"

    rng = np.random.default_rng(7)
    baseline = abs(spearmanr(rng.normal(size=len(joined)), joined["label"]).statistic)
    assert abs(ic) > baseline, f"model IC {ic:.4f} no better than random {baseline:.4f}"
```

- [ ] **Step 2: 运行**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_05_model_train.py -v
```

Expected: 3 passed。|IC| 落在 (0, 0.5) 且优于随机基线。若 `model.predict(dataset, segment=...)` 不接受 `segment` 关键字，读 `.venv/Lib/site-packages/qlib/model/base.py` 确认签名后按实际参数调用，**不要**改成不校验。

- [ ] **Step 3: 暂存**

```bash
git add tests/test_05_model_train.py
```

---

### Task 9: 回测层

**Files:**
- Create: `tests/test_06_backtest.py`

**Interfaces:**
- Consumes: fixture `qlib_ready`、`predictions`（Task 8）
- Produces: `make_strategy(signal) -> dict`、fixture `backtest_result`

- [ ] **Step 1: 写测试**

```python
"""Layer 6: matching engine, account, costs and the report frame."""

import pandas as pd
import pytest

from config.settings import BENCHMARK, SEGMENTS

pytestmark = pytest.mark.usefixtures("qlib_ready")

TOPK = 30
N_DROP = 3


def make_executor():
    return {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }


def make_exchange_kwargs():
    return {
        "limit_threshold": 0.095,
        "deal_price": "close",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
    }


@pytest.fixture(scope="session")
def backtest_result(predictions):
    from qlib.backtest import backtest
    from qlib.contrib.strategy import TopkDropoutStrategy

    signal = predictions["test"].iloc[:, 0].to_frame("score")
    strategy = TopkDropoutStrategy(signal=signal, topk=TOPK, n_drop=N_DROP)
    portfolio, indicator = backtest(
        start_time=SEGMENTS["test"][0],
        end_time=SEGMENTS["test"][1],
        strategy=strategy,
        executor=make_executor(),
        benchmark=BENCHMARK,
        account=100_000_000,
        exchange_kwargs=make_exchange_kwargs(),
    )
    return portfolio, indicator


def test_report_frame_has_return_and_risk_columns(backtest_result):
    portfolio, _ = backtest_result
    assert portfolio, "empty portfolio metrics"
    key = next(iter(portfolio))
    report = portfolio[key][0]
    for col in ["return", "benchmark", "cost"]:
        assert col in report.columns, f"missing {col}"


def test_positions_never_exceed_topk(backtest_result):
    _portfolio, indicator = backtest_result
    key = next(iter(indicator))
    positions = indicator[key][1]
    counts = [len(p) for p in positions if len(p)]
    assert counts, "no positions recorded"
    assert max(counts) <= TOPK, f"held {max(counts)} names, topk is {TOPK}"


def test_turnover_and_cost_are_in_a_sane_range(backtest_result):
    portfolio, _ = backtest_result
    key = next(iter(portfolio))
    report = portfolio[key][0]
    assert (report["cost"].abs() < 0.05).all(), "daily cost > 5% — cost config or matching is wrong"


def test_returns_are_not_identical_to_the_benchmark(backtest_result):
    """完全相等说明回测根本没在做决策。"""
    import numpy as np

    portfolio, _ = backtest_result
    key = next(iter(portfolio))
    report = portfolio[key][0]
    diff = (report["return"].fillna(0.0) - report["benchmark"].fillna(0.0)).abs()
    assert float(diff.max()) > 1e-6, "strategy return == benchmark return everywhere: no real trading"
```

- [ ] **Step 2: 运行**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_06_backtest.py -v
```

Expected: 4 passed。若 `TopkDropoutStrategy` 不接受 DataFrame 形式的 signal，改为传入 `pd.Series`；若 `indicator[key][1]` 结构不同，先 `print(indicator)` 再据实断言 —— **不许**把断言改成 `assert True` 放行。

- [ ] **Step 3: 全量回归**

```bash
./.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: 六层 + 前置全部通过，无 xfail、无 skip（`qlib_data` 已存在）。

- [ ] **Step 4: 暂存**

```bash
git add tests/test_06_backtest.py
```

---

### Task 10: research 全流程脚本

`tests/` 证明功能可用，`research/` 产出真实研究结论。这个脚本是唯一允许写 `artifacts/` 的地方。

**Files:**
- Create: `research/alpha158_lgbm_backtest.py`
- Create: `research/__init__.py`（空文件）

**Interfaces:**
- Consumes: `settings` 全部常量；`qlib.contrib.data.handler.Alpha158`、`qlib.contrib.model.gbdt.LGBModel`、`qlib.data.dataset.DatasetH`、`qlib.backtest.backtest`、`qlib.contrib.strategy.TopkDropoutStrategy`
- Produces（`artifacts/` 下）: `factors.parquet`、`predictions.parquet`、`backtest_report.csv`、`metrics.json`、`FINDINGS.md`

- [ ] **Step 1: 写 `research/alpha158_lgbm_backtest.py`**

```python
"""Full Alpha158 + LightGBM + TopkDropout run on local data. Writes artifacts/ only."""

from __future__ import annotations

import json
import sys
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

TOPK, N_DROP = 50, 3
ACCOUNT = 100_000_000


def main() -> int:
    import qlib
    from qlib.backtest import backtest
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data.dataset import DatasetH
    from scipy.stats import spearmanr

    qlib.init(provider_uri=str(QLIB_DIR), region=REGION)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    handler = Alpha158(
        instruments="cn_stocks",
        start_time=HISTORY_START,
        end_time=SEGMENTS["valid"][1],
        fit_start_time=HISTORY_START,
        fit_end_time=SEGMENTS["train"][1],
    )
    dataset = DatasetH(handler, segments=SEGMENTS)
    dataset.prepare("train", col_set="feature").to_parquet(ARTIFACTS / "factors.parquet")

    model = LGBModel(
        loss="mse", colsample_bytree=0.8879, learning_rate=0.2, subsample=0.8789,
        lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=8,
    )
    model.fit(dataset)

    pred = pd.concat(
        {seg: model.predict(dataset, segment=seg) for seg in ("valid", "test")}, names=["segment"]
    )
    pred.to_parquet(ARTIFACTS / "predictions.parquet")

    labels = dataset.prepare("test", col_set="label").iloc[:, 0]
    scores = pred.xs("test", level="segment").iloc[:, 0].reindex(labels.index)
    joined = pd.concat([scores.rename("score"), labels.rename("label")], axis=1).dropna()
    metrics = {"test_ic": float(spearmanr(joined["score"], joined["label"]).statistic), "n_test_rows": int(len(joined))}

    strategy = TopkDropoutStrategy(
        signal=pred.xs("test", level="segment").iloc[:, 0].to_frame("score"), topk=TOPK, n_drop=N_DROP
    )
    portfolio, _indicator = backtest(
        start_time=SEGMENTS["test"][0],
        end_time=SEGMENTS["test"][1],
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        benchmark=BENCHMARK,
        account=ACCOUNT,
        exchange_kwargs={
            "limit_threshold": 0.095, "deal_price": "close",
            "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
        },
    )
    report = portfolio[next(iter(portfolio))][0]
    report.to_csv(ARTIFACTS / "backtest_report.csv")

    excess = report["return"].fillna(0) - report["benchmark"].fillna(0)
    nav = (1 + report["return"].fillna(0)).cumprod()
    metrics.update(
        annual_return=float(nav.iloc[-1] ** (252 / len(nav)) - 1),
        annual_excess=float(excess.mean() * 252),
        max_drawdown=float((nav / nav.cummax() - 1).min()),
        volatility=float(report["return"].std() * np.sqrt(252)),
    )
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (ARTIFACTS / "FINDINGS.md").write_text(
        f"""# Alpha158 + LightGBM on local A-share daily bars

- Universe: cn_stocks, {HISTORY_START} .. {SEGMENTS['valid'][1]} (fit window ends {SEGMENTS['train'][1]})
- Test window: {SEGMENTS['test'][0]} .. {SEGMENTS['test'][1]}; 2026 data deliberately held out
- Benchmark: {BENCHMARK}; topk={TOPK} n_drop={N_DROP}; account={ACCOUNT:,}
- Test IC: {metrics['test_ic']:.4f} over {metrics['n_test_rows']:,} rows
- Annual return: {metrics['annual_return']:.2%} | annual excess: {metrics['annual_excess']:.2%}
- Max drawdown: {metrics['max_drawdown']:.2%} | volatility: {metrics['volatility']:.2%}

## Interpretation caveats
- The label is `Ref($close,-2)/Ref($close,-1)-1`, a two-day-forward return by design.
- Costs are the configured 5bp open / 15bp close; slippage is not modelled.
- `limit_threshold=0.095` approximates the 10% daily band but cannot model a limit-up that
  genuinely blocks entry — real fills would be worse.
""",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行**

```bash
./.venv/Scripts/python.exe research/alpha158_lgbm_backtest.py
```

Expected: 打印 metrics JSON；`artifacts/` 出现 5 个文件。跑完**必须**人工核看 `FINDINGS.md` 的 `annual_return` 与 `max_drawdown` 是否量级合理（散户实盘的 Alpha158 年化超额通常在个位数百分点；若出现两位数超额加极低回撤，先怀疑数据泄漏而非庆祝）。

- [ ] **Step 3: 暂存**

```bash
git add research/__init__.py research/alpha158_lgbm_backtest.py
```

---

### Task 11: README 与验证结论

把"到底有哪些功能、各自验证结果如何"写成人能读的一页。

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 全部前序任务的实测输出
- Produces: 无

- [ ] **Step 1: 写 `README.md`**

结构固定如下，每节填入实际数字，不留空句：

1. **一句话**：本工程用本机 36 万行真实 A 股日线验证 Qlib 六层闭环。
2. **环境事实表**：Python 3.12.10 / pyqlib 0.9.7 / numpy 2.5.4 / pandas 3.0.6 / 官方 `cp312-win_amd64` wheel，Windows 原生无需 WSL。
3. **Qlib 功能地图**：按 `.py` 文件数列出 `contrib 102 / rl 38 / data 20 / model 18 / workflow 16 / backtest 12 / utils 11 / cli 3 / strategy 2`，每项标注本轮验证状态（已实跑 / 仅导入 / 未覆盖），未覆盖的显式写明原因。
4. **三步上手**：建数据、跑测试、跑研究的确切命令。
5. **六层验证结果表**：每层一条结论 + 实际数字（标的数、IC、年化、回撤）。
6. **两个坑**：`asset_type` 标签自相矛盾（`000001.SH`/`000905.SH`）；非 CN 市场数据会污染交易日历。
7. **下一步消除幸存者偏差**：注明 hs300 成分历史时点快照的位置 `codex/量化分析/data/lake/**/source=user_file_hs300_pit_*`，以及为什么 `cn_stocks` 全池回测会高估收益（今天还在的 626 只股票，是几十年退市筛选之后的幸存者）。这条同时承接 spec §4.5 里「注释写进 cn_index.txt」那个实现不了的决定。
8. **本轮没验证的东西**：qrun 工作流、mlflow、滚动训练、cvxpy 组合优化、PIT、强化学习 —— 明确列出，不含糊。

- [ ] **Step 2: 确认无占位表述**

通读一遍，删除任何"待补充""结果如上"式的空句。每个数字都必须来自实际输出。

- [ ] **Step 3: 暂存**

```bash
git add README.md
```

---

## 自检清单（实现全部完成后）

- [ ] `pytest tests/ -v` 全绿，0 skip 0 xfail
- [ ] `artifacts/FINDINGS.md` 存在且数字被人工判过合理性
- [ ] `量化分析/data` 文件名清单与执行前一致（不比 mtime/size）
- [ ] `git add` 已覆盖 config / tools / tests / research / README；未产生任何 commit
- [ ] spec 的 8 节每条都能指到一个实现任务
