from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DUCKDB_PATH = Path("D:/Desktop/个人事项/codex/量化分析/data/market.duckdb")
VENDOR_DUMP_BIN = ROOT / "vendor" / "qlib" / "scripts" / "dump_bin.py"
EXPORT_CSV_DIR = ROOT / "tools" / "_export" / "csv"
QLIB_DIR = ROOT / "qlib_data"
ARTIFACTS = ROOT / "artifacts"

REGION = "cn"
FREQ = "day"

# 非 CN 市场的 10 个标的（US/HK/KR/GLOBAL）会把它们各自的交易日并进 calendars/day.txt，
# 制造出所有 A 股全为 NaN 的幽灵交易日，所以必须按 market='CN' 过滤。
# 行数实测于 2026-10-16（只读查询）：全库 465,013 行，其中非 CN 45,413 行、
# CN+1d+qfq 314,683 行（再叠加 HISTORY_START 之后落到导出闸门的 194,738 行）。
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
