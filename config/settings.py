from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DUCKDB_PATH = Path("D:/Desktop/个人事项/codex/量化分析/data/market.duckdb")
VENDOR_DUMP_BIN = ROOT / "vendor" / "qlib" / "scripts" / "dump_bin.py"
EXPORT_CSV_DIR = ROOT / "tools" / "_export" / "csv"
QLIB_DIR = ROOT / "qlib_data"
ARTIFACTS = ROOT / "artifacts"

REGION = "cn"
FREQ = "day"

# 非 CN 市场的 10 个标的（US/HK/KR/GLOBAL 共约 3.5 万行）会把它们各自的
# 交易日并进 calendars/day.txt，制造出数千个所有 A 股全为 NaN 的幽灵交易日。
SQL_FILTER = "frequency = '1d' AND adjust = 'qfq' AND market = 'CN'"

_SH_STOCK = ("600", "601", "603", "605", "688", "689")
_SH_ETF = ("51", "52", "56", "58")
_SZ_STOCK = ("000", "001", "002", "003", "300", "301", "302")
_SZ_ETF = ("15", "16")
_BJ_STOCK = ("43", "83", "87", "88", "89", "92")


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
        if code.startswith(_BJ_STOCK):
            return "stock"
    return None


def to_qlib_code(symbol: str) -> str:
    code, _, exch = symbol.partition(".")
    return f"{exch.upper()}{code}"


POOLS = {"cn_stocks": "stock", "cn_etf": "etf", "cn_index": "index"}

BENCHMARK = "SH000300"
HISTORY_START = "2016-01-01"

SEGMENTS = {
    "train": ("2018-01-01", "2022-12-31"),
    "valid": ("2023-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
}
