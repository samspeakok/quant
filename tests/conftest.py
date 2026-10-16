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
