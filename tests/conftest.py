import duckdb
import pytest

from mmi.utils.db import init_schemas


@pytest.fixture
def con():
    """In-memory DuckDB with medallion schemas created."""
    c = duckdb.connect(":memory:")
    init_schemas(c)
    yield c
    c.close()
