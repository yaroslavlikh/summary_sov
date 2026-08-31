from contextlib import contextmanager

import psycopg2
from psycopg2 import pool as pg_pool

from config import get_database_url

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(1, 10, dsn=get_database_url())
    return _pool


@contextmanager
def get_conn():
    conn = _get_pool().getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)
