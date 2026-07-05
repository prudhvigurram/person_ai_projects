"""
SQLite storage for side-effect tools (tickets, refunds, cancellations, escalations).

Uses a single file at data/support.db. Idempotent schema creation.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "support.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id      TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL,
    order_id       TEXT,
    subject        TEXT,
    category       TEXT,
    details        TEXT,
    status         TEXT DEFAULT 'open',
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id      TEXT PRIMARY KEY,
    order_id       TEXT NOT NULL,
    customer_id    TEXT NOT NULL,
    amount         REAL NOT NULL,
    reason         TEXT,
    status         TEXT DEFAULT 'processed',
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cancellations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id       TEXT NOT NULL,
    customer_id    TEXT NOT NULL,
    reason         TEXT,
    cancelled_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id  TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL,
    order_id       TEXT,
    reason         TEXT NOT NULL,
    priority       TEXT DEFAULT 'medium',
    status         TEXT DEFAULT 'pending',
    created_at     TEXT NOT NULL,
    resolved_at    TEXT
);
"""


def init_db():
    """Create tables if they don't exist. Idempotent — safe to call every startup."""
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_db():
    """Context manager: opens connection, commits on success, rolls back on error, closes always."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets you access columns by name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query_all(table: str, limit: int = 100) -> list[dict]:
    """Quick read helper — for debugging/testing."""
    with get_db() as conn:
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


if __name__ == '__main__':
    init_db()
    print(f"Database initialized at {DB_PATH}")
    print(f"Tables: {[t for t in ['tickets', 'refunds', 'cancellations', 'escalations']]}")