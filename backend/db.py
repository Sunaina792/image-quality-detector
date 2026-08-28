"""SQLite persistence for analysis results."""

import datetime
import json
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "analyses.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            created_at TEXT,
            quality_score REAL,
            quality_label TEXT,
            result_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_analysis(filename: str, result: dict) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO analyses (filename, created_at, quality_score, quality_label, result_json) VALUES (?, ?, ?, ?, ?)",
        (
            filename,
            datetime.datetime.utcnow().isoformat(),
            result["quality_score"],
            result["quality_label"],
            json.dumps(result),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_history(limit: int = 50) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, filename, created_at, quality_score, quality_label FROM analyses ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis(analysis_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["result"] = json.loads(d.pop("result_json"))
    return d
