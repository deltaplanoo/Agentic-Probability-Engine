import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "trees.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_trees (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_type TEXT NOT NULL UNIQUE,
                variables     TEXT NOT NULL,
                tree          TEXT NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_template(decision_type: str, variables: list[str], tree: dict):
    """Save a parameterized tree template. IF values are zeroed before storing."""
    def zero_if(node):
        node = dict(node)
        node["favor"]    = 0.0
        node["neutral"]  = 0.0
        node["unfavor"]  = 0.0
        node["children"] = [zero_if(c) for c in node.get("children", [])]
        return node

    clean_tree = zero_if(tree)
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO decision_trees (decision_type, variables, tree, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(decision_type) DO UPDATE SET
                variables  = excluded.variables,
                tree       = excluded.tree,
                updated_at = CURRENT_TIMESTAMP
        """, (
            decision_type.lower().strip(),
            json.dumps(variables),
            json.dumps(clean_tree)
        ))
        conn.commit()
    print(f"[DB] Template saved for '{decision_type}'")

def load_template(decision_type: str) -> dict | None:
    """Search and return stored template or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT variables, tree FROM decision_trees WHERE decision_type = ?",
            (decision_type.lower().strip(),)
        ).fetchone()
    if not row:
        return None
    return {
        "variables": json.loads(row[0]),
        "tree":      json.loads(row[1])
    }

def list_templates() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT decision_type, variables, created_at, updated_at "
            "FROM decision_trees ORDER BY updated_at DESC"
        ).fetchall()
    return [
        {
            "decision_type": r[0],
            "variables":     json.loads(r[1]),
            "created_at":    r[2],
            "updated_at":    r[3]
        }
        for r in rows
    ]

def print_templates():
    templates = list_templates()
    if not templates:
        print("No templates saved yet.")
        return

    print(f"\n{'='*55}")
    print(f" SAVED DECISION TREE TEMPLATES ({len(templates)} total)")
    print(f"{'='*55}")

    for t in templates:
        print(f"\n  Decision type : {t['decision_type']}")
        print(f"  Variables     : {', '.join(t['variables'])}")
        print(f"  Created       : {t['created_at']}")
        print(f"  Updated       : {t['updated_at']}")

    print(f"\n{'='*55}\n")