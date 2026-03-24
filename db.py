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
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_type TEXT NOT NULL UNIQUE,  -- e.g. "open a restaurant"
                tree        TEXT NOT NULL,            -- JSON, weights only, IF zeroed
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_tree(decision_type: str, tree: dict):
    """Save or overwrite a tree for a given decision type."""
    def zero_if(node):
        """Recursively strip IF values before storing, keep weights."""
        node = dict(node)
        node["favor"]   = 0.0
        node["neutral"] = 0.0
        node["unfavor"] = 0.0
        node["children"] = [zero_if(c) for c in node.get("children", [])]
        return node
