import sqlite3
import json
import os
from pick import pick

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

def print_templates():
    """
    Displays an interactive menu to browse stored decision templates.
    Once a template is selected, it prints its full JSON structure.
    """
    
    if not os.path.exists(DB_PATH):
        print(f"\n[DB ERROR] Database file '{DB_PATH}' not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT decision_type FROM decision_trees")
        rows = cursor.fetchall()

        if not rows:
            print("\n[DB INFO] No templates found in the database.")
            return

        options = [row[0] for row in rows]
        title = "Select a decision template to inspect (Use arrows, press ENTER):"
        
        option, index = pick(options, title, indicator='=>', default_index=0)

        cursor.execute("SELECT variables, tree FROM decision_trees WHERE decision_type = ?", (option,))
        result = cursor.fetchone()

        if result:
            variables = result[0]
            tree_obj = json.loads(result[1])

            print(f"\n" + "█" * 80)
            print(f" SELECTED TEMPLATE: {option.upper()}")
            print(f" DEFINED VARIABLES: {variables}")
            print("█" * 80 + "\n")
            
            print(json.dumps(tree_obj, indent=4))
            
            print("\n" + "█" * 80)
            print(" End of Template View.")

    except sqlite3.Error as e:
        print(f"[DB ERROR] SQLite error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error during navigation: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    print("--- Template in db ---")
    print_templates()