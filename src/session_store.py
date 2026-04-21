import json

store: dict[str, dict] = {}


def save_template(decision_type: str, variables: list[str], tree: dict) -> None:
    """Save a parameterized tree template. IF values are zeroed before storing."""
    def zero_if(node: dict) -> dict:
        node = dict(node)
        node["favor"]    = 0.0
        node["neutral"]  = 0.0
        node["unfavor"]  = 0.0
        node["children"] = [zero_if(c) for c in node.get("children", [])]
        return node

    key = decision_type.lower().strip()
    store[key] = {
        "variables": variables,
        "tree":      zero_if(tree),
    }
    print(f"[Session] Template saved for '{key}'")


def load_template(decision_type: str) -> dict | None:
    """Return stored template or None if not found in this session."""
    return store.get(decision_type.lower().strip())


def list_templates() -> list[str]:
    """Return all decision types stored in this session."""
    return list(store.keys())


def print_templates() -> None:
    """Print all templates stored in the current session."""
    if not store:
        print("\n[Session] No templates stored in this session.")
        return

    for key, data in store.items():
        print(f"\n{'█' * 80}")
        print(f" TEMPLATE: {key.upper()}")
        print(f" VARIABLES: {data['variables']}")
        print("█" * 80)
        print(json.dumps(data["tree"], indent=4))
        print("█" * 80)
        print(" End of Template View.")
