import asyncio
import json
from fastmcp import Client

MCP_URL = "http://localhost:8000/mcp"

# ── Test cases ────────────────────────────────────────────────────────────────
# text, city, province
TEST_QUERIES = [
    ("Via Calzaiuoli 50",         "Firenze",    "Firenze"),
    ("Gelatando",                 "Scandicci",  "Firenze"),
    ("Piazzale Michelangelo",     "Firenze",    "Firenze"),
    ("Colosseo",                  "Roma",       "Roma"),
    ("Stazione Bologna Centrale", "Bologna",    "Bologna"),
]
# ─────────────────────────────────────────────────────────────────────────────


def _extract_coords(features: list) -> tuple[float, float] | None:
    """Return (lat, lon) from the first GeoJSON feature that has coordinates."""
    for feat in features:
        coords = feat.get("geometry", {}).get("coordinates", [])
        if len(coords) >= 2:
            return coords[1], coords[0]   # GeoJSON is [lon, lat]
    return None


async def run_test():
    print(f"{'='*60}")
    print(f"  geocode_with_city — MCP Tool Test")
    print(f"  Server: {MCP_URL}")
    print(f"{'='*60}\n")

    client = Client(MCP_URL)
    await client.__aenter__()

    try:
        queries = [
            {"text": text, "city": city, "province": province, "maxresults": 5}
            for text, city, province in TEST_QUERIES
        ]

        print(f"[TEST] Sending {len(queries)} queries to geocode_with_city...\n")

        mcp_res = await client.call_tool(
            "geocode_with_city",
            arguments={"queries": queries}
        )

        raw = mcp_res.content[0].text
        data = json.loads(raw)

        top_error = data.get("error")
        if top_error:
            print(f"[ERROR] Top-level tool error: {top_error}")
            return

        results = data.get("results", [])

        print(f"  {'QUERY':<35}  {'LAT':>10}  {'LON':>11}  ADDRESS")
        print(f"  {'-'*90}")

        for item in results:
            query_label = f"{item['text']}, {item['city']}"
            item_error  = item.get("error")
            features    = item.get("results") or []

            if item_error:
                print(f"  {query_label:<35}  {'ERROR':>10}           {item_error}")
                continue

            if not features:
                print(f"  {query_label:<35}  {'no results':>10}")
                continue

            coords = _extract_coords(features)
            if coords:
                lat, lon = coords
                addr = features[0].get("properties", {}).get("address", "—")
                print(f"  {query_label:<35}  {lat:>10.6f}  {lon:>11.6f}  {addr}")
            else:
                print(f"  {query_label:<35}  {'no coords':>10}")

        print(f"\n  Raw feature count per query:")
        for item in results:
            n = len(item.get("results") or [])
            print(f"    {item['text']}, {item['city']}: {n} feature(s)")

    except Exception as e:
        print(f"[FATAL] {e}")
    finally:
        await client.__aexit__(None, None, None)
        print(f"\n{'='*60}")
        print(f"  Test completed.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception as e:
        print(f"\n[FATAL] Could not run test: {e}")
        print("Make sure the MCP server is running:  python src/snap4agentic_advisor_experimental.py")