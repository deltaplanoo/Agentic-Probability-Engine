import json
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
import requests
from tavily import TavilyClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

mcp = FastMCP("Server")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

@mcp.tool()
def web_search(query: str) -> str:
    """
    Searches the web for information relevant to the decision.
    """
    print(f"[LOG SERVER] Searching key factors for decision: {query}")
    extended_query = f"Fattori per decidere: {query}"

    try:
        response = tavily.search(
            query=extended_query, 
            search_depth="basic", 
            max_results=10,
            include_answer=True
        )
        
        results = response.get("results", [])
        if not results:
            return "No results found."

        formatted_output = f"Tavily answer: {response.get('answer', 'N/A')}\n\n"
        formatted_output += "Detailed results:\n"
        
        for i, r in enumerate(results, 1):
            # Tavily returns 'content' of webpage
            formatted_output += f"{i}. {r['title']}\n   Content: {r['content'][:300]}...\n\n"
            
        return formatted_output

    except Exception as e:
        return f"Error during Tavily search: {str(e)}"
    
@mcp.tool()
def geocode_address(address: str) -> str:
    """
    Converts a textual address or location name into Latitude and Longitude.
    Example input: 'Piazza della Signoria, Firenze'
    """

    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "Agentic-Probability-Engine"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not data:
                return json.dumps({"error": f"Address '{address}' not found."})
            
            result = {
                "address": data[0].get("display_name"),
                "lat": float(data[0].get("lat")),
                "lon": float(data[0].get("lon"))
            }
            return json.dumps(result)
        else:
            return json.dumps({"error": f"Geocoding service error: {response.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Exception during geocoding: {str(e)}"})

@mcp.tool()
def get_poi_nearby(search_term: str, lat: float, lon: float, max_dist_km: float) -> str:
    """
    Queries the Snap4City SuperServiceMap API for POIs (Restaurants, Parking, etc.).
    Returns the total count of specific venue names.
    """
    position = f"{lat};{lon}"
    base_url = "https://www.snap4city.org/superservicemap/api/v1/location/"
    
    params = {
        "position": position,
        "search": search_term,
        "maxDists": max_dist_km,
        "maxResults": 50000
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            total_count = data.get("count", 0)
            
            if total_count == 0:
                return f"No results found for '{search_term}' in this area."

            result_text = f"Total count of '{search_term}' in the area: {total_count}\n"
            return result_text
        else:
            return f"Snap4City API Error: {response.status_code}"
    except Exception as e:
        return f"Exception during Snap4City lookup: {str(e)}"
    
@mcp.tool()
def process_decision_tree(tree_structure: str) -> str:
    """
    Calculates Italian Flag triplets (favor, neutral, unfavor) for every
    non-leaf node by averaging children triplets, weighted only by node weight.
    IF values themselves are plain probabilities — no internal weighting.
    """
    def calculate_node(node):
        children = node.get("children", [])

        # Leaf: triplet already set, just return it as-is
        if not children:
            return (
                node.get("favor",   0.0),
                node.get("neutral", 0.0),
                node.get("unfavor", 0.0),
            )

        total_weight = sum(abs(child.get("weight", 0.0)) for child in children)

        favor   = 0.0
        neutral = 0.0
        unfavor = 0.0

        for child in children:
            f, n, u = calculate_node(child)
            w = abs(child.get("weight", 0.0)) / total_weight if total_weight > 0 else 1 / len(children)
            favor   += f * w
            neutral += n * w
            unfavor += u * w

        node["favor"]   = round(favor,   4)
        node["neutral"] = round(neutral, 4)
        node["unfavor"] = round(unfavor, 4)
        return favor, neutral, unfavor

    try:
        data = json.loads(tree_structure)
        calculate_node(data)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Invalid tree structure: {str(e)}"})

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)