import json
import httpx
from tavily import tavily

def register_tools(mcp, tavily_client=None):
    """
    Registra i tool di ricerca, albero decisionale e geocodifica 
    sull'istanza MCP passata come argomento.
    """

    @mcp.tool()
    def web_search(query: str) -> str:
        """
        Searches the web for information relevant to the decision.
        """
        print(f"[LOG SERVER] Searching key factors for decision: {query}")
        extended_query = f"Fattori per decidere: {query}"

        try:
            client = tavily_client if tavily_client else tavily
            response = client.search(
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
                formatted_output += f"{i}. {r['title']}\n   Content: {r['content'][:300]}...\n\n"
                
            return formatted_output

        except Exception as e:
            return f"Error during Tavily search: {str(e)}"

    @mcp.tool()
    def process_decision_tree(tree_structure: str) -> str:
        """
        Calculates Italian Flag triplets (favor, neutral, unfavor) for every
        non-leaf node by averaging children triplets, weighted only by node weight.
        IF values themselves are plain probabilities — no internal weighting.
        """
        def calculate_node(node):
            children = node.get("children", [])

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

    @mcp.tool()
    async def geocode_nominatim(address: str, city: str = "", province: str = "") -> dict:
        """
        Geocodes an address using the Nominatim OpenStreetMap API.
        Returns GeoJSON-style result.
        """
        query_parts = [address]
        if city: query_parts.append(city)
        if province: query_parts.append(province)
        full_query = ", ".join(query_parts)

        params = {
            "q": full_query,
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": 1
        }
        
        headers = {
            "User-Agent": "AgenticProbabilityEngine",
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return {"error": "No results found"}

            result = data[0]
            return {
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
                "display_name": result.get("display_name"),
                "raw": result
            }