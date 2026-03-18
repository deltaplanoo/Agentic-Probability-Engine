import json
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
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
def process_decision_tree(tree_structure: str) -> str:
    """
    Calculates Italian Flag triplets (favor, neutral, unfavor) for every
    non-leaf node using weighted averages of children, bottom-up to root.
    """
    def calculate_node(node):
        children = node.get("children", [])

        # Leaf: triplet already set, just return it
        if not children:
            return (
                node.get("favor",   0.0),
                node.get("neutral", 0.0),
                node.get("unfavor", 0.0),
            )

        total_weight = 0.0
        w_favor   = 0.0
        w_neutral = 0.0
        w_unfavor = 0.0

        for child in children:
            f, n, u = calculate_node(child)
            w = abs(child.get("weight", 0.0))
            w_favor   += f * w
            w_neutral += n * w
            w_unfavor += u * w
            total_weight += w

        if total_weight > 0:
            favor   = round(w_favor   / total_weight, 4)
            neutral = round(w_neutral / total_weight, 4)
            unfavor = round(w_unfavor / total_weight, 4)
        else:
            favor, neutral, unfavor = 0.0, 1.0, 0.0

        node["favor"]   = favor
        node["neutral"] = neutral
        node["unfavor"] = unfavor
        return favor, neutral, unfavor

    try:
        data = json.loads(tree_structure)
        calculate_node(data)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Invalid tree structure: {str(e)}"})

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)