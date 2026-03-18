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
def generate_decision_tree(goal: str, factors: list[dict]) -> str:
    """
    Generates a structured decision tree for a specific goal.
    'factors' should be a list of dicts with 'label' and 'weight' (-1 to +1).
    """
    tree = {
        "decision": goal,
        "criteria": factors,
        "status": "initialized"
    }
    
    #FIXME: save tree to db for faster retrieval later
    return json.dumps(tree, indent=2)

@mcp.tool()
def process_decision_tree(tree_structure: str) -> str:
    """
    Calculates the values of a decision tree from leaves to root using weighted averages.
    Input must be a JSON string with: id, label, weight, value, children.
    """   
    def calculate_node(node):
        children = node.get("children", [])
        
        # if leaf: end recursion
        if not children:
            return node.get("value", 0.0)
        
        weighted_sum = 0.0
        sum_of_weights = 0.0
        
        for child in children:
            child_value = calculate_node(child)
            weight = child.get("weight", 0.0)
            
            weighted_sum += (child_value * weight)
            sum_of_weights += abs(weight)
            
        # father node value
        if sum_of_weights == 0:
            node_value = 0.0
        else:
            node_value = weighted_sum / sum_of_weights
            
        node["value"] = round(node_value, 4)
        return node_value

    try:
        data = json.loads(tree_structure)
        calculate_node(data)
        return json.dumps(data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Invalid tree structure: {str(e)}"})

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)