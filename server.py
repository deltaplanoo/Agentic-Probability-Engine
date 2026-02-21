from fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("Server")

@mcp.tool()
def web_search(query: str) -> str:
    """
    Searches the web for information relevant to the decision.
    """
    print(f"[LOG SERVER] Searching key factors for decision: {query}")
    
    with DDGS() as ddgs:
        # top 5 results
        extended_query = f"What factors should I consider when making this decision: {query}"
        results = ddgs.text(extended_query, max_results=5)
        
        if not results:
            return "This query returned no results."

        formatted_output = "Web search results:\n"
        for i, r in enumerate(results, 1):
            formatted_output += f"{i}. {r['title']}: {r['body']}\n"
            
        return formatted_output

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)