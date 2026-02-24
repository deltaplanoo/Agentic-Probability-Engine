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
            # Tavily fornisce 'content', che è il testo estratto dalla pagina
            formatted_output += f"{i}. {r['title']}\n   Content: {r['content'][:300]}...\n\n"
            
        return formatted_output

    except Exception as e:
        return f"Error during Tavily search: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)