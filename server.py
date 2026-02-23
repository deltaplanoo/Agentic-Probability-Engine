import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from googleapiclient.discovery import build

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

mcp = FastMCP("Server")

@mcp.tool()
def web_search(query: str) -> str:
    """
    Searches the web for information relevant to the decision.
    """
    print(f"[LOG SERVER] Searching key factors for decision: {query}")

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CX, num=5, gl="it").execute()
        results = res.get("items", [])
        
        if not results:
            return "This query returned no results."

        formatted_output = "Search results:\n"
        for i, item in enumerate(results, 1):
            title = item.get("title")
            snippet = item.get("snippet")
            link = item.get("link")
            formatted_output += f"{i}. {title}\n   Description: {snippet}\n   Source: {link}\n\n"
            
        return formatted_output
    except Exception as e:
        return f"An error occurred during web search: {e}"

if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)