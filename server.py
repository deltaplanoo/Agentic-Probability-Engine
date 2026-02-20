from fastmcp import FastMCP


mcp = FastMCP("Server")


@mcp.tool()
def saluta_da_server(args: str = "Argomento Server") -> str:
	"""Restituisce un saluto dal Server"""

	msg = f"Ciao, Il server è acceso!, questi sono gli argomenti: {args}"
	print(f"[LOG SERVER] Eseguito tool con args: {args}")
	
	return msg


if __name__ == "__main__":
	mcp.run(transport="sse", port=8000)
