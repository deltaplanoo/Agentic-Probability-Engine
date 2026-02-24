import asyncio
from fastmcp import Client
from rich import print

server_url = "http://localhost:8000/sse"

async def main():
    print("Connessione ai server in corso...")
    
    client = Client(server_url)
    
    async with client:
        print("[green] Connesso \n")
        
        mcp_tools_list = await client.list_tools()
        print(mcp_tools_list)
        print("-" * 50)
        
        user_query = "Should I open a restaurant in via dei Calzaiuoli 50 a Firenze?"

        try:
            risultato = await client.call_tool("web_search", arguments={"query": user_query})
            print(f"[green]Risposta dal Server:[/green] {risultato} \n")

        except Exception as e:
            print(f"[red]Errore Server:[/red] {e}")

if __name__ == "__main__":
    asyncio.run(main())