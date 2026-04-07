from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

async def get_mcp_tools(mcp_client):
    """
    Retrieves MCP tools and wraps them as LangChain tools, 
    perfectly aligned with server.py definitions.
    """
    
    def make_wrapper(name):
        async def wrapper(**kwargs) -> str:
            result = await mcp_client.call_tool(name, arguments=kwargs)
            return result.content[0].text
        return wrapper

    # --- Input Schema ---

    class WebSearchInput(BaseModel):
        query: str = Field(description="The search query to look up")
        
    class GeocodeInput(BaseModel):
        address: str = Field(description="The textual address or location name to geocode")

    class GetPoiNearbyInput(BaseModel):
        search_term: str = Field(description="POI category or search term (e.g., Restaurant, Parking)")
        lat: float = Field(description="Latitude")
        lon: float = Field(description="Longitude")
        max_dist_km: float = Field(default=2.0, description="Search radius in kilometers")

    class ProcessTreeInput(BaseModel):
        tree_structure: str = Field(description="The JSON string of the decision tree to calculate")
        
	# --- Tool Mapping ---

    tools = {
        "web_search": StructuredTool.from_function(
            name="web_search",
            description="Search the web for general information",
            coroutine=make_wrapper("web_search"),
            args_schema=WebSearchInput,
        ),
        "geocode_address": StructuredTool.from_function(
            name="geocode_address",
            description="Convert a location name or address into numerical coordinates (lat/lon)",
            coroutine=make_wrapper("geocode_address"), # Deve matchare @mcp.tool()
            args_schema=GeocodeInput,
        ),
        "get_poi_nearby": StructuredTool.from_function(
            name="get_poi_nearby",
            description="Retrieve POI count from Snap4City near coordinates",
            coroutine=make_wrapper("get_poi_nearby"), # Deve matchare @mcp.tool()
            args_schema=GetPoiNearbyInput,
        ),
        "process_decision_tree": StructuredTool.from_function(
            name="process_decision_tree",
            description="Calculate the Italian Flag values for the entire tree structure",
            coroutine=make_wrapper("process_decision_tree"),
            args_schema=ProcessTreeInput,
        )
    }
    
    return tools