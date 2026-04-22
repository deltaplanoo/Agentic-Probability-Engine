import math
import re
import httpx
import json
from fastmcp import FastMCP
from datetime import datetime, timezone
from urllib.parse import quote
import shapely.wkt
import shapely.geometry
from typing import Annotated, Literal, Dict, Any, Union, Optional, List
from pydantic import Field, BaseModel
import asyncio
import os
from enum import Enum
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ######## Autentication ########
# import os
# from dotenv import load_dotenv
# from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

# load_dotenv()

# token = os.getenv("TOKEN_snap4Agentic_Advisor_native")
# if not token:
#     raise ValueError("Error: token not found")

# tokens_config = {
#     token: {
#         "client_id": "agente-test",
#         "scopes": []
#     }
# }
# auth = StaticTokenVerifier(tokens=tokens_config)

############ Utility #############

class ToolResponse(BaseModel):
    """Struttura standard e blindata per tutte le risposte dei tool."""
    results: Optional[Any] = None
    error: Optional[str] = None
    total: Optional[int] = None

def create_success(data: Any, total: int = None) -> dict:
    """Crea una risposta di successo standardizzata."""
    return ToolResponse(results=data, total=total).model_dump()

def create_error(message: str) -> dict:
    """Crea una risposta di errore standardizzata."""
    return ToolResponse(error=message).model_dump()

def _load_data(filename: str) -> list:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "resources", filename)
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

REGIONI_DATA = _load_data("regioni_optimized_wkt.json")
PROVINCE_DATA = _load_data("province_optimized_wkt.json")
COMUNI_DATA = _load_data("comuni_optimized_wkt.json")

# Cache per l'autocompletamento in caso di errore dell'agente (calcolate a costo zero all'avvio)
VALID_REGIONS = sorted(list({r["proprieta"].get("REGION_NAME", "") for r in REGIONI_DATA if r.get("proprieta", {}).get("REGION_NAME")}))
VALID_PROVINCES = sorted(list({p["proprieta"].get("PROVINCE_NAME", "") for p in PROVINCE_DATA if p.get("proprieta", {}).get("PROVINCE_NAME")}))

ItalianRegion = Literal[
    "Piemonte", "Valle d'Aosta", "Lombardia", "Trentino-Alto Adige", "Veneto",
    "Friuli-Venezia Giulia", "Liguria", "Emilia-Romagna", "Toscana", "Umbria",
    "Marche", "Lazio", "Abruzzo", "Molise", "Campania", "Puglia", "Basilicata",
    "Calabria", "Sicilia", "Sardegna"
]

##################################

# Maximum results returned by any spatial search tool.
# Intentionally low: larger payloads degrade LLM attention and inflate context.
MAX_RESULTS = 15

# Fields present in every snap4city feature that carry no useful information
_NOISE_PROPS = set()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two GPS points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _clean_props(props: dict) -> dict:
    """Strip noise fields from a feature's properties dict."""
    return {k: v for k, v in props.items() if k not in _NOISE_PROPS}


def _extract_api_error(resp) -> str:
    """Extract a clean error message from a failed API response, stripping URL noise."""
    try:
        err = resp.json()
        msg = err.get('message') or err.get('error') or err.get('error_message') or ''
        if msg:
            msg = re.sub(r'https?://\S+', '', str(msg)).strip()
            return msg
    except Exception:
        pass
    return f"HTTP {resp.status_code}"


def _extract_features(raw_response: dict, typeQuery: str = None, max_results: int = MAX_RESULTS) -> tuple:
    """
    Flatten the snap4city API response into a list of clean feature dicts.
    Returns (features_list, total_count).
    Handles both entity (direct FeatureCollection) and standard (grouped-by-category) responses.
    """
    all_feats = []
    if typeQuery == "entity":
        all_feats = raw_response.get("features", [])
    else:
        for cat_data in raw_response.values():
            if isinstance(cat_data, dict) and "features" in cat_data:
                all_feats.extend(cat_data["features"])

    total = len(all_feats)
    features = []
    for feat in all_feats[:max_results]:
        if "properties" in feat:
            feat = dict(feat)
            feat["properties"] = _clean_props(feat["properties"])
        features.append(feat)
    return features, total

def _read_dynamic_categories() -> dict:
    """Legge il JSON delle categorie dal disco in tempo reale."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "resources", "service_categories.json")
    
    if not os.path.exists(file_path):
        return {}
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _read_iot_filters_map() -> dict:
    """Legge il file generato dallo script notturno"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "resources", "iot_filters_map.json")
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception: return {}
# -----------------------------------------------------------------------------------------

mcp = FastMCP("snap4Agentic_Advisor_experimental") #, auth = auth

tavily = TavilyClient(api_key=TAVILY_API_KEY)

@mcp.tool(name="get_region_boundary", tags={"locator"}, meta={"tags": ["locator"]})
def get_region_boundary(
    region: Annotated[ItalianRegion, Field(description="Select the exact Italian region. ONLY IN ITALIAN")]
) -> Dict[str, Any]:
    """Returns the WKT boundary of an Italian region. region ONLY IN ITALIAN"""
    search = region.lower()
    for r in REGIONI_DATA:
        if r.get("proprieta", {}).get("REGION_NAME", "").lower() == search:
            return create_success(f"Region: {region}\nWKT: {r['wkt']}")
    return create_error(f"Error: Data for region {region} not found.")

@mcp.tool(name="get_province_boundary", tags={"locator"}, meta={"tags": ["locator"]})
def get_province_boundary(
    province: Annotated[str, Field(description="Exact name of the Province or Metropolitan City (e.g., 'Torino'). ONLY IN ITALIAN")]
) -> Dict[str, Any]:
    """Returns the WKT boundary of an Italian province. province ONLY IN ITALIAN"""
    search = province.lower()
    for p in PROVINCE_DATA:
        name = p.get("proprieta", {}).get("PROVINCE_NAME", "")
        if name.lower() == search:
            return create_success(f"Province: {name}\nWKT: {p['wkt']}")
            
    return create_error(f"Province '{province}' not found. Valid options: {', '.join(VALID_PROVINCES)}")

def _lookup_municipality(province: str, municipality: str):
    """Shared lookup: returns (mun_name, prov_name, raw_wkt) or raises a dict error."""
    search_prov = province.lower()
    search_mun = municipality.lower()
    found_muns = []

    for c in COMUNI_DATA:
        props = c.get("proprieta", {})
        prov_name = props.get("PROVINCE_NAME", "")

        if prov_name.lower() == search_prov:
            mun_name = props.get("MUNICIPALITY_NAME", "")
            found_muns.append(mun_name)

            if mun_name.lower() == search_mun:
                return mun_name, prov_name, c['wkt']

    if not found_muns:
        return None, None, {"error": f"Province '{province}' not found. Valid options: {', '.join(VALID_PROVINCES)}"}

    found_muns.sort()
    return None, None, {"error": f"Municipality '{municipality}' not found in '{province}'. Valid options: {', '.join(found_muns)}"}

@mcp.tool(name="get_municipality_info", tags={"locator"}, meta={"tags": ["locator"]})
def get_municipality_info(
    province: Annotated[str, Field(description=(
        "Italian name of the Province or Metropolitan City (NOT the region). "
        "Examples: 'Firenze' (NOT 'Tuscany' or 'Toscana' — those are the region, not the province), "
        "'Roma' (NOT 'Lazio'), 'Milano' (NOT 'Lombardia'). "
        "If the tool returns 'not found', READ the 'Valid options' list in the error and pick the correct name from it."
    ))],
    municipality: Annotated[str, Field(description=(
        "Italian name of the municipality. Examples: 'Firenze' (NOT 'Florence'), 'Roma' (NOT 'Rome'). "
        "If the tool returns 'not found', READ the 'Valid options' list in the error and pick the correct name from it."
    ))]
) -> Dict[str, Any]:
    """
    Returns lightweight geographic metadata for a municipality: bounding box, centroid coordinates, and approximate radius.

    Fields returned: 'municipality', 'province', 'bbox' [min_lon, min_lat, max_lon, max_lat],
    'center_lat', 'center_lon' (centroid), 'radius_km'.

    CRITICAL: 'province' is the Italian PROVINCE name (e.g., 'Firenze', 'Roma', 'Milano'),
    NOT the region (e.g., 'Toscana', 'Lazio', 'Lombardia' are WRONG).
    If the tool returns an error, READ the 'Valid options' list carefully and retry with the correct name.

    Use get_municipality_wkt ONLY when a tool explicitly requires a WKT polygon string (e.g., service_search_within_polygon).
    """
    mun_name, prov_name, raw_wkt = _lookup_municipality(province, municipality)
    if mun_name is None:
        return {"results": None, **raw_wkt}

    try:
        geometry = shapely.wkt.loads(raw_wkt)
        bbox = list(geometry.bounds)
        centroid = geometry.centroid
        center_lat = round(centroid.y, 6)
        center_lon = round(centroid.x, 6)
        radius_km = round(_haversine_km(center_lat, center_lon, bbox[3], bbox[2]), 2)
    except Exception:
        bbox = None
        center_lat = None
        center_lon = None
        radius_km = None

    return {
        "results": {
            "municipality": mun_name,
            "province": prov_name,
            "bbox": bbox,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km
        },
        "error": None
    }

@mcp.tool(name="get_municipality_wkt", tags={"locator"}, meta={"tags": ["locator"]})
def get_municipality_wkt(
    province: Annotated[str, Field(description="Province or Metropolitan City the municipality belongs to. ONLY IN ITALIAN (e.g Roma NOT RO)")],
    municipality: Annotated[str, Field(description="Exact name of the municipality. ONLY IN ITALIAN (e.g (Roma))")]
) -> Dict[str, Any]:
    """
    Returns the full WKT polygon string for a municipality boundary.
    Use ONLY when a tool explicitly requires a WKT polygon (e.g., service_search_within_polygon).
    Do NOT use this for IoT sensor searches, GPS area searches, or any workflow using bbox/radius —
    those only need get_municipality_info (bbox, center_lat, center_lon, radius_km).
    """
    mun_name, prov_name, raw_wkt = _lookup_municipality(province, municipality)
    if mun_name is None:
        return {"results": None, **raw_wkt}

    return {
        "results": {
            "municipality": mun_name,
            "province": prov_name,
            "wkt": raw_wkt
        },
        "error": None
    }

@mcp.tool(name="get_service_categories_old", tags = {"shared"}, meta={"tags": ["shared"]})
async def get_service_categories_old(
    mode: Annotated[
        Literal["macro", "detailed"], 
        Field(
            default="macro",
            description=(
                "Determines the granularity of the categories returned. "
                "'macro' returns only main high-level categories (e.g., Healthcare). "
                "'detailed' returns specific subcategory (e.g., Hospital)."
            )
        )
    ] = "macro"
) -> Dict[str, Any]:
    """
    Retrieves the list of valid Service Categories from the Snap4City Knowledge Base.
    Use this tool to find the correct category names required for search tools.
    """
    
    try:
        endpoint = "http://192.168.0.228:8890/sparql" # "https://log.disit.org/sparql_query_frontend/"
        

        query_macro = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX km4c: <http://www.disit.org/km4city/schema#>
        SELECT DISTINCT ?category
        WHERE {
          ?category rdfs:subClassOf km4c:Service .
        }
        ORDER BY ?category
        """
        
        query_detailed = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX km4c: <http://www.disit.org/km4city/schema#>
        SELECT DISTINCT ?subcategory
        WHERE {
            ?category rdfs:subClassOf km4c:Service .
            ?subcategory rdfs:subClassOf ?category .
            }
        ORDER BY ?category
        """
        
        selected_query = query_detailed if mode == "detailed" else query_macro

        params = {
            "query": selected_query,
            "format": "application/json",
            "timeout": 30000
        }
        
        headers = {
            "User-Agent": "Snap4City-Native-Agent/1.0",
            "Connection": "close"
        }

        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            resp = await client.get(endpoint, params=params, headers=headers)
            
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    bindings = data.get("results", {}).get("bindings", [])
                    vars_list = data.get("head", {}).get("vars", [])
                    
                    cleaned_results = []
                    prefix_to_remove = "http://www.disit.org/km4city/schema#"

                    for row in bindings:
                        cleaned_row = {}
                        for var in vars_list:
                            if var in row:
                                val = row[var].get("value", "")

                                if val.startswith(prefix_to_remove):
                                    val = val.replace(prefix_to_remove, "")
                                cleaned_row[var] = val
                        
                        if cleaned_row:
                            cleaned_results.append(cleaned_row)
                    
                
                    if mode == "detailed":
                        return {"results": cleaned_results, "error": None}
                    else:
                        simple_list = [item["category"] for item in cleaned_results if "category" in item]
                        return {"results": simple_list, "error": None}

                except Exception as e:
                    return {"results": None, "error": f"JSON Parsing Error: {e}"}
            else:
                return {"results": None, "error": f"SPARQL Error: {resp.status_code}"}

    except Exception as e:
        return {"results": None, "error": str(e)}

@mcp.tool(name="get_poi_categories", tags={"poi"}, meta={"tags": ["poi"]})
def get_poi_categories(
    macro_category: Annotated[
        Optional[str], 
        Field(
            default=None,
            description=(
                "If empty, returns the list of all valid MACRO-categories for Points of Interest (POI). "
                "If you provide an exact MACRO-category name, it returns all its specific subcategories. "
                "If you get a 'not found' error, read the 'Valid options' list in the error and retry."
            )
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves dynamic POI and Service Categories from the Snap4City Knowledge Base.
    Use this tool to find the correct category names required for POI search tools.
    """
    try:
        categories_data = _read_dynamic_categories() # Legge service_categories.json
        
        if not categories_data:
            return create_error("Error: POI Categories data file is missing or empty.")
            
        if not macro_category:
            macro_list = list(categories_data.keys())
            return create_success(macro_list)
            
        search_key = macro_category.strip().lower()
        
        for valid_macro, subcategories in categories_data.items():
            if valid_macro.lower() == search_key:
                return create_success(subcategories)
                
        valid_options = ", ".join(categories_data.keys())
        return create_error(f"Macro-category '{macro_category}' not found. Valid options are: {valid_options}")

    except Exception as e:
        return create_error(f"Internal Tool Error: {str(e)}")

@mcp.tool(name="get_iot_categories", tags={"iot"}, meta={"tags": ["iot"]})
def get_iot_categories() -> Dict[str, Any]:
    """
    Retrieves the full list of valid IoT Device Categories from the Snap4City Knowledge Base.
    Use this tool BEFORE searching IoT devices to find the exact category names required.
    """
    try:
        filters_map = _read_iot_filters_map() # Legge iot_filters_map.json
        
        if not filters_map:
            return create_error("Error: IoT Categories data file is missing or empty.")
        
        # Estraiamo solo i nomi delle categorie (le chiavi del dizionario) e le ordiniamo
        categorie_iot = sorted(list(filters_map.keys()))
        return create_success(categorie_iot)
        
    except Exception as e:
        return create_error(f"Internal Tool Error: {str(e)}")

@mcp.tool(name="get_iot_category_filters", tags={"iot"}, meta={"tags": ["iot"]})
def get_iot_category_filters(
    category: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "The EXACT name of the IoT category you want to investigate (e.g. 'Weather_sensor'). "
                "If left empty, returns ALL available IoT categories. "
                "CRITICAL: Use this BEFORE searching IoT devices to discover valid 'model' names and exactly which 'filter' string to build."
            )
        )
    ] = None
) -> Dict[str, Any]:
    """
    Returns the valid Models and filterable Attributes (value_name, data_type, unit) for a specific IoT Category.
    """
    filters_map = _read_iot_filters_map()
    
    if not filters_map:
        return create_error("IoT Filters map is missing. Please notify the administrator.")
        
    if not category:
        return create_success(sorted(list(filters_map.keys())))
        
    search_key = category.strip().lower()
    
    # Ricerca flessibile (case-insensitive)
    for valid_cat, data in filters_map.items():
        if valid_cat.lower() == search_key:
            return create_success({
                "category": valid_cat,
                "supported_models": data.get("models", []),
                "available_attributes": data.get("values", {}),
                "instructions": (
                    "To build the 'filter' parameter, use valueName<operator>value. "
                    "If type is 'string' or 'binary', use ':'. If type is numeric (float/integer), use >, <, >=, <=, or =. "
                    "Example: 'temperature>20' or 'status:Active'."
                )
            })
            
    valid_options = ", ".join(sorted(list(filters_map.keys())))
    return create_error(f"Category '{category}' not found. Valid options: {valid_options}")

@mcp.tool(name="address_search_location", tags={"locator"}, meta={"tags": ["locator"]})
async def address_search_location(
    search: Annotated[
        str,
        Field(description=(
            "Search text. ALWAYS use the LOCAL language of the place: Italian for Italy (e.g. 'Duomo Firenze'), "
            "Spanish for Spain, French for France, etc. Do NOT include country names. "
            "Use plain ASCII without accents (e.g. 'Piazza del Duomo Firenze' not 'Florence Cathedral Italy')."
        ))
    ],
    logic: Annotated[
        Literal["AND", "ANDOR"],
        Field(
            default="or",
            description=(
                "Logical operator for multi-word queries. 'ANDOR' (default) matches any word — best for landmarks and named places. "
                "'AND' requires ALL words to match — use only for precise street+number lookups. "
                "NEVER use 'AND' for landmark searches: it produces poor or empty results."
            )
        )
    ] = "or",
    excludePOI: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "If true, excludes Points of Interest and returns only streets/addresses. "
                "Use false (default) for any landmark, square, building, or named place. "
                "Use true only when you explicitly need a street address."
            )
        )
    ] = False,
    maxresults: Annotated[
        int,
        Field(
            default=10,
            description=(
                "Number of results to return. For landmark lookups fetch 15–30 results: "
                "the correct match may not be first. Read all results, check the 'address' field, "
                "and pick the feature whose city matches the requested city."
            )
        )
    ] = 10,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(
            default="it",
            description="Language code for the response. Match the language of the search query (e.g. 'it' for Italian places)."
        )
    ] = "it",
    latitude: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Latitude of center point for location-biased search. Range: -90 to 90. Provide together with longitude and maxDists (e.g. from get_municipality_info's center_lat/radius_km).",
            ge=-90, le=90
        )
    ] = None,
    longitude: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Longitude of center point for location-biased search. Range: -180 to 180. Must be provided together with latitude.",
            ge=-180, le=180
        )
    ] = None,
    maxDists: Annotated[
        Optional[float],
        Field(
            default=None,
            description="Maximum search radius in km when latitude/longitude are provided. Use get_municipality_info's radius_km as value."
        )
    ] = None,
    authentication: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Search addresses and landmarks by text near a GPS position.
    When latitude/longitude are provided, results are spatially filtered within maxDists km of that point.
    Returns results list of GeoJSON features e.g longitude, latitude, address.

    Strategy for landmark lookups (e.g. 'find the Pantheon in Rome'):
    1. Call get_municipality_info to get center_lat, center_lon, radius_km of the city.
    2. Search: 'Pantheon', latitude=center_lat, longitude=center_lon, maxDists=radius_km, maxresults=15-20.

    IMPORTANT: This tool is for FORWARD geocoding only (text → location).
    Do NOT use it for reverse geocoding (coordinates → address/street name).
    For sensor or service address lookup, use service_info or check the entity's own metadata fields.
    """
    try:
        if (latitude is None) != (longitude is None):
            return {"results": None, "error": "Both latitude and longitude must be provided together."}

        base_url = "https://www.snap4city.org/superservicemap/api/v1/location/"

        def q(val): return quote(str(val))

        query_parts = [
            f"search={q(search)}",
            f"logic={logic.upper()}",
            f"format=json",
            f"maxResults={maxresults}",
            f"lang={q(lang)}",
            f"excludePOI={'true' if excludePOI else 'false'}"
        ]

        if latitude is not None and longitude is not None:
            query_parts.append(f"position={latitude};{longitude}")
            if maxDists is not None:
                query_parts.append(f"maxDists={maxDists}")

        request_url = base_url + "?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    data = resp.json()
                    features = data.get("features", [])[:MAX_RESULTS]
                    return {"results": features, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# Extended from original geocode_with_city to support batch input
@mcp.tool(name="geocode_with_city", tags={"locator"}, meta={"tags": ["locator"]})
async def geocode_with_city(
    queries: Annotated[
        List[Dict[str, Any]],
        Field(description=(
            "List of geocoding requests. Each item must have: "
            "'text' (str, landmark/address in local language), "
            "'city' (str, city name IN ITALIAN), "
            "'province' (str, province IN ITALIAN (e.g Roma NOT RO)). "
            "Optional: 'maxresults' (int, default 5). "
            "Example: [{'text': 'Duomo', 'city': 'Firenze', 'province': 'Firenze'}]"
        ))
    ]
) -> Dict[str, Any]:
    """
    Geocodes one or more landmarks/addresses within specific cities concurrently.
    Combines municipality boundary lookup with location-biased text search internally.

    Use this whenever you need the exact coordinates of named places (landmarks, squares, streets)
    in known cities. Returns one result list per query item.

    Each result contains a list of GeoJSON features with:
    - geometry.coordinates: [longitude, latitude]
    - properties.address: human-readable address string
    Pick the feature whose address most closely matches the requested city.
    """
    base_url = "https://www.snap4city.org/superservicemap/api/v1/location/"

    async def fetch_single(client: httpx.AsyncClient, item: Dict[str, Any]) -> Dict[str, Any]:
        text = item.get("text", "")
        city = item.get("city", "")
        province = item.get("province", "")
        maxresults = item.get("maxresults", 5)
        try:
            mun_name, prov_name, raw_wkt = _lookup_municipality(province, city)
            if mun_name is None:
                return {"text": text, "city": city, "results": None, **raw_wkt}

            geometry = shapely.wkt.loads(raw_wkt)
            bbox = list(geometry.bounds)
            centroid = geometry.centroid
            center_lat = round(centroid.y, 6)
            center_lon = round(centroid.x, 6)
            radius_km = round(_haversine_km(center_lat, center_lon, bbox[3], bbox[2]), 2)

            def q(val): return quote(str(val))
            query_parts = [
                f"search={q(text)}", "logic=ANDOR", "format=json",
                f"maxResults={maxresults}", "lang=it", "excludePOI=false",
                f"position={center_lat};{center_lon}", f"maxDists={radius_km}"
            ]
            request_url = base_url + "?" + "&".join(query_parts)

            resp = await client.get(request_url, headers={"Connection": "close"})
            if resp.status_code == 200 and resp.text:
                try:
                    data = resp.json()
                    features = data.get("features", [])[:maxresults]
                    return {"text": text, "city": city, "results": features, "error": None}
                except Exception as e:
                    return {"text": text, "city": city, "results": None, "error": f"Problem parsing JSON: {e}"}
            else:
                return {"text": text, "city": city, "results": None, "error": f"API Error: {_extract_api_error(resp)}"}
        except Exception as e:
            return {"text": text, "city": city, "results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

    try:
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            raw_results = await asyncio.gather(*[fetch_single(client, item) for item in queries])
        return {"results": list(raw_results), "error": None}
    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# old server's tools

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

# Snap4City Tools

# Extended from original distance_from_coordinates to support batch input
#snap4city-user
@mcp.tool(name="distance_from_coordinates",  tags = {"poi", "iot"}, meta={"tags": ["poi", "iot"]})
async def distance_from_coordinates(
    pairs: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "List of coordinate pairs to compute Haversine distance for. "
                "Each item must have: 'sourcelatitude' (float), 'sourcelongitude' (float), "
                "'destinationlatitude' (float), 'destinationlongitude' (float). "
                "Example: [{'sourcelatitude': 43.77, 'sourcelongitude': 11.25, "
                "'destinationlatitude': 43.80, 'destinationlongitude': 11.30}]"
            )
        )
    ]
) -> Dict[str, Any]:
    """
    Calculates the Haversine distance in meters for one or more pairs of GPS coordinates.
    Returns a list of distances in meters, one per input pair.
    """
    var_R = 6371  # Radius of the earth in km

    def deg2rad(deg):
        return deg * (math.pi / 180)

    def haversine(src_lat, src_lon, dst_lat, dst_lon):
        dLat = deg2rad(dst_lat - src_lat)
        dLon = deg2rad(dst_lon - src_lon)
        a = math.sin(dLat / 2) ** 2 + \
            math.cos(deg2rad(src_lat)) * math.cos(deg2rad(dst_lat)) * math.sin(dLon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return var_R * c * 1000  # meters

    try:
        results = []
        for item in pairs:
            try:
                dist = haversine(
                    item["sourcelatitude"], item["sourcelongitude"],
                    item["destinationlatitude"], item["destinationlongitude"]
                )
                results.append({"distance_meters": dist, "error": None})
            except Exception as e:
                results.append({"distance_meters": None, "error": str(e)})
        return {"results": results, "error": None}
    except Exception as e:
        return {"results": None, "error": str(e)}

# Extended from original coordinates_to_address to support batch input
#snap4city-user
@mcp.tool(name="coordinates_to_address", tags = {"poi", "iot"}, meta={"tags": ["poi", "iot"]})
async def coordinates_to_address(
    points: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "List of GPS points to reverse geocode."
                "Each item must have: 'latitude' (float, -90 to 90) and 'longitude' (float, -180 to 180). "
                "Example: [{'latitude': 43.7696, 'longitude': 11.2558}]"
            )
        )
    ]
) -> Dict[str, Any]:
    """
    Retrieves address information (reverse geocoding) for one or more GPS points concurrently.
    Returns a list of results, each containing address details like street, city, and country.
    """
    base_uri = "https://www.snap4city.org/superservicemap/api/v1"

    async def fetch_single(client: httpx.AsyncClient, item: Dict[str, Any]) -> Dict[str, Any]:
        lat = item.get("latitude")
        lon = item.get("longitude")
        try:
            resp = await client.get(f"{base_uri}/location/?position={lat};{lon}")
            if resp.status_code == 200:
                if resp.text:
                    try:
                        return {"latitude": lat, "longitude": lon, "data": resp.json(), "error": None}
                    except Exception:
                        return {"latitude": lat, "longitude": lon, "data": None, "error": "Problem Parsing data"}
                else:
                    return {"latitude": lat, "longitude": lon, "data": None, "error": "Empty response"}
            else:
                return {"latitude": lat, "longitude": lon, "data": None, "error": f"API Error: {_extract_api_error(resp)}"}
        except Exception as e:
            return {"latitude": lat, "longitude": lon, "data": None, "error": str(e)}

    try:
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            raw_results = await asyncio.gather(*[fetch_single(client, item) for item in points])
        return {"results": [r["data"] if r["error"] is None else r for r in raw_results], "error": None}
    except Exception as e:
        return {"results": None, "error": str(e)}

@mcp.tool(name="service_info", tags={"poi", "iot"}, meta={"tags": ["poi", "iot"]})
async def service_info(
    serviceuris: Annotated[
        List[str], 
        Field(
            description="A list of full Service URIs (e.g., 'http://...') to retrieve information for. This field is mandatory."
        )
    ],
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"], 
        Field(
            default="en",
            description="ISO 2-char language code for descriptions. Defaults to 'en' if the specific language is not available."
        )
    ] = "en",
    authentication: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Optional Bearer token for authorization if the service requires it."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Fetch full details for multiple services by their serviceUris concurrently.
    Returns a list of results with name, address, civic, cap, city, type, phone, and realtime IoT values if available.
    """
    base_url = "https://www.snap4city.org/superservicemap/api/v1/"
    
    # Preparazione degli header di autenticazione
    headers = {}
    if authentication:
        headers["Authorization"] = f"Bearer {authentication}"

    # Funzione helper per scaricare un singolo URI
    async def fetch_single_uri(client: httpx.AsyncClient, uri: str) -> Dict[str, Any]:
        params = {
            "serviceUri": uri,
            "realtime": "true",
            "appID": "iotapp",
            "lang": lang
        }
        try:
            resp = await client.get(base_url, params=params, headers=headers)
            if resp.status_code == 200:
                if resp.text:
                    try:
                        return {"serviceUri": uri, "data": resp.json(), "error": None}
                    except Exception:
                        return {"serviceUri": uri, "data": None, "error": "Problem Parsing data"}
                else:
                    return {"serviceUri": uri, "data": None, "error": "Empty response"}
            else:
                return {"serviceUri": uri, "data": None, "error": f"API Error: {_extract_api_error(resp)}"}
        except Exception as e:
            return {"serviceUri": uri, "data": None, "error": str(e)}

    try:
        # Usiamo un singolo client per sfruttare il connection pooling (molto più veloce)
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            # Creiamo i task per tutte le URI
            tasks = [fetch_single_uri(client, uri) for uri in serviceuris]
            # Eseguiamo tutti i task contemporaneamente
            raw_results = await asyncio.gather(*tasks)

        # Assembliamo i risultati in una lista per non innescare errori globali
        # se solo una URI fallisce
        final_data = []
        for res in raw_results:
            if res["error"] is None:
                final_data.append(res["data"])
            else:
                # Se una specifica URI fallisce, restituiamo l'errore associato ad essa
                final_data.append({"serviceUri": res["serviceUri"], "error": res["error"]})

        # Restituiamo sotto la chiave "results" affinché `_unwrap_tool_result` lo estragga come array
        return {"results": final_data}

    except Exception as e:
        return {"results": None, "error": f"Global execution error: {str(e)}"}

#snap4city-user
@mcp.tool(name="routing", tags = {"locator"}, meta={"tags": ["locator"]})
async def routing(
    startlatitude: Annotated[
        float, 
        Field(
            description="Latitude of the starting point. Range: -90 to 90.",
            ge=-90, 
            le=90
        )
    ], 
    startlongitude: Annotated[
        float, 
        Field(
            description="Longitude of the starting point. Range: -180 to 180.",
            ge=-180, 
            le=180
        )
    ], 
    endlatitude: Annotated[
        float, 
        Field(
            description="Latitude of the destination point. Range: -90 to 90.",
            ge=-90, 
            le=90
        )
    ], 
    endlongitude: Annotated[
        float, 
        Field(
            description="Longitude of the destination point. Range: -180 to 180.",
            ge=-180, 
            le=180
        )
    ],
    routetype: Annotated[
        Literal["car", "public_transport", "foot_quiet", "foot_shortest"], 
        Field(
            default="car",
            description="Type of transportation/route calculation."
        )
    ] = "car",
    startdatetime: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Date and time of the journey. Formats accepted: 'DD/MM/YYYY, HH:MM', ISO 8601, etc. Defaults to current time if omitted."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Calculates the best route between two GPS coordinates based on the selected transportation mode.
    Returns routing information including turn-by-turn arcs, duration, and the complete route geometry.
    The WKT LINESTRING of the full path is available at journey.routes[0].wkt — pass it directly to
    service_search_along_path. No conversion step is needed.
    """
    
    try:
        # Date Handling Logic
        dt_iso = ""
        formats = [
            "%d/%m/%Y, %H:%M", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
            "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M"
        ]
        
        parsed_dt = None
        
        if startdatetime:
            for fmt in formats:
                try:
                    parsed_dt = datetime.strptime(startdatetime, fmt)
                    break
                except ValueError:
                    continue
        
        # Default to now if parsing failed or no date provided
        if not parsed_dt:
            parsed_dt = datetime.now(timezone.utc)
            
        # Format required by API: 2023-10-27T10:00:00.000Z
        dt_iso = parsed_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        # API Request Construction
        base_url = "https://www.snap4city.org/superservicemap/api/v1/shortestpath/"
        
        # Note: Pydantic ensures floats are valid numbers.
        # We manually construct the query string to match the API's specific semicolon format
        query_string = (
            f"?source={startlatitude};{startlongitude}"
            f"&destination={endlatitude};{endlongitude}"
            f"&routeType={quote(routetype)}"
            f"&startDatetime={dt_iso}"
            f"&format=json"
        )
        
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            resp = await client.get(base_url + query_string)

            if resp.status_code == 200:
                if resp.text:
                    try:
                        data = resp.json()
                        response_field = data.get('response', {})
                        if response_field.get('error_code', '0') != '0':
                            error_msg = response_field.get('error_message', 'Unknown routing error')
                            return {"results": None, "error": f"Routing failed: {error_msg}"}

                        journey = data.get('journey', {})
                        routes = journey.get('routes', [])
                        if not routes:
                            error_msg = response_field.get('error_message', 'No route found')
                            return {"results": None, "error": f"Routing failed: {error_msg}"}

                        route = routes[0]
                        return {"results": {
                            "wkt": route.get('wkt', ''),
                            "distance_km": route.get('distance'),
                            "duration_seconds": route.get('time'),
                            "eta": route.get('eta'),
                        }, "error": None}
                    except Exception:
                        return {"results": None, "error": "Problem Parsing data"}
                else:
                    return {"results": None, "error": "Empty response from API"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": str(e)}

# Extended from original wkt_to_geojson to support batch input
#snap4city-user
@mcp.tool(name="wkt_to_geojson",  tags = {"shared"}, meta={"tags": ["shared"]})
async def wkt_to_geojson(
    wkt_list: Annotated[
        List[str],
        Field(
            description="List of Well-Known Text (WKT) geometry strings to convert to GeoJSON. "
                        "Example: ['POINT(30 10)', 'LINESTRING(0 0, 1 1)']",
            min_length=1
        )
    ]
) -> Dict[str, Any]:
    """
    Converts one or more Well-Known Text (WKT) geometry strings into GeoJSON objects.
    Returns a list of GeoJSON dictionaries including the bounding box (bbox) for each.
    """
    results = []
    for wkt in wkt_list:
        try:
            geometry = shapely.wkt.loads(wkt)
            geojson_output = shapely.geometry.mapping(geometry)
            geojson_output["bbox"] = list(geometry.bounds)
            results.append({"geojson": geojson_output, "error": None})
        except Exception:
            results.append({"geojson": None, "error": "Invalid WKT string"})
    return {"results": results, "error": None}

# Extended from original geojson_to_wkt to support batch input
#snap4city-user
@mcp.tool(name="geojson_to_wkt", tags = {"shared"}, meta={"tags": ["shared"]})
async def geojson_to_wkt(
    geojson_list: Annotated[
        List[str],
        Field(
            description="List of GeoJSON strings to convert to WKT. "
                        "Example: ['{\"type\": \"Point\", \"coordinates\": [30, 10]}']",
            min_length=1
        )
    ]
) -> Dict[str, Any]:
    """
    Converts one or more GeoJSON string representations into Well-Known Text (WKT) format.
    Returns a list of WKT strings, one per input item.
    """
    results = []
    for geojson in geojson_list:
        try:
            try:
                geo_data = json.loads(geojson)
            except json.JSONDecodeError:
                results.append({"wkt": None, "error": "Invalid GeoJSON: Malformed JSON string"})
                continue
            geometry = shapely.geometry.shape(geo_data)
            results.append({"wkt": geometry.wkt, "error": None})
        except Exception:
            results.append({"wkt": None, "error": "Invalid GeoJSON geometry"})
    return {"results": results, "error": None}

# Extended from original point_within_polygon to support batch input
#snap4city-user
@mcp.tool(name="point_within_polygon",  tags = {"shared"}, meta={"tags": ["shared"]})
async def point_within_polygon(
    checks: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "List of point-in-polygon checks. Each item must have: "
                "'pointlatitude' (float, -90 to 90), 'pointlongitude' (float, -180 to 180), "
                "'polygon' (str, WKT polygon string e.g. 'POLYGON((...))' ). "
                "Example: [{'pointlatitude': 43.77, 'pointlongitude': 11.25, 'polygon': 'POLYGON((...))'}]"
            )
        )
    ]
) -> Dict[str, Any]:
    """
    Checks if one or more GPS points are located inside their respective Polygons (WKT format).
    Returns a list of boolean results, one per input check.
    """
    results = []
    for item in checks:
        lat = item.get("pointlatitude")
        lon = item.get("pointlongitude")
        polygon = item.get("polygon", "")
        try:
            try:
                poly_shape = shapely.wkt.loads(polygon)
            except Exception:
                results.append({"inside": None, "error": "Invalid WKT Polygon string"})
                continue
            point_shape = shapely.geometry.Point(lon, lat)
            is_inside = poly_shape.contains(point_shape)
            results.append({"inside": is_inside, "error": None})
        except Exception as e:
            results.append({"inside": None, "error": str(e)})
    return {"results": results, "error": None}

#snap4city-user
@mcp.tool(name="service_search_along_path",  tags = {"poi"}, meta={"tags": ["poi"]})
async def service_search_along_path(
    path: Annotated[
        str, 
        Field(
            description="The geographic path described as a Well-Known Text (WKT) string (e.g., 'LINESTRING(11.2 43.7, 11.3 43.8)')."
        )
    ],
    categories: Annotated[
        str,
        Field(
            description="REQUIRED. Semicolon-separated list of categories (e.g., 'Bank;Restaurant')."
        )
    ],
    maxresults: Annotated[
        int,
        Field(
            default=100,
            description="Maximum number of results to be returned. Set to 0 for all results."
        )
    ] = 100,
    model: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Filter for IoT devices created with a specific model name."
        )
    ] = None,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ] = "en",
    authentication: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of services located near or along a given GPS path (WKT).
    Useful for finding amenities along a route.
    """
    
    try:
        # Helper to safely quote params (preserving ; and ,)
        def q(val): return quote(str(val), safe=';,')

        # Defaults are handled by Pydantic
        cat = categories
        max_r = str(maxresults)
        l = lang

        # URL Construction
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"
        
        common_params = (
            f"categories={q(cat)}"
            f"&maxResults={q(max_r)}"
            f"&lang={q(l)}"
            f"&format=json"
            f"&geometry=true"
            f"&fullCount=false"
            f"&appID=iotapp"
        )
        
        if model:
            common_params += f"&model={q(model)}"

        # CRITICAL: The API requires 'wkt:' prefix for the selection parameter when passing WKT.
        # We quote the path component, but keep 'wkt:' literal.
        query = f"?selection=wkt:{q(path)}&" + common_params
        
        request_url = base_url + query

        # HTTP Request
        headers = {}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        # Timeout set to 60s as spatial queries along paths can be heavy
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), max_results=min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

#snap4city-user
@mcp.tool(name="service_search_within_polygon",  tags = {"poi"}, meta={"tags": ["poi"]})
async def service_search_within_polygon(
    polygon: Annotated[
        str, 
        Field(
            description="The geographic area described as a Well-Known Text (WKT) Polygon (e.g., 'POLYGON((11.2 43.7, 11.3 43.7, 11.3 43.8, 11.2 43.7))')."
        )
    ],
    categories: Annotated[
        str,
        Field(
            description="REQUIRED. Semicolon-separated list of categories (e.g., 'Bank;Restaurant')."
        )
    ],
    maxresults: Annotated[
        int,
        Field(
            default=100,
            description="Maximum number of results to be returned. Set to 0 for all results."
        )
    ] = 100,
    model: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Filter for IoT devices created with a specific model name."
        )
    ] = None,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"], 
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ] = "en",
    authentication: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of services and devices contained within a specified WKT Polygon area.
    Ideal for geofencing or searching within specific district boundaries.
    """
    
    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"
        
        def q(val): return quote(str(val), safe=';,:=()')

        common_params = (
            f"categories={q(categories)}"
            f"&maxResults={maxresults}"
            f"&lang={q(lang)}"
            f"&format=json"
            f"&geometry=true"
            f"&fullCount=false"
            f"&appID=iotapp"
        )
        
        if model:
            common_params += f"&model={q(model)}"

        query = f"?selection=wkt:{q(polygon)}&" + common_params
        request_url = base_url + query

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), max_results=min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

#snap4city-user
@mcp.tool(name="full_text_search_usr",  tags = {"text","poi"}, meta={"tags": ["text", "poi"]})
async def full_text_search_usr(
    search: Annotated[
        str, 
        Field(
            description="Keywords to search for (e.g., 'Indipendenza'). This field is mandatory.",
            min_length=1
        )
    ],
    maxresults: Annotated[
        int, 
        Field(
            default=100,
            description="Maximum number of results to be returned. keep them high too have more context eg. 50, 100"
        )
    ] = 100,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"], 
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ] = "en",
    authentication: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Performs a full-text search for geolocated entities and services based on keywords. WARNING: Use a maximum of 1 keywords (e.g., 'Boboli', 'station'). Do not use complex phrases."
    Returns a list of matching Service URIs and the full GeoJSON data.
    WARNING: return result on all database filter them later to municipality address cordinate ecc..
    """
    
    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"
        
        # Helper for URL encoding
        def q(val): return quote(str(val))

        # Construct Query String
        # using manual string construction to ensure exact parameter ordering and encoding expected by the legacy API
        query_params = (
            f"?search={q(search)}"
            f"&maxResults={maxresults}"
            f"&format=json"
            f"&lang={q(lang)}"
            f"&geometry=true"
            f"&appID=iotapp"
        )
        
        full_url = base_url + query_params
        
        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        # Timeout 60s for safety on legacy API
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            resp = await client.get(full_url, headers=headers)
            
            if resp.status_code == 200 and resp.text:
                try:
                    raw_response = resp.json()
                    
                    # Extract Service URIs
                    service_uri_array = []
                    
                    # This endpoint usually returns a FeatureCollection directly
                    if "features" in raw_response:
                        for feature in raw_response["features"]:
                            if "properties" in feature:
                                if "serviceUri" in feature["properties"]:
                                    service_uri_array.append(feature["properties"]["serviceUri"])
                                
                                # Compatibility patch: ensure userStars exists
                                if "userStars" not in feature["properties"]:
                                    feature["properties"]["userStars"] = 0

                    # OUTPUT WRAPPER:
                    # Returns a list: [Array of URIs, Full GeoJSON Object]
                    return create_success([service_uri_array, raw_response])

                except Exception as e:
                    return create_error(f"Problem Parsing JSON: {str(e)}")
            else:
                return create_error(f"API Error: {_extract_api_error(resp)} - Details: {resp.text[:300]}")

    except Exception as e:
        return create_error(f"Internal Tool Error: {type(e).__name__}: {str(e)}")

@mcp.tool(name="service_search", tags={"poi"}, meta={"tags": ["poi"]})
async def service_search(
    selection: Annotated[
        str, 
        Field(
            description="A generic selection string (e.g., '43.79;11.24' for a point, or 'wkt:POLYGON(...)'). Mandatory."
        )
    ],
    filter: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Advanced IoT search filters (e.g., 'temperature>20')."
        )
    ] = None,
    values: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Semicolon-separated list of value names to return."
        )
    ] = None,
    sort: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Criteria to sort results."
        )
    ] = None,
    categories: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Semicolon-separated list of categories to filter. Omit to return all."
        )
    ] = None,
    maxresults: Annotated[
        int, 
        Field(
            default=100,
            description="Maximum number of results to return. Set to 0 for all results."
        )
    ] = 100,
    maxdists: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Maximum search radius in Km (e.g., '0.1') or 'inside'."
        )
    ] = None,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"], 
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ] = "en",
    geometry: Annotated[
        bool, 
        Field(
            default=False,
            description="If true, returns a 'hasGeometry' property for complex WKT geometries."
        )
    ] = False,
    model: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Filter for IoT devices created with a specific model name."
        )
    ] = None,
    uid: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Unique Identifier filter."
        )
    ] = None,
    typeQuery: Annotated[
        Optional[Literal["entity"]], 
        Field(
            default=None,
            description="Specify 'entity' to search for IoT devices instead of services."
        )
    ] = None,
    authentication: Annotated[
        Optional[str], 
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of services or iot-device matching a generic spatial selection.
    Returns the raw JSON response directly from the API.
    """
    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"
        
        def q(val): return quote(str(val), safe=';,:=()')

        query_parts = [f"selection={q(selection)}"]
        
        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if maxdists is not None: query_parts.append(f"maxDists={q(maxdists)}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")
        if model is not None: query_parts.append(f"model={q(model)}")
        if uid is not None: query_parts.append(f"uid={q(uid)}")
        
        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")

        if typeQuery == "entity":
            endpoint = "iot-search/?"
            if filter: query_parts.append(f"valueFilters={q(filter)}")
            if sort: query_parts.append(f"sortOnValue={q(sort)}")
            if values: query_parts.append(f"values={q(values)}")
        else:
            endpoint = "?"
            query_parts.append("format=json")
            query_parts.append("fullCount=false")

        request_url = base_url + endpoint + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)
            
            if resp.status_code == 200 and resp.text:
                try:
                    return create_success(resp.json())
                except Exception as e:
                    return create_error(f"Problem Parsing JSON: {str(e)}")
            else:
                return create_error(f"API Error: {_extract_api_error(resp)} - Details: {resp.text[:300]}")

    except Exception as e:
        return create_error(f"Internal Tool Error: {type(e).__name__}: {str(e)}")

# Derived from original service_search_near_gps_position (poi branch)
@mcp.tool(name="poi_search_near_gps_position", tags={"poi"}, meta={"tags": ["poi"]})
async def poi_search_near_gps_position(
    latitude: Annotated[
        float,
        Field(
            description="Latitude of the center GPS position. Range: -90 to 90.",
            ge=-90,
            le=90
        )
    ],
    longitude: Annotated[
        float,
        Field(
            description="Longitude of the center GPS position. Range: -180 to 180.",
            ge=-180,
            le=180
        )
    ],
    categories: Annotated[
        str,
        Field(
            description="Semicolon-separated list of POI categories to filter. Omit to return all."
        )
    ],
    maxdistance: Annotated[
        Optional[float],
        Field(
            default=1,
            description="Maximum search radius in Km (e.g., '0.5' == 500 m). Use 'inside' to search for geometries containing the point."
        )
    ],
    maxresults: Annotated[
        int,
        Field(
            default=100,
            description="Maximum number of results to return."
        )
    ],
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ],
    geometry: Annotated[
        bool,
        Field(
            default=False,
            description="If true, returns a 'hasGeometry' property for complex WKT geometries."
        )
    ],
    authentication: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ]
) -> Dict[str, Any]:
    """
    Search for POI/services near GPS coordinates within a radius.
    Returns results list of features with serviceUri, name, tipo (category), distance (km), and coordinates.
    maxdistance is in km (e.g. 1.0 = 1 km, 0.5 = 500 m).
    categories: snap4city category name (e.g. 'Bank', 'Restaurant', 'Museum', 'Hospital').
    """
    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"

        def q(val): return quote(str(val), safe=';,:=')

        query_parts = [f"selection={latitude};{longitude}"]

        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if maxdistance is not None: query_parts.append(f"maxDists={q(maxdistance)}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")

        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")
        query_parts.append("format=json")
        query_parts.append("fullCount=false")

        request_url = base_url + "?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), None, min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# Derived from original service_search_near_gps_position (iot branch)
@mcp.tool(name="iot_search_near_gps_position", tags={"iot"}, meta={"tags": ["iot"]})
async def iot_search_near_gps_position(
    latitude: Annotated[
        float,
        Field(
            description="Latitude of the center GPS position. Range: -90 to 90.",
            ge=-90,
            le=90
        )
    ],
    longitude: Annotated[
        float,
        Field(
            description="Longitude of the center GPS position. Range: -180 to 180.",
            ge=-180,
            le=180
        )
    ],
    categories: Annotated[
        str,
        Field(
            description="Semicolon-separated list of IoT device categories to filter. Omit to return all."
        )
    ],
    maxdistance: Annotated[
        Optional[float],
        Field(
            default=1,
            description="Maximum search radius in Km (e.g., '0.5' == 500 m)."
        )
    ],
    maxresults: Annotated[
        int,
        Field(
            default=100,
            description="Maximum number of results to return."
        )
    ],
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ],
    geometry: Annotated[
        bool,
        Field(
            default=False,
            description="If true, returns a 'hasGeometry' property for complex WKT geometries."
        )
    ],
    authentication: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ],
    filter: Annotated[
        Optional[str],
        Field(
            default=None,
            description="A list of conditions on value names. Use get_iot_category_filters to find valid names and types. Format: 'temperature>20' (numeric) or 'status:Active' (string). Multiple conditions are separated by ';'."
        )
    ] = None,
    values: Annotated[
        Optional[str],
        Field(
            default=None,
            description="If filter is not empty, semicolon-separated list of value names to return (e.g., 'temperature;humidity')."
        )
    ] = None,
    sortOnValue: Annotated[
        Optional[str],
        Field(
            default=None,
            description="If filter is not empty, how to sort. Format: 'valueName:asc|desc:type' (e.g. 'temperature:desc:short')."
        )
    ] = None,
    model: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Filter for specific IoT hardware model. Find valid models using get_iot_category_filters."
        )
    ] = None,
    uid: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Filter by a specific Unique Identifier (the last part of a device's serviceUri)."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Search for IoT devices/entities near GPS coordinates within a radius.
    Returns results list of features with serviceUri, name, tipo (category), distance (km), and coordinates.
    maxdistance is in km (e.g. 1.0 = 1 km, 0.5 = 500 m).
    categories: snap4city IoT device category name.
    """
    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"

        def q(val): return quote(str(val), safe=';,:=')

        query_parts = [f"selection={latitude};{longitude}"]

        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if maxdistance is not None: query_parts.append(f"maxDists={q(maxdistance)}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")
        if model is not None: query_parts.append(f"model={q(model)}")
        if uid is not None: query_parts.append(f"uid={q(uid)}")

        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")
        if filter: query_parts.append(f"valueFilters={q(filter)}")
        if sortOnValue: query_parts.append(f"sortOnValue={q(sortOnValue)}")
        if values: query_parts.append(f"values={q(values)}")

        request_url = base_url + "iot-search/?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), "entity", min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# Derived from original service_search_near_service (poi branch)
@mcp.tool(name="poi_search_near_service", tags={"poi"}, meta={"tags": ["poi"]})
async def poi_search_near_service(
    serviceuri: Annotated[
        str,
        Field(
            description="The URI of the central service (e.g., http://www.disit.org/km4city/resource/...). Mandatory."
        )
    ],
    categories: Annotated[
        str,
        Field(
            description="REQUIRED. Semicolon-separated list of POI categories (e.g., 'Bank;Restaurant')."
        )
    ],
    maxdistance: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Maximum search radius in Km (e.g., '0.1')."
        )
    ] = None,
    maxresults: Annotated[
        int,
        Field(
            default=100,
            description="Maximum number of results to return."
        )
    ] = 100,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ] = "en",
    geometry: Annotated[
        bool,
        Field(
            default=False,
            description="If true, returns a 'hasGeometry' property for complex WKT geometries."
        )
    ] = False,
    authentication: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of POI/services that are near a specific service (using its serviceUri).
    Returns a complex structure including an array of URIs, raw grouped GeoJSON, and flattened GeoJSON.
    """
    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"

        def q(val): return quote(str(val), safe=';,/?:@&=+$-_.!~*\'()#')

        query_parts = [f"selection={q(serviceuri)}"]

        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if maxdistance is not None: query_parts.append(f"maxDists={q(maxdistance)}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")

        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")
        query_parts.append("format=json")
        query_parts.append("fullCount=false")

        request_url = base_url + "?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), None, min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# Derived from original service_search_near_service (iot branch)
@mcp.tool(name="iot_search_near_service", tags={"iot"}, meta={"tags": ["iot"]})
async def iot_search_near_service(
    serviceuri: Annotated[
        str,
        Field(
            description="The URI of the central IoT device (e.g., http://www.disit.org/km4city/resource/...). Mandatory."
        )
    ],
    categories: Annotated[
        str,
        Field(
            description="REQUIRED. Semicolon-separated list of IoT device categories."
        )
    ],
    maxdistance: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Maximum search radius in Km (e.g., '0.1')."
        )
    ] = None,
    maxresults: Annotated[
        int,
        Field(
            default=100,
            description="Maximum number of results to return."
        )
    ] = 100,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(
            default="en",
            description="Language code for descriptions."
        )
    ] = "en",
    geometry: Annotated[
        bool,
        Field(
            default=False,
            description="If true, returns a 'hasGeometry' property for complex WKT geometries."
        )
    ] = False,
    authentication: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Bearer token for authorization if required."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of IoT devices/entities that are near a specific service (using its serviceUri).
    Returns a complex structure including an array of URIs, raw grouped GeoJSON, and flattened GeoJSON.
    """
    # Advanced IoT filters — disabled until agent training supports them (move to signature to re-enable)
    filter: Optional[str] = None
    values: Optional[str] = None
    sortOnValue: Optional[str] = None
    model: Optional[str] = None
    uid: Optional[str] = None

    try:
        base_url = "https://www.snap4city.org/superservicemap/api/v1/"

        def q(val): return quote(str(val), safe=';,/?:@&=+$-_.!~*\'()#')

        query_parts = [f"selection={q(serviceuri)}"]

        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if maxdistance is not None: query_parts.append(f"maxDists={q(maxdistance)}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")
        if model is not None: query_parts.append(f"model={q(model)}")
        if uid is not None: query_parts.append(f"uid={q(uid)}")

        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")
        if filter: query_parts.append(f"valueFilters={q(filter)}")
        if sortOnValue: query_parts.append(f"sortOnValue={q(sortOnValue)}")
        if values: query_parts.append(f"values={q(values)}")

        request_url = base_url + "iot-search/?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), "entity", min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# Derived from original service_search_within_gps_area (poi branch)
@mcp.tool(name="poi_search_within_gps_area", tags={"poi"}, meta={"tags": ["poi"]})
async def poi_search_within_gps_area(
    bbox: Annotated[
        List[float],
        Field(description=(
            "Bounding box: EXACTLY 4 floats [min_lon, min_lat, max_lon, max_lat]. "
            "Use the 'bbox' field returned by get_municipality_info — do NOT use point coordinates (2 values) from geocode_with_city."
        ))
    ],
    categories: Annotated[
        str,
        Field(description="REQUIRED. Semicolon-separated list of POI categories (e.g., 'Bank;Restaurant').")
    ],
    maxresults: Annotated[
        int,
        Field(default=100, description="Maximum number of results to return.")
    ] = 100,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(default="en", description="Language code for descriptions.")
    ] = "en",
    geometry: Annotated[
        bool,
        Field(default=False, description="If true, returns a 'hasGeometry' property for complex WKT geometries.")
    ] = False,
    authentication: Annotated[
        Optional[str],
        Field(default=None, description="Bearer token for authorization if required.")
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of POI/services within a given GPS Area (Bounding Box).
    Returns a complex structure including an array of URIs, raw grouped GeoJSON, and flattened GeoJSON.
    """
    try:
        if len(bbox) != 4:
            return create_error(
                f"bbox must have exactly 4 values [min_lon, min_lat, max_lon, max_lat]. "
                f"You passed {len(bbox)} value(s): {bbox}. "
                "Use the 'bbox' field from get_municipality_info, NOT point coordinates from geocode_with_city."
            )

        base_url = "https://www.snap4city.org/superservicemap/api/v1/"

        def q(val): return quote(str(val), safe=';,:=()')

        # bbox = [min_lon, min_lat, max_lon, max_lat] — convert to API format lat_bl;lon_bl;lat_tr;lon_tr
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
        selection_string = f"{min_lat};{min_lon};{max_lat};{max_lon}"

        query_parts = [f"selection={selection_string}"]

        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")

        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")
        query_parts.append("format=json")
        query_parts.append("fullCount=false")

        request_url = base_url + "?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), None, min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}

# Derived from original service_search_within_gps_area (iot branch)
@mcp.tool(name="iot_search_within_gps_area", tags={"iot"}, meta={"tags": ["iot"]})
async def iot_search_within_gps_area(
    bbox: Annotated[
        List[float],
        Field(description=(
            "Bounding box: EXACTLY 4 floats [min_lon, min_lat, max_lon, max_lat]. "
            "Use the 'bbox' field returned by get_municipality_info — do NOT use point coordinates (2 values) from geocode_with_city."
        ))
    ],
    categories: Annotated[
        str,
        Field(description="REQUIRED. Semicolon-separated list of IoT device categories.")
    ],
    maxresults: Annotated[
        int,
        Field(default=100, description="Maximum number of results to return.")
    ] = 100,
    lang: Annotated[
        Literal["en", "it", "fr", "de", "es"],
        Field(default="en", description="Language code for descriptions.")
    ] = "en",
    geometry: Annotated[
        bool,
        Field(default=False, description="If true, returns a 'hasGeometry' property for complex WKT geometries.")
    ] = False,
    authentication: Annotated[
        Optional[str],
        Field(default=None, description="Bearer token for authorization if required.")
    ] = None,
    filter: Annotated[
        Optional[str],
        Field(
            default=None,
            description="A list of conditions on value names. Use get_iot_category_filters to find valid names and types. Format: 'temperature>20' (numeric) or 'status:Active' (string). Multiple conditions are separated by ';'."
        )
    ] = None,
    values: Annotated[
        Optional[str],
        Field(
            default=None,
            description="If filter is not empty, semicolon-separated list of value names to return (e.g., 'temperature;humidity')."
        )
    ] = None,
    sortOnValue: Annotated[
        Optional[str],
        Field(
            default=None,
            description="If filter is not empty, how to sort. Format: 'valueName:asc|desc:type' (e.g. 'temperature:desc:short')."
        )
    ] = None,
    model: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Filter for specific IoT hardware model. Find valid models using get_iot_category_filters."
        )
    ] = None,
    uid: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Filter by a specific Unique Identifier (the last part of a device's serviceUri)."
        )
    ] = None
) -> Dict[str, Any]:
    """
    Retrieves a set of IoT devices/entities within a given GPS Area (Bounding Box).
    Returns a complex structure including an array of URIs, raw grouped GeoJSON, and flattened GeoJSON.
    """
    try:
        if len(bbox) != 4:
            return create_error(
                f"bbox must have exactly 4 values [min_lon, min_lat, max_lon, max_lat]. "
                f"You passed {len(bbox)} value(s): {bbox}. "
                "Use the 'bbox' field from get_municipality_info, NOT point coordinates from geocode_with_city."
            )
        if not categories or not categories.strip():
            return create_error(
                "categories is required and cannot be empty. "
                "Call get_service_categories() first to discover valid category names."
            )

        # Defensive deduplication: remove repeated entries from semicolon-separated categories string.
        _seen_cats: set = set()
        _deduped: list = []
        for _c in [c.strip() for c in categories.split(";") if c.strip()]:
            if _c not in _seen_cats:
                _deduped.append(_c)
                _seen_cats.add(_c)
        categories = ";".join(_deduped)

        base_url = "https://www.snap4city.org/superservicemap/api/v1/"

        def q(val): return quote(str(val), safe=';,:=()')

        # bbox = [min_lon, min_lat, max_lon, max_lat] — convert to API format lat_bl;lon_bl;lat_tr;lon_tr
        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
        selection_string = f"{min_lat};{min_lon};{max_lat};{max_lon}"

        query_parts = [f"selection={selection_string}"]

        if categories is not None: query_parts.append(f"categories={q(categories)}")
        if maxresults is not None: query_parts.append(f"maxResults={maxresults}")
        if lang is not None: query_parts.append(f"lang={q(lang)}")
        if model is not None: query_parts.append(f"model={q(model)}")
        if uid is not None: query_parts.append(f"uid={q(uid)}")

        query_parts.append(f"geometry={'true' if geometry else 'false'}")
        query_parts.append("appID=iotapp")
        if filter: query_parts.append(f"valueFilters={q(filter)}")
        if sortOnValue: query_parts.append(f"sortOnValue={q(sortOnValue)}")
        if values: query_parts.append(f"values={q(values)}")

        request_url = base_url + "iot-search/?" + "&".join(query_parts)

        headers = {"Connection": "close"}
        if authentication:
            headers["Authorization"] = f"Bearer {authentication}"

        async with httpx.AsyncClient(verify=False, timeout=90.0) as client:
            resp = await client.get(request_url, headers=headers)

            if resp.status_code == 200 and resp.text:
                try:
                    features, total = _extract_features(resp.json(), "entity", min(maxresults, MAX_RESULTS))
                    return {"results": features, "total": total, "error": None}
                except Exception as e:
                    return {"results": None, "error": f"Problem Parsing JSON: {e}"}
            else:
                return {"results": None, "error": f"API Error: {_extract_api_error(resp)}"}

    except Exception as e:
        return {"results": None, "error": f"Internal Tool Error: {type(e).__name__}: {e}"}


# full_text_search_usr.disable()
# get_province_boundary.disable()
# get_region_boundary.disable()
# service_search.disable()
# address_search_location.disable()
# get_service_categories_old.disable()

app_advisor_experimental = mcp.http_app(path='/tool/search')

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)