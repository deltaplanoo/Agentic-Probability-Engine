Data-driven probability assestment agent.

## Current workflow

```mermaid
graph TD
    %% Entry Point
    Start((User Question)) --> Parse[<b>Step 1: Parse Question</b><br/>Extract Decision Type & Variables<br/><i>Geocoding with Nominatim</i>]
    
    %% Routing logic
    Parse --> CheckDB{Template in DB?}
    %% Path A: Reusing Template
    CheckDB -- Yes --> Inject[<b>Step 6b: Inject Variables</b><br/>Replace placeholders in template]
    Inject --> Score

    %% Path B: New Template Generation
    CheckDB -- No --> Reword[<b>Step 2: Reword Query</b><br/>Optimize for Global Search]
    Reword --> GlobalSearch[<b>Step 3: Global Web Search</b><br/>Tavily API]
    GlobalSearch --> ExtractParams[<b>Step 4: Extract Parameters</b><br/>Identify key decision factors]

    subgraph Parallel_Generation [Tree Generation]
        ExtractParams --> GenTrees[<b>Step 5a: Generate 3 Candidates</b><br/>Parallel LLM Generation]
        GenTrees --> PickBest[<b>Step 5b: Pick Best Tree</b><br/>Selection by coverage/weights]
    end
    
    PickBest --> Annotate[<b>Step 6a: Annotate & Save</b><br/>Add Search Hints & Save to SQLite]
    Annotate --> Score

    %% Core Scoring Logic
    subgraph Leaf_Scoring [Targeted Leaf Scoring]
        Score[<b>Step 7: Score Leaf IF</b><br/>Process each leaf in parallel]
        Score --> Choice{Tool Choice?}
        Choice -- POI Category --> POITool[<b>Km4City API Tool</b><br/>Physical Proximity & Services]
        Choice -- Contextual --> WebTool[<b>Web Search Tool</b><br/>Market Trends & Regulations]
        POITool --> AggResults[Assign IF Triplet]
        WebTool --> AggResults
    end
    
    %% Final Calculation
    AggResults --> Propagate[<b>Step 8: Propagate IF</b><br/>MCP process_decision_tree<br/>Recursive Weighted Average]
    Propagate --> Display[<b>Step 9: Present Results</b><br/>Italian Flag UI & Verdict]
    Display --> End((Final Decision))
    
    %% Styling
    style CheckDB fill:#f9f,stroke:#333,stroke-width:2px
    style Choice fill:#f9f,stroke:#333,stroke-width:2px
    style Parallel_Generation fill:#e1f5fe,stroke:#01579b
    style Leaf_Scoring fill:#f1f8e9,stroke:#33691e
```

## Tree Generation
The agent searches the web for key decision-making factors, builds the tree, assigns scores to the leaves, and propagates the weighted average up through the intermediate layers to the root node.

Example: "Is opening a restaurant in Via Calzaiuoli 50 in Florence a good idea?"

```mermaid
graph TD
    %% Root Node
    root["<b>Root:</b> Is opening a restaurant in {address} a good idea?<br/><i>(Weight: 1.0)</i>"]

    %% Group 1: Location & Market
    root --> group_loc["<b>Group:</b> Location & Market Potential<br/><i>(Weight: 0.5)</i>"]
    group_loc --> leaf_foot["<b>Leaf:</b> High Foot Traffic<br/><i>(Weight: 0.3)</i>"]
    group_loc --> leaf_tour["<b>Leaf:</b> Tourist Appeal & Volume<br/><i>(Weight: 0.3)</i>"]
    group_loc --> leaf_econ["<b>Leaf:</b> Economic Activity<br/><i>(Weight: 0.2)</i>"]
    group_loc --> leaf_hist["<b>Leaf:</b> Location Type Historic/Cultural<br/><i>(Weight: 0.1)</i>"]
    group_loc --> leaf_social["<b>Leaf:</b> Commercial Hubs/Social Condensers<br/><i>(Weight: 0.1)</i>"]

    %% Group 2: Business Viability
    root --> group_viab["<b>Group:</b> Industry & Business Viability<br/><i>(Weight: 0.3)</i>"]
    group_viab --> leaf_rest_viab["<b>Leaf:</b> Location Viability for Restaurant<br/><i>(Weight: 0.7)</i>"]
    group_viab --> leaf_horeca["<b>Leaf:</b> Horeca Sector Outlook<br/><i>(Weight: 0.3)</i>"]

    %% Group 3: Challenges & Competition
    root --> group_chal["<b>Group:</b> Potential Challenges & Competition<br/><i>(Weight: 0.2)</i>"]
    group_chal --> leaf_comp["<b>Leaf:</b> Existing Competition at {address}<br/><i>(Weight: 0.6)</i>"]
    group_chal --> leaf_tourism_impact["<b>Leaf:</b> Impact of Mass Tourism on Identity<br/><i>(Weight: 0.4)</i>"]

    %% Styling
    style root fill:#f9f,stroke:#333,stroke-width:2px
    style group_loc fill:#bbf,stroke:#333,stroke-width:1px
    style group_viab fill:#bbf,stroke:#333,stroke-width:1px
    style group_chal fill:#bbf,stroke:#333,stroke-width:1px
    
    classDef leafNode fill:#fff,stroke:#999,stroke-dasharray: 5 5
    class leaf_foot,leaf_tour,leaf_econ,leaf_hist,leaf_social,leaf_rest_viab,leaf_horeca,leaf_comp,leaf_tourism_impact leafNode
```
