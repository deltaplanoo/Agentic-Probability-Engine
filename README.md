Data-driven probability assestment agent.

Current workflow

```mermaid
graph TD
    START((START)) --> reword_query
    present_results --> END((END))

    reword_query[reword_query] --> run_search[run_search]
    run_search --> extract_and_score_parameters[extract_and_score_parameters]
    extract_and_score_parameters --> generate_decision_tree[generate_decision_tree]
    generate_decision_tree --> score_leaf_if[score_leaf_if]
    score_leaf_if --> calculate_tree[calculate_tree]
    calculate_tree --> present_results[present_results]

    %% Styling
    style START fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
    style END fill:#f44336,stroke:#333,stroke-width:2px,color:#fff
```

## Tree Generation
The agent searches the web for key decision-making factors, builds the tree, assigns scores to the leaves, and propagates the weighted average up through the intermediate layers to the root node.

Example: "Is opening a restaurant in Via Calzaiuoli 50 in Florence a good idea?"

```mermaid
graph TD
    %% Root Node
    root["Is opening a restaurant in Via Calzaiuoli 50...<br/>(Total Weight: 1.0)"]

    root --> market["Market Attractiveness & Customer Base<br/>(W: 0.5)"]
    root --> competition["Competitive Landscape<br/>(W: 0.3)"]
    root --> operations["Operational & Financial Viability<br/>(W: 0.2)"]

    market --> traffic["Street Popularity/Foot Traffic<br/>(W: 0.4)"]
    market --> tourism["Tourism<br/>(W: 0.4)"]
    market --> residents["Local Residents<br/>(W: 0.2)"]

    competition --> existing["Existing Competition<br/>(W: 1.0)"]

    operations --> estate["Real Estate & Econ Conditions<br/>(W: 0.6)"]
    operations --> access["Accessibility & Parking<br/>(W: 0.4)"]
```