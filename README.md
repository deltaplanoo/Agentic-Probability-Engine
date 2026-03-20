Data-driven probability assestment agent.

Current workflow

```mermaid
graph TD
    START((START)) --> Node1
    Node8 --> END((END))

    Node1[1. Rephrase query] --> Node2[2. Run web search]
    Node2 --> Node3[3. Extract and score parameters]
    Node3 --> Node4[4. Generate decision trees]
    Node4 --> Node5[5. Pick best tree]
    Node5 --> Node6[6. Score IF leaf nodes]
    Node6 --> Node7[7. Calculate tree]
    Node7 --> Node8[8. Present results]

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