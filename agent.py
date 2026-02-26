import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from fastmcp import Client

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, "Lista dei messaggi della conversazione"]

async def run_agent():
    async with Client("http://localhost:8000/sse") as mcp_client:
        
        mcp_tools = await mcp_client.list_tools()
        
        # model = ChatOpenAI(model="gpt-4o").bind_tools(mcp_tools)

        # definizione nodi
        def call_model(state: AgentState):
            messages = state['messages']
            # response = mock_model.invoke(messages)
            # return {"messages": [response]}
            return {"messages": ["ok boss"]}

        tool_node = ToolNode(mcp_tools)

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        # LLM vuole usare un tool? nodo tools : finisci
        def should_continue(state: AgentState):
            last_message = state['messages'][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        app = workflow.compile()

        # esecuzione
        inputs = {"messages": [("user", "Should I open a restaurant in Amsterdam?")]}
        async for chunk in app.astream(inputs, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

if __name__ == "__main__":
    asyncio.run(run_agent())