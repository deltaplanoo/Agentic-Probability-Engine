import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from fastmcp import Client
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up")



async def run_agent():
    async with Client("http://localhost:8000/sse") as mcp_client:
        
        # MCP tools -> LangChain tools
        raw_mcp_tools = await mcp_client.list_tools()
        print(f"Found {len(raw_mcp_tools)} MCP tools: {[t.name for t in raw_mcp_tools]}")
        langchain_tools = []
        
        for m_tool in raw_mcp_tools:
            def create_tool_wrapper(name):
                async def wrapper(query: str) -> str:
                    result = await mcp_client.call_tool(name, arguments={"query": query})
                    return result.content[0].text
                return wrapper

            lc_tool = StructuredTool.from_function(
                name=m_tool.name,
                description=m_tool.description,
                coroutine=create_tool_wrapper(m_tool.name),
                args_schema=WebSearchInput,
            )
            langchain_tools.append(lc_tool)

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        ).bind_tools(langchain_tools)

        # nodes
        def call_model(state: AgentState):
            response = model.invoke(state['messages'])
            return {"messages": [response]}

        tool_node = ToolNode(langchain_tools)

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)

        # edges
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState):
            last_message = state['messages'][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END
        
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        app = workflow.compile()

        # execution
        inputs = {"messages": [HumanMessage(content="Should I open a restaurant in Amsterdam? Use your web search tool.")]}
        
        async for chunk in app.astream(inputs, stream_mode="values"):
            final_msg = chunk["messages"][-1]
            final_msg.pretty_print()

if __name__ == "__main__":
    asyncio.run(run_agent())