import os
import time
from dotenv import load_dotenv
 
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage
 
from uagents_adapter import LangchainRegisterTool, cleanup_uagent
import asyncio
# Load environment variables
load_dotenv()
loop=asyncio.new_event_loop()
asyncio.set_event_loop(loop)
 
# Set your API keys - for production, use environment variables instead of hardcoding
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
 
# Get API token for Agentverse
API_TOKEN = os.environ["AGENTVERSE_API_KEY"]
 
if not API_TOKEN:
    raise ValueError("Please set AGENTVERSE_API_KEY environment variable")
 
# Set up tools and LLM
tools = [TavilySearchResults(max_results=3)]
model = ChatOpenAI(temperature=0)
 
# LangGraph-based executor
app = chat_agent_executor.create_tool_calling_executor(model, tools)
 
# Wrap LangGraph agent into a function for UAgent
def langgraph_agent_func(query):
    # Handle input if it's a dict with 'input' key
    if isinstance(query, dict) and 'input' in query:
        query = query['input']
    
    messages = {"messages": [HumanMessage(content=query)]}
    final = None
    for output in app.stream(messages):
        final = list(output.values())[0]  # Get latest
    return final["messages"][-1].content if final else "No response"
 
# Register the LangGraph agent via uAgent
tool = LangchainRegisterTool()
agent_info = tool.invoke(
    {
        "agent_obj": langgraph_agent_func,
        "name": "omnitrix",
        "port": 8080,
        "description": "A LangGraph-based Tavily-powered search agent",
        "api_token": API_TOKEN,
        "mailbox": True
    }
)
 
print(f"✅ Registered LangGraph agent: {agent_info}")
 
# Keep the agent alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🛑 Shutting down LangGraph agent...")
    cleanup_uagent("langgraph_tavily_agent")
    print("✅ Agent stopped.")