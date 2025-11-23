from typing import Annotated, TypedDict, List, Union, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # Reducer Function 
#Rule that controls how updates from nodes are combined into the existing state.
#tells us how to merge new messages into the current state
#without a replacer function that overwrites the entire message history each time.
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Annotated - used to provide additional context without affect the type itself.
# Sequence - to automatically handle the states updates for seqeuences of such as adding new messages to chat history. 
# TypedDict - used to define the structure of the AgentState.
# BaseMessage - is the foundation for all message types in LangGraph.
# ToolMessage - passes data back to llm after it calls a tool such as content and the tool-call metadata.
# SystemMessage - used to set the behavior of the AI.


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int,b:int)->int:
    """Adds two numbers and returns the result."""
    return a+b

@tool
def subtract(a:int,b:int)->int:
    """Subtracts b from a and returns the result."""
    return a-b

@tool
def multiply(a:int,b:int)->int:
    """Multiplies two numbers and returns the result."""
    return a*b

tools =[add, subtract, multiply]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """Calls the LLM with the current messages and returns the updated state."""
    system_prompt = SystemMessage(
        content="You are a helpful AI assistant. please answer my queries to best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}
   

def should_continue(state: AgentState):
    message = state["messages"]
    last_message = message[-1] 
    if not last_message.tool_calls:
        return "end"
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools =tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Calculate 12 multiplied by 7, then add 10 and subtract 5 from the result." )]}
print_stream(app.stream(inputs,stream_mode="values"))
        


