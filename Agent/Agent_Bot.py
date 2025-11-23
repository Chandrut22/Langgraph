from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    message: List[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def process_message(state: AgentState) -> AgentState:
    response = llm.invoke([HumanMessage(content=state["message"])])
    print(response)
    print(f"AI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    app.invoke(AgentState(message=[user_input]))