from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List,Union
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class AgentState(TypedDict):
    measage: List[Union[HumanMessage, AIMessage]]

def process_node(state: AgentState) -> AgentState:
    """Process the message using the LLM and update the state."""
    response = llm.invoke(state["measage"])
    print(f"AI: {response.content}")
    state["measage"].append(AIMessage(content=response.content))
    # print("CURRENT_STATE :",state["measage"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

connversation_history = []
user_input = input("You: ")
while user_input!="exit":
    connversation_history.append(HumanMessage(content=user_input))
    response_state = app.invoke(AgentState(measage=connversation_history))
    # print(response_state["measage"])
    connversation_history = response_state["measage"]
    user_input = input("You: ")


with open("logger.txt","w") as f:
    f.write("Conversation History:\n")
    for message in connversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n\n")
    f.write("\nEnd of Conversation\n")
print("Conversation history saved to logger.txt")
