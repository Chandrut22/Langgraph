from typing import TypedDict, List, Sequence, Union, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

documunt_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def upload_document(content: str) -> str:
    """Uploads a document and returns a confirmation message."""
    global documunt_content
    documunt_content = content
    return f"Document uploaded successfully! the content is \n{documunt_content}"

@tool
def save_document(filename: str) -> str:
    """Saves the document content to a file."""
    global documunt_content

    if not filename.endswith(".txt"):
        filename = filename + ".txt"

    try:
        with open(filename, "w") as file:
            file.write(documunt_content)
        print(f"Document saved to {filename}")
        return f"Document saved successfully as {filename}."
    except Exception as e:
        return f"Failed to save document: {str(e)}"
    
tools = [upload_document, save_document]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    """Calls the LLM with the current messages and returns the updated state."""
    system_prompt = SystemMessage(content=
        f"""You are a Drafter AI assistant, a helpful writing assistant. you are going to help the user udate and modify documents.
        
        -if the user provides new content, use the 'upload_document' tool to upload it.
        -if the user wants to save the document, use the 'save_document' tool to save it.
        -make sure to always show the current document state after modification.

        the current document content is:{documunt_content}
        """
    )

    if not state["messages"]:
        user_input = "I'm ready to help you upload a document. what would you like to create ?"
        user_meassage = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document ?")
        print(f"\n USER: {user_input}")
        user_meassage = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_meassage]
    response = model.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response,"tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"])+[user_meassage,response]}

def should_continue(state: AgentState) -> str:
    
    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if(isinstance(message,ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end"
    
    return "continue"

def print_messages(messages):
    if not messages: return 
    for message in messages[:-3]:
        if(isinstance(message, ToolMessage)):
            print(f"\n TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
    print("\n======== DRAFTER =========")

    state = {"messages":[]}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED ===== ")

if __name__ == "__main__":
    run_document_agent()