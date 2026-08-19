from langgraph.graph import StateGraph,MessagesState,START,END

from typing import Annotated
from langgraph.graph.message import add_messages
from .llm import model
from tools import retrieve_documents
from langgraph.prebuilt import ToolNode
from dataclasses import dataclass
model_with_tools = model.bind_tools([retrieve_documents])
@dataclass
class Context:
    session_id:str
class AgentState(MessagesState):
    messages:Annotated[list,add_messages]

from langchain_core.messages import SystemMessage, HumanMessage
SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful Learning Assistant. Use tool to search uploaded study materials before answering any user question about course topics, documents, or learning content. "
)

def llm_node(state: AgentState):
    # Ensure system prompt is always at the beginning of the messages list sent to the LLM
    # (Without duplicating it inside state["messages"])
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
def should_continue(state:AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tool"
    return END

workflow = StateGraph(AgentState,context_schema=Context,)
workflow.add_node("llm",llm_node)
workflow.add_node("tool",ToolNode(tools=[retrieve_documents]))
workflow.add_edge(START,"llm")
workflow.add_conditional_edges("llm",should_continue,["tool",END])
workflow.add_edge("tool","llm")
agent = workflow.compile()
