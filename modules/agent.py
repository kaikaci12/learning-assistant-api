from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from .llm import model
from tools import retrieve_documents


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

model_with_tools = model.bind_tools(
    [retrieve_documents]
)


# ─────────────────────────────────────────────
# NeMo Guardrails
# ─────────────────────────────────────────────

config = RailsConfig.from_path("./config")

rails = RunnableRails(
    config,
    passthrough=True

)

guarded_model = rails | model_with_tools

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

@dataclass
class Context:
    session_id: str

class AgentState(MessagesState):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful Learning Assistant. "
        "Use the tool to search uploaded study materials before "
        "answering any user question about course topics, documents, "
        "or learning content."
    )
)



# ─────────────────────────────────────────────
# LLM node
# ─────────────────────────────────────────────


def llm_node(state: AgentState):
    messages = [
        SYSTEM_PROMPT,
        *state["messages"],
    ]

    response = guarded_model.invoke(messages)

    return {
        "messages": [response]
    }


# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tool"

    return END


# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────

checkpointer = InMemorySaver()

workflow = StateGraph(
    AgentState,
    context_schema=Context,
)

workflow.add_node("llm", llm_node)

workflow.add_node(
    "tool",
    ToolNode(
        tools=[retrieve_documents]
    ),
)

workflow.add_edge(START, "llm")

workflow.add_conditional_edges(
    "llm",
    should_continue,
    ["tool", END],
)

workflow.add_edge("tool", "llm")

agent = workflow.compile(
    checkpointer=checkpointer
)