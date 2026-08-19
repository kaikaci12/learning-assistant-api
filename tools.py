from modules.vectorstore import load_vector_store
from modules.vectorstore import vector_store
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from langgraph.prebuilt import ToolRuntime


@tool
def retrieve_documents(
    query: str,
    k: int = 4,
    runtime: Annotated[ToolRuntime, InjectedToolArg] = None
) -> str:
    """Retrieve context from vector store using similarity search to answer questions from uploaded study materials.
    
    Args:
        query: The search query string to look up relevant documents.
        k: Number of documents to retrieve (default is 4).
    """
    print("Retrieving documents...")
    if runtime is None or not hasattr(runtime, "context") or runtime.context is None:
        return "No session found"

    session_id = getattr(runtime.context, "session_id", None)
    if session_id is None:
        return "No session found"
    vector_store = load_vector_store()
    results = vector_store.similarity_search(
        query=query,
        k=k,
        filter=lambda doc: doc.metadata.get("session_id") == session_id
    )
    docstrings = "\n\n".join([d.page_content for d in results])
    print(f"Retrieved {len(results)} documents for session {session_id}")
    return docstrings if docstrings else "No relevant documents found."



