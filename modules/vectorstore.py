from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .embedding import embedding


vector_store = None

def load_vector_store():
    global vector_store
    if vector_store is None:
        print("Loading embedding vector_store...")
        vector_store = InMemoryVectorStore(embedding=embedding)
    return vector_store







    
    