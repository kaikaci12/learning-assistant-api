from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .embedding import SentenceTransformerEmbeddings



vector_store = InMemoryVectorStore(embedding=SentenceTransformerEmbeddings())








    
    