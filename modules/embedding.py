from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
# class EmbeddingManager:
#     def __init__(self,model_name:str="all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.model = None
#         self._load_model()
#     def _load_model(self):
#         """
#         Loads the SentenceTransformer model.
#         """
#         try:
#             self.model = SentenceTransformer(self.model_name)
#             print(f"Embedding model loaded successfully {self.model}. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
#         except Exception as e:
#             print(f"Error loading embedding model: {e}")
#     def get_model(self):
#         if not self.model:
#             raise ValueError("Model not defined")
#         return self.model
#     def get_embedding(self, text):
#         if not self.model:
#             raise ValueError("Model not defined")

#         embeddings = self.model.encode(text,show_progress_bar=True)
#         print(f"Generated Embeddings with shape {embeddings.shape}")
#         return embeddings
    
#     def get_embeddings(self, texts):
#         return self.model.encode(texts,show_progress_bar=True)

class SentenceTransformerEmbeddings:
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"Embedding model loaded successfully {self.model}. Embedding dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = None
    def embed_documents(self, texts):
        if not self.model:
            raise ValueError("Model not defined")
        return self.model.encode(texts,show_progress_bar=True)
    def embed_query(self,query):
        if not self.model:
            raise ValueError("Model not defined")
        return self.model.encode(query,show_progress_bar=True)

def chunk_docs(docs:list[Document],chunk_size:int=500,chunk_overlap:int=50)->list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
