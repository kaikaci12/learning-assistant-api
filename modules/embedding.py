from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_huggingface import HuggingFaceEndpointEmbeddings
def create_embedding():
    try:
        embedding = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=os.getenv("HF_TOKEN"))
        print("Embeddings loaded successfully")
        return embedding
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None
embedding = create_embedding()


def chunk_docs(docs:list[Document],chunk_size:int=500,chunk_overlap:int=50)->list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
