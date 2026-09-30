from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()
import requests
from langchain_core.embeddings import Embeddings
EMBEDDING_MODEL=os.getenv("OPENROUTER_EMBEDDING_MODEL")
class OpenRouterEmbeddings(Embeddings):
    """Custom Embeddings class for OpenRouter embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str =EMBEDDING_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
    ):    
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided or set in environment.")
        self.model = model
        self.base_url = base_url.rstrip("/")
    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []


        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        
        # Sort embeddings by index to preserve order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search documents."""
        # Batch in chunks of 50 to avoid payload limits
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        result = self._embed([text])
        return result[0] if result else []


def create_embedding():
    try:
        embedding = OpenRouterEmbeddings(
            model=EMBEDDING_MODEL
        )
        print("Embeddings loaded successfully (OpenRouter API)")
        return embedding
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

embedding = create_embedding()


def chunk_docs(docs:list[Document],chunk_size:int=500,chunk_overlap:int=50)->list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
