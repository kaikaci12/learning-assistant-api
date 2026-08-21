A lightweight FastAPI service that lets you upload PDF study materials and chat with an AI assistant that can search your documents in real time.  
Built with **LangChain** + **FastAPI** + **InMemory** vector store.

> **TL;DR**  
> 1️⃣ Upload a PDF → it’s chunked, embedded & stored.  
> 2️⃣ Start a chat session → the model answers using only your uploaded docs.  

---

### 🚀 Features

| Feature | Description |
|---------|-------------|
| **PDF Upload** | Upload any PDF, it’s parsed, chunked and embedded automatically. |
| **Session‑based Chat** | Each upload gets a unique `session_id`. Use it to keep context per user. |
| **RAG Agent** |   Agent decides to answer directly to user or call "retrieve_documents" tool that retrieves relevant context from vector store. 
| **Streaming Responses** | Chat replies are streamed via Server‑Sent Events (SSE) – perfect for real‑time UIs.

# Clone the repo
git clone https://github.com/your-username/learning-assistant-api.git
cd learning-assistant-api

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# or, if you use uv (recommended for speed)
uv pip install -r requirements.txt
```

> **Docker** (optional)  
> ```bash
> docker build -t learning-assistant .
> docker run -p 8000:8000 learning-assistant
> ```

---

### ⚙️ Configuration
