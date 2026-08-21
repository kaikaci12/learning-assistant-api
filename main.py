

from fastapi.responses import StreamingResponse,Response

from langchain_core.messages import HumanMessage
from modules.vectorstore import load_vector_store
from modules.embedding import chunk_docs
from langchain_community.document_loaders import PyPDFLoader
import os
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from uuid import uuid4
from modules.agent import agent

app = FastAPI(title="Learning Assistant API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOCUMENTS_DIR = Path("./documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
sessions = []
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        
        # validate pdf
        print(file)
        # Read raw byte content asynchronously into memory
        contents = await file.read()
    
        
        file_path = DOCUMENTS_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        

        # Write raw bytes directly in binary mode ('wb')

        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        chunks = chunk_docs(docs)
        session_id = str(uuid4())
        sessions.append(session_id)
        docs_with_ids = []
        for doc in chunks:
            doc.metadata["session_id"] = session_id
            docs_with_ids.append(doc)


        vector_store = load_vector_store()
        vector_store.add_documents(docs_with_ids)
        # delete the document
        os.remove(file_path)
       
        
        return {"message":"Documents uploaded successfully", "session_id":session_id}
    except Exception as e:
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

import json

@app.post("/chat/{session_id}")
@app.post("/chatbot/{session_id}")
async def chat_with_session(session_id: str, request: dict):
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    config = { 
        "configurable": {
            "thread_id": session_id
        }
    }
    query = request.get("query", "")

    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=query)]},
                config=config,
                context={"session_id": session_id},
                version="v2"
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, str) and content:
                            yield f"data: {json.dumps({'token': content})}\n\n"
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    yield f"data: {json.dumps({'token': block.get('text', '')})}\n\n"
                                elif isinstance(block, str) and block:
                                    yield f"data: {json.dumps({'token': block})}\n\n"
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    yield f"data: {json.dumps({'status': f'Searching study documents...'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
@app.get("/health")
def health_check(response:Response):
    
    
    return {"status": "ok"}
