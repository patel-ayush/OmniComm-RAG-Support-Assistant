from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from .rag_chain import get_rag_chain

app = FastAPI(title="OmniComm AI Support API")

# Initialize chain globally (in production this might be loaded differently, but good for take-home)
try:
    rag_chain = get_rag_chain()
except Exception as e:
    print(f"Warning: Failed to initialize RAG chain: {e}")
    rag_chain = None

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG system not initialized properly. Check API keys and DB.")
        
    try:
        result = rag_chain.invoke(request.question)
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
        return AskResponse(
            answer=result_dict.get("answer", "No answer provided"),
            sources=result_dict.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
