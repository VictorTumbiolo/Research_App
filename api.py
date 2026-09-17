from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from models import UserQuery
import rag
from rag import paper_already_chunked

router = APIRouter()

@router.post("/ask")
async def ask(query: UserQuery):
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        return rag.run_conversation(question, query.history, query.filename)
    except Exception as exc:
        print(f"[/ask] error answering question: {exc}", flush=True)
        raise HTTPException(status_code=500, detail="Failed to answer question") from exc

@router.post("/papers")
async def upload_papers(files: List[UploadFile] = File(...)):
    results = []

    for file in files:

        if ( rag.paper_already_chunked(file.filename) ):
            results.append({
                "filename": file.filename,
                "num_chunks": 0,
                "status": "already_chunked",
            })
            continue

        contents = await file.read()
        full_text = rag.extract_full_text(contents)
        chunks = rag.chunk_text(full_text)

        rag.store_chunks(file.filename, chunks)

        print(f"\n--- {file.filename}: {len(chunks)} chunks stored ---")

        results.append({
            "filename": file.filename,
            "num_chunks": len(chunks),
            "status": "chunked",
        })

    return {"papers": results}

@router.post("/test")
async def test():
    return {"test": "test"}