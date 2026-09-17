from typing import List
import pymupdf
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

import uuid

from dotenv import load_dotenv
import anthropic

from models import ChatTurn
import os



load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# qdrant = QdrantClient(host="localhost", port=6333)
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)
claude_client = anthropic.Anthropic()

COLLECTION_NAME = "paper_chunks"
TOP_K = 20
CLAUDE_MODEL = "claude-sonnet-4-6"

HISTORY_EXCHANGES = 5
MAX_TOOL_ROUNDS = 3



# if not qdrant.collection_exists(COLLECTION_NAME):
#     qdrant.create_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(size=384, distance=Distance.COSINE),
#     )

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(...)

qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="filename",
    field_schema="keyword",
)

def store_chunks(filename: str, chunks: list[str]):
    embeddings = embedding_model.encode(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def extract_abstract(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    first_page_text = doc[0].get_text()
    doc.close()

    match = re.search(
        r"abstract\s*[:\n]?\s*(.*?)(?=\n\s*(introduction|1\.|1\s+introduction|keywords)\b)",
        first_page_text,
        re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    return first_page_text[:500].strip()


def extract_full_text(pdf_bytes: bytes) -> str:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return full_text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    return splitter.split_text(text)



def retrieve_chunks(question: str, filename: str, top_k: int = TOP_K) -> list[str]:
    query_vector = embedding_model.encode(question).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
        limit=top_k,
        with_payload=True,
    ).points

    return [point.payload["text"] for point in results]

def paper_already_chunked(filename: str) -> bool:
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
        limit=1,
    )
    return len(results) > 0


SYSTEM_PROMPT = (
    "You are helping a researcher understand an academic paper they have "
    "uploaded. You have a search tool that retrieves relevant passages from "
    "that paper.\n\n"
    "Use the search tool whenever the question asks about the paper's content "
    "and you need passages you don't already have. Do NOT search when the "
    "question can be answered from the conversation so far — for example when "
    "the user asks you to clarify, rephrase, expand on, or summarize something "
    "you already said, or asks a general question that isn't about this "
    "specific paper.\n\n"
    "When you do search, base your answer only on the retrieved passages and "
    "the conversation. If the passages don't contain the answer, say so plainly "
    "rather than guessing."
)

SEARCH_TOOL = {
    "name": "search_paper",
    "description": (
        "Search the uploaded paper for passages relevant to a query. Returns "
        "the most semantically similar excerpts from the paper's text. Use a "
        "self-contained, descriptive query — if the user's question refers back "
        "to earlier conversation (e.g. 'explain point 2'), rewrite it into a "
        "standalone query describing the actual topic before searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A standalone description of the information needed.",
            }
        },
        "required": ["query"],
    },
}


def build_messages(question: str, history: List[ChatTurn]) -> list[dict]:
    trimmed = history[-(HISTORY_EXCHANGES * 2):] if history else []

    while trimmed and trimmed[0].role != "user":
        trimmed = trimmed[1:]

    messages = [{"role": turn.role, "content": turn.text} for turn in trimmed]
    messages.append({"role": "user", "content": question})
    return messages


def run_conversation(question: str, history: List[ChatTurn], filename: str) -> dict:
    messages = build_messages(question, history)
    searches: list[str] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            answer = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {
                "received": answer,
                "searches": searches,
                "searched": len(searches) > 0,
            }

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            query = block.input["query"]
            searches.append(query)
            print(f"[/ask] search_paper: {query}", flush=True)

            chunks = retrieve_chunks(query, filename)
            result_text = (
                "\n\n---\n\n".join(chunks)
                if chunks
                else "No passages found for that query."
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    answer = "".join(block.text for block in response.content if block.type == "text")
    return {"received": answer, "searches": searches, "searched": True}