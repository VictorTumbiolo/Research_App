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

# Prod / Local setup
# qdrant = QdrantClient(host="localhost", port=6333)
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

claude_client = anthropic.Anthropic()

COLLECTION_NAME = "paper_chunks"
CLAUDE_MODEL = "claude-sonnet-4-6"

HISTORY_EXCHANGES = 5
MAX_TOOL_ROUNDS = 1
MAX_CANDIDATE_CHUNKS = 500
TARGET_FRACTION = 0.33

# Payl;oad
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
                "tokens": count_tokens(chunk),  # computed once, here
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def count_tokens(text: str) -> int:
    result = claude_client.messages.count_tokens(
        model=CLAUDE_MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens

def get_page_count(pdf_bytes: bytes) -> int:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count

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



def retrieve_chunks(question: str, filename: str, target_fraction: float = TARGET_FRACTION) -> dict:
    query_vector = embedding_model.encode(question).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
        limit=MAX_CANDIDATE_CHUNKS,
        with_payload=True,
    ).points

    if not results:
        return {"chunks": [], "tokens_used": 0, "total_tokens": 0}

    total_tokens = sum(point.payload["tokens"] for point in results)
    target_tokens = int(total_tokens * target_fraction)

    selected = []
    running_tokens = 0

    for point in results:
        selected.append(point.payload["text"])
        running_tokens += point.payload["tokens"]
        if running_tokens >= target_tokens:
            break

    return {"chunks": selected, "tokens_used": running_tokens, "total_tokens": total_tokens}

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
    tokens_used_total = 0
    paper_total_tokens = 0

    for _ in range(MAX_TOOL_ROUNDS):
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            answer = "".join(block.text for block in response.content if block.type == "text")
            return {
                "received": answer,
                "searches": searches,
                "searched": len(searches) > 0,
                "tokens_used": tokens_used_total,
                "total_tokens": paper_total_tokens,
            }

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            query = block.input["query"]
            searches.append(query)
            print(f"[/ask] search_paper: {query}", flush=True)

            retrieval = retrieve_chunks(query, filename)
            tokens_used_total += retrieval["tokens_used"]
            paper_total_tokens = retrieval["total_tokens"]

            result_text = (
                "\n\n---\n\n".join(retrieval["chunks"])
                if retrieval["chunks"]
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
    return {
        "received": answer,
        "searches": searches,
        "searched": True,
        "tokens_used": tokens_used_total,
        "total_tokens": paper_total_tokens,
    }
