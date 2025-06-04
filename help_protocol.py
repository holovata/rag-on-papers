import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import yaml
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError
from gridfs import GridFS
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────── ENV & CONSTANTS ───────────────────────────────────
load_dotenv()

MONGODB_URI       = os.getenv("MONGODB_URI")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
FILE_BUFFER_DB    = "file_buffer_db"
MANUAL_COLLECTION = "manual_chunks"
GRIDFS_BUCKET     = "manual_files"

# Полный путь к исходному Markdown-файлу руководства
MANUAL_PATH       = Path(r"C:\Work\diplom2\rag_on_papers\data\user_manual\user_manual.md")
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200

if not MONGODB_URI or not OPENAI_API_KEY:
    print("[ERROR] MONGODB_URI or OPENAI_API_KEY not set in environment variables.")
    sys.exit(1)

# ─────────────────────────────────── DB HELPERS ───────────────────────────────────

def mongo_client() -> MongoClient:
    try:
        client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=10_000,
            socketTimeoutMS=10_000
        )
        client.admin.command("ping")
        return client
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        sys.exit(1)

# ─────────────────────────────── MANUAL INGESTION ───────────────────────────────

def ingest_manual_to_db() -> None:
    """Читает MANUAL_PATH, разбивает на чанки, делает эмбеддинги и сохраняет в MongoDB."""
    client = mongo_client()
    db     = client[FILE_BUFFER_DB]

    # 1) Сохранить исходный Markdown в GridFS (bucket: manual_files)
    fs = GridFS(db, collection=GRIDFS_BUCKET)
    try:
        for old in fs.find({"filename": MANUAL_PATH.name}):
            fs.delete(old._id)
        with MANUAL_PATH.open("rb") as fh:
            fs.put(fh, filename=MANUAL_PATH.name)
        print("[INFO] Manual saved to GridFS")
    except Exception as e:
        print(f"[WARNING] GridFS save failed: {e}")

    # 2) Чанкить и генерация эмбеддингов
    try:
        text = MANUAL_PATH.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Cannot read manual: {e}")
        sys.exit(1)

    splitter   = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents  = splitter.create_documents([text])
    chunks     = [d.page_content for d in documents]

    embedder   = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = embedder.embed_documents(chunks)

    coll = db[MANUAL_COLLECTION]
    coll.delete_many({})  # очистить старые записи

    records = [{
        "_id":        f"{MANUAL_PATH.stem}_chunk_{i}",
        "chunk_idx":  i,
        "content":    chunk,
        "embedding":  emb,
        "source":     MANUAL_PATH.name
    } for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]

    try:
        coll.insert_many(records, ordered=False)
        print(f"[INFO] Inserted {len(records)} chunks to '{MANUAL_COLLECTION}'")
    except DuplicateKeyError:
        print("[WARNING] Some chunks already existed (duplicate keys).")
    finally:
        client.close()

# ─────────────────────────────── COSINE SEARCH ───────────────────────────────

def get_top_relevant_documents(query: str, top_n: int = 3, **_) -> List[Dict[str, Any]]:
    """
    Возвращает top_n сегментов из MANUAL_COLLECTION,
    ранжируя по косинусной близости к query.
    Каждый элемент списка содержит: content, chunk_index, similarity.
    """
    client = mongo_client()
    db     = client[FILE_BUFFER_DB]
    coll   = db[MANUAL_COLLECTION]
    docs   = list(coll.find({}, {"embedding": 1, "content": 1, "chunk_idx": 1}))

    if not docs:
        client.close()
        return []

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    q_vec    = embedder.embed_query(query)

    doc_vectors = np.array([d["embedding"] for d in docs])
    sims        = cosine_similarity([q_vec], doc_vectors)[0]

    ranked = sorted(zip(sims, docs), key=lambda x: -x[0])[:top_n]
    results = []
    for score, doc in ranked:
        results.append({
            "content":      doc["content"],
            "chunk_index":  doc["chunk_idx"],
            "similarity":   float(score)
        })

    client.close()
    return results

# ─────────────────────────────── QA OVER GUIDE ───────────────────────────────

def answer_query_from_guide(query: str, prompt_template: str | None = None) -> Any:
    """
    Возвращает либо готовую строку (при invoke), либо генератор chunks (при stream).
    Здесь мы помещаем сам stream() генератор в context["streamed_response"].
    """
    top_chunks = get_top_relevant_documents(query, top_n=3)
    if not top_chunks:
        return "No relevant information found in the guide."

    context_text = "\n\n".join(chunk["content"] for chunk in top_chunks)

    if prompt_template:
        prompt = prompt_template.format(query=query, guide_chunks=context_text)
    else:
        prompt = (
            "You are an assistant answering a user based solely on the documentation excerpts below.\n\n"
            "### USER QUERY\n" + query +
            "\n\n### DOCUMENTATION\n" + context_text +
            "\n\n### ANSWER\n"
        )

    # **ВОТ ГДЕ МЫ ВОЗВРАЩАЕМ STREAM-ГЕНЕРАТОР, А НЕ INVOKE()**
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.0).stream([("user", prompt)])

# ────────────────────────────────── PIPELINE ──────────────────────────────────

class UserGuidePipeline:
    def __init__(self, protocol_path: str):
        with open(protocol_path, "r", encoding="utf-8") as f:
            self.protocol = yaml.safe_load(f)
        self.context: Dict[str, Any] = {}

    def resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                var_name = value[2:-2].strip()
                resolved[key] = self.context.get(var_name, value)
            else:
                resolved[key] = value
        return resolved

    def run_pipeline(self, user_query: str):
        self.context.clear()
        self.context["query"] = user_query

        for step in self.protocol["pipeline"]:
            stage = step["stage"]
            func_name = step["function"]
            params = self.resolve_parameters(step.get("parameters", {}))
            self.context.update(params)

            cot = step.get("cot_template", "")
            if cot:
                print(f"\n=== {stage.upper()} ===")
                print("CoT:", cot.format(**self.context))

            if func_name == "answer_query_from_guide":
                # Вместо обычного func(): возвращает stream-генератор
                streamed = answer_query_from_guide(params["query"], step.get("prompt_template"))
                # помещаем сам генератор в context
                self.context["streamed_response"] = streamed
                result = None
            else:
                func = globals().get(func_name)
                if not func:
                    print(f"[ERROR] Function '{func_name}' not found.")
                    result = None
                else:
                    result = func(**params)

            # сохраняем результат (либо None, либо какой-то output)
            self.context[f"{func_name}_output"] = result

            # логируем для стадии retrieval
            if stage == "retrieval" and isinstance(result, list):
                print("\n[Top Chunks]:")
                for idx, ch in enumerate(result, 1):
                    snippet = ch["content"][:120].replace("\n", " ")
                    print(f" {idx}. idx={ch['chunk_index']} sim={ch['similarity']:.2f} | {snippet}...")

            yield stage, self.context.copy()

        yield "done", self.context.copy()

# ─────────────────────────────────── CLI ───────────────────────────────────

if __name__ == "__main__":
    pipeline = UserGuidePipeline("help_protocol.yaml")
    print("=== QA for User Guide ===")
    query = input("Enter your query: ").strip()
    for _stage, _ctx in pipeline.run_pipeline(query):
        pass
