#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query arXiv papers stored in MongoDB Atlas by cosine-similarity using OpenAI embeddings.
Підтримує фільтрацію по порогу min_similarity.
"""

import sys
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# Завантаження змінного середовища
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY не встановлено в середовищі!", file=sys.stderr)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Підключення до MongoDB Atlas
# ──────────────────────────────────────────────────────────────────────────────

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    serverSelectionTimeoutMS=20000,
    socketTimeoutMS=20000
)
try:
    client.admin.command("ping")
except Exception as e:
    print(f"❌ Не вдалося підключитися до MongoDB: {e}")
    sys.exit(1)

db = client["arxiv_db"]
papers_col = db["arxiv_metadata"]

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI Embedding Initialization
# ──────────────────────────────────────────────────────────────────────────────

embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

def generate_query_embedding(query: str) -> list[float] | None:
    """
    Generate an embedding vector for the user's query using OpenAI.
    Returns a list of floats, or None on failure.
    """
    try:
        return embed_model.embed_query(query)
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Основна функція пошуку
# ──────────────────────────────────────────────────────────────────────────────

def get_top_relevant_articles(
    query: str,
    top_n: int = 3,
    min_similarity: float = 0.0,
    num_candidates: int = 150
) -> list[dict]:
    """
    Retrieve up to top_n articles whose vector-similarity ≥ min_similarity.
    Використовує $vectorSearch (MongoDB Atlas Search) і фільтрує результати.
    """
    qe = generate_query_embedding(query)
    if qe is None:
        print("⚠ Failed to generate embedding for query.")
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index":         "embedding_vector_index",
                "path":          "embedding",
                "queryVector":   qe,
                "numCandidates": num_candidates,
                "limit":         num_candidates
            }
        },
        {
            "$project": {
                "_id":       1,
                "title":     1,
                "authors":   1,
                "abstract":  1,
                "score":     {"$meta": "vectorSearchScore"}
            }
        }
    ]

    try:
        cursor = papers_col.aggregate(pipeline, maxTimeMS=45000)
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return []

    candidates = []
    for doc in cursor:
        score = float(doc.get("score", 0.0))
        if score >= min_similarity:
            candidates.append({
                "id":         doc["_id"],
                "title":      doc.get("title", "N/A"),
                "authors":    doc.get("authors", "N/A"),
                "abstract":   (doc.get("abstract", "")[:500] + "...") if doc.get("abstract") else "No abstract",
                "similarity": score,
                "pdf_url":    f"https://arxiv.org/pdf/{doc['_id']}.pdf"
            })

    if not candidates:
        print(f"⚠ No articles with similarity ≥ {min_similarity:.2f}")
        return []

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    results = candidates[:top_n]
    print(f"🔍 Found {len(results)} articles (max similarity = {candidates[0]['similarity']:.4f}).")
    return results

# ──────────────────────────────────────────────────────────────────────────────
# Інтерактивний запуск
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    user_query = input("🔎 Введіть пошуковий запит: ").strip()
    if not user_query:
        print("⚠ Порожній запит. Завершення.")
        sys.exit(0)

    top_articles = get_top_relevant_articles(
        query=user_query,
        top_n=5,
        min_similarity=0.6,
        num_candidates=150
    )

    if top_articles:
        print("\n📄 Найрелевантніші статті:")
        for i, art in enumerate(top_articles, 1):
            print(f"\n#{i}")
            print(f"ID:         {art['id']}")
            print(f"Title:      {art['title']}")
            print(f"Authors:    {art['authors']}")
            print(f"Similarity: {art['similarity']:.6f}")
            print(f"Abstract:   {art['abstract']}")
            print(f"PDF:        {art['pdf_url']}")
    else:
        print("😕 Нічого не знайдено.")
