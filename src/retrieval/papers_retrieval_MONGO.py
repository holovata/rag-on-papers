import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# ────────────── Константы ──────────────
MONGODB_URI        = os.getenv("MONGODB_URI")
DB_NAME            = "arxiv_db"
CHUNK_COL          = "paper_chunks"
CHNK_INDEX         = "chunk_embedding_index"  # имя vectorSearch-индекса в коллекции paper_chunks
MIN_SIMILARITY     = 0.0
NUM_CANDIDATES     = 150
DEFAULT_TOP_K      = 4
# ───────────────────────────────────────

class Config:
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

config = Config()

def debug_log(message: str):
    print(f"[DEBUG] {message}")

def mongo_client() -> MongoClient:
    if not MONGODB_URI:
        raise RuntimeError("❌ MONGODB_URI not set in environment (.env)")
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
        raise RuntimeError(f"❌ MongoDB connection failed: {e}")

def get_relevant_chunks(
    query: str,
    top_n: int = DEFAULT_TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    num_candidates: int = NUM_CANDIDATES,
    db_name: str = "file_buffer_db",
    chunk_collection: str = CHUNK_COL,              # Новый аргумент
    chunk_index_name: str = CHNK_INDEX              # Новый аргумент
) -> list[Document]:
    try:
        qe = config.embedding_function.embed_documents([query])[0]
    except Exception as e:
        debug_log(f"Failed to embed query: {e}")
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": chunk_index_name,
                "path": "embedding",
                "queryVector": qe,
                "numCandidates": num_candidates,
                "limit": num_candidates
            }
        },
        {
            "$project": {
                "_id":         1,
                "source":      1,
                "chunk_index": 1,
                "content":     1,
                "score":       {"$meta": "vectorSearchScore"}
            }
        }
    ]

    client = mongo_client()
    db = client[db_name]
    coll = db[chunk_collection]  # теперь можно передавать нужную коллекцию

    try:
        cursor = coll.aggregate(pipeline, maxTimeMS=45_000)
    except Exception as e:
        debug_log(f"❌ Vector search failed: {e}")
        return []

    candidates = []
    for doc in cursor:
        score = float(doc.get("score", 0.0))
        if score >= min_similarity:
            candidates.append({
                "id":          doc["_id"],
                "source":      doc.get("source", ""),
                "chunk_index": doc.get("chunk_index", -1),
                "content":     doc.get("content", ""),
                "similarity":  score
            })

    if not candidates:
        debug_log(f"⚠ No chunks with similarity ≥ {min_similarity:.2f}")
        return []

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    top_candidates = candidates[:top_n]
    debug_log(f"🔍 Found {len(top_candidates)} chunks (max sim = {candidates[0]['similarity']:.4f}).")

    docs: list[Document] = []
    for c in top_candidates:
        metadata = {
            "source":      c["source"],
            "chunk_index": c["chunk_index"],
            "similarity":  c["similarity"],
            "mongo_id":    c["id"]
        }
        docs.append(Document(page_content=c["content"], metadata=metadata))

    return docs


def main():
    try:
        conn_status = mongo_client().admin.command("ping")
        print("✅ MongoDB Atlas: Connected.")
    except Exception as e:
        print(f"❌ MongoDB Atlas: {e}")
        return

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break
        debug_log(f"Retrieving chunks for query: {query}")
        chunks = get_relevant_chunks(query=query, top_n=5, chunk_collection="multi_paper_chunks", chunk_index_name="multi_chunk_embedding_index")
        if not chunks:
            print("⚠ No relevant chunks found.")
        else:
            print(f"\n🔎 Top {len(chunks)} relevant chunks:\n")
            for i, doc in enumerate(chunks, 1):
                meta = doc.metadata
                print(f"--- Chunk #{i} ---")
                print(f"📄 Source:     {meta.get('source')}")
                print(f"🔢 Index:      {meta.get('chunk_index')}")
                print(f"📈 Similarity: {meta.get('similarity'):.4f}")
                print(f"🧩 Content:\n{doc.page_content[:500].strip()}...\n")

if __name__ == "__main__":
    main()
