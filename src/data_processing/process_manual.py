import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError

from gridfs import GridFS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# Constants
MONGODB_URI       = os.getenv("MONGODB_URI")
FILE_BUFFER_DB    = "file_buffer_db"
MANUAL_COLLECTION = "manual_chunks"
GRIDFS_BUCKET     = "manual_files"

MANUAL_PATH       = Path(r"C:\Work\diplom2\rag_on_papers\data\user_manual\user_manual.md")
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200

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

def get_gridfs_bucket(client: MongoClient, db_name: str, bucket: str) -> GridFS:
    db = client[db_name]
    return GridFS(db, collection=bucket)

def save_to_bucket(fs: GridFS, file_path: Path, filename_in_db: str, overwrite: bool = True):
    if overwrite:
        try:
            for old in fs.find({"filename": filename_in_db}):
                fs.delete(old._id)
        except Exception:
            pass
    with file_path.open("rb") as fh:
        fs.put(fh, filename=filename_in_db)
    print(f"[INFO] Saved to GridFS: {filename_in_db}")

def process_manual():
    client = mongo_client()
    db_buffer = client[FILE_BUFFER_DB]

    fs_manual = get_gridfs_bucket(client, FILE_BUFFER_DB, bucket=GRIDFS_BUCKET)
    save_to_bucket(fs_manual, MANUAL_PATH, MANUAL_PATH.name)

    try:
        text = MANUAL_PATH.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to read manual: {e}")
        client.close()
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    documents = splitter.create_documents([text])
    chunks = [doc.page_content for doc in documents]

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        embeddings = embedder.embed_documents(chunks)
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        client.close()
        sys.exit(1)

    coll = db_buffer[MANUAL_COLLECTION]
    deleted = coll.delete_many({})
    print(f"[INFO] Cleared collection '{MANUAL_COLLECTION}', removed {deleted.deleted_count} documents.")

    for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        doc_id = f"{MANUAL_PATH.stem}_chunk_{idx}"
        rec = {
            "_id":       doc_id,
            "source":    MANUAL_PATH.name,
            "chunk_idx": idx,
            "content":   chunk_text,
            "embedding": emb
        }
        try:
            coll.insert_one(rec)
            print(f"[INFO] Inserted chunk #{idx} (id={doc_id})")
        except DuplicateKeyError:
            print(f"[WARNING] Chunk #{idx} already exists. Skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to insert chunk #{idx}: {e}")

    client.close()
    print("[SUCCESS] Manual processing complete.")

def search_manual(query: str, top_k: int = 3) -> List[Tuple[float, str]]:
    client = mongo_client()
    db = client[FILE_BUFFER_DB]
    coll = db[MANUAL_COLLECTION]

    docs = list(coll.find({}, {"embedding": 1, "content": 1}))
    if not docs:
        print("[ERROR] No documents found in manual_chunks.")
        return []

    query_embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vec = query_embedder.embed_query(query)

    doc_vectors = np.array([d["embedding"] for d in docs])
    similarities = cosine_similarity([query_vec], doc_vectors)[0]

    ranked = sorted(zip(similarities, docs), key=lambda x: -x[0])[:top_k]
    results = [(sim, doc["content"]) for sim, doc in ranked]
    client.close()
    return results

if __name__ == "__main__":
    if not MONGODB_URI:
        print("[ERROR] MONGODB_URI is not set in the environment.")
        sys.exit(1)

    process_manual()

    # Example query (manual test)
    test_query = "How can I upload a PDF and ask questions about it?"
    results = search_manual(test_query)
    print("\nTop matching chunks:")
    for score, text in results:
        print(f"\n[Score: {score:.3f}]\n{text[:300]}...")