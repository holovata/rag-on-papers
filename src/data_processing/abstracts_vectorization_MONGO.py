#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive pipeline for bulk-loading arXiv metadata into MongoDB Atlas with
OpenAI embeddings.  The pipeline state (last processed batch) is persisted
inside MongoDB so that the script can be resumed at any time.

Menu
─────
1) Process the next batches and add embeddings
2) Clear the metadata collection
3) Reset the pipeline state   (last_batch = 0)
4) Exit
"""

import os
import sys
import json
import time
from typing import List

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Configuration
# ╚═══════════════════════════════════════════════════════════════════════════╝
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_FILE  = os.path.join(BASE_DIR, "data", "raw_metadata",
                          "arxiv-metadata-oai-snapshot.json")

BATCH_SIZE          = 1_000          # JSON lines read per iteration of the loader
MAX_BATCHES_PER_RUN = 20             # Safety stop – max batches per script run
EMBED_CHUNK_SIZE    = 100            # How many abstracts to embed in one API call

load_dotenv()

MONGODB_URI   = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    sys.exit("❌  The environment variable OPENAI_API_KEY is missing.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  MongoDB connection
# ╚═══════════════════════════════════════════════════════════════════════════╝
client      = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
db          = client["arxiv_db"]
papers_col  = db["arxiv_metadata"]
state_col   = db["pipeline_state"]

try:
    client.admin.command("ping")
    print("✔ MongoDB connection established.")
except Exception as exc:
    sys.exit(f"✘  Failed to connect to MongoDB: {exc}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Pipeline-state helpers
# ╚═══════════════════════════════════════════════════════════════════════════╝
def get_last_batch() -> int:
    doc = state_col.find_one({"_id": "checkpoint"})
    return doc.get("last_batch", 0) if doc else 0


def save_last_batch(num: int) -> None:
    state_col.update_one({"_id": "checkpoint"},
                         {"$set": {"last_batch": num}},
                         upsert=True)
    print(f"⟲  Checkpoint saved: last_batch = {num}")


def reset_checkpoint() -> None:
    state_col.update_one({"_id": "checkpoint"},
                         {"$set": {"last_batch": 0}},
                         upsert=True)
    print("⟲  Pipeline state reset (last_batch = 0).")


def clear_metadata_collection() -> None:
    deleted = papers_col.delete_many({}).deleted_count
    print(f"⟲  Collection 'arxiv_metadata' cleared: {deleted} documents removed.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  JSON streaming loader
# ╚═══════════════════════════════════════════════════════════════════════════╝
def stream_json_batches(path: str, batch_size: int = 1000):
    """
    Lazy-load a gigantic JSON-lines file (one object per line) and yield
    lists of `batch_size` dictionaries.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            fh.readline()            # skip the opening '['
            batch: List[dict] = []
            for line in fh:
                line = line.strip().rstrip(",]").strip()
                if not line:
                    continue
                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError as err:
                    print(f"⚠  JSON decode error: {err}")
                    continue
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    except FileNotFoundError:
        sys.exit(f"✘  Data file not found: {path}")
    except Exception as err:
        sys.exit(f"✘  Error reading JSON: {err}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Insertion with batched embeddings
# ╚═══════════════════════════════════════════════════════════════════════════╝
def add_batch_to_mongodb(batch: List[dict], model: OpenAIEmbeddings) -> int:
    """
    Embed abstracts in groups of `EMBED_CHUNK_SIZE` and insert the documents.
    Returns the number of new documents written.
    """
    # Keep only valid records (id + abstract present)
    valid_entries = [e for e in batch if e.get("id") and e.get("abstract")]
    total_added = 0

    for start in range(0, len(valid_entries), EMBED_CHUNK_SIZE):
        chunk = valid_entries[start:start + EMBED_CHUNK_SIZE]
        abstracts = [entry["abstract"] for entry in chunk]

        try:
            vectors = model.embed_documents(abstracts)
        except Exception as exc:
            print(f"⚠  Embedding API error (skipping chunk): {exc}")
            continue

        for entry, vec in zip(chunk, vectors):
            try:
                papers_col.insert_one({
                    "_id":         entry["id"],
                    "title":       entry.get("title", ""),
                    "authors":     entry.get("authors", ""),
                    "abstract":    entry["abstract"],
                    "categories":  entry.get("categories", ""),
                    "comments":    entry.get("comments", ""),
                    "journal_ref": entry.get("journal-ref", ""),
                    "doi":         entry.get("doi", ""),
                    "embedding":   vec
                })
                total_added += 1
            except DuplicateKeyError:
                # Document already exists – ignore
                pass
            except Exception as exc:
                print(f"⚠  Failed to insert {entry['id']}: {exc}")

    print(f"✔  {total_added} documents added in this batch.")
    return total_added

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Interactive menu
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

    while True:
        print("\n=== Actions ===")
        print("1) Process next batches (add embeddings)")
        print("2) Clear 'arxiv_metadata' collection")
        print("3) Reset pipeline state (last_batch = 0)")
        print("4) Exit")
        choice = input("Select an option (1–4): ").strip()

        if choice == "1":
            last = get_last_batch()
            print(f"⏩  Resuming from batch #{last + 1}")
            total_new = 0

            for idx, batch in enumerate(stream_json_batches(DATA_FILE, BATCH_SIZE), start=1):
                if idx <= last:
                    continue

                print(f"→ Processing batch {idx} …")
                added = add_batch_to_mongodb(batch, embed_model)
                total_new += added
                save_last_batch(idx)

                if idx >= last + MAX_BATCHES_PER_RUN:
                    print("⚠  Batch limit reached for this run.")
                    break

            print(f"\n✅  Done: {total_new} new documents inserted.")

        elif choice == "2":
            confirm = input("⚠  This will delete ALL documents. Type 'yes' to confirm: ").lower()
            if confirm == "yes":
                clear_metadata_collection()
            else:
                print("↩  Operation cancelled.")

        elif choice == "3":
            reset_checkpoint()

        elif choice == "4":
            print("👋  Goodbye!")
            break

        else:
            print("❗  Invalid selection. Enter a number between 1 and 4.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
#  Entry point
# ╚═══════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    main()
