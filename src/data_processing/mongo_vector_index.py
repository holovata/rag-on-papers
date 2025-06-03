#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive tool for managing vector indexes in MongoDB Atlas.

You can:
1) View existing vector search indexes
2) Delete an index by name
3) Create a new vector index on 'embedding' field
4) Exit
"""

import os
import time
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# MongoDB Atlas connection
# ──────────────────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi("1"),
    serverSelectionTimeoutMS=20000,
    socketTimeoutMS=20000
)
db = client["arxiv_db"]
papers_col = db["arxiv_metadata"]

try:
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas.")
except Exception as e:
    raise SystemExit(f"❌ MongoDB ping failed: {e}")


def list_vector_indexes():
    """
    Display all existing vector search indexes in the collection.
    """
    print("\n📦 Existing search indexes:")
    indexes = list(papers_col.list_search_indexes())
    if not indexes:
        print("  (No search indexes found.)")
    for i, idx in enumerate(indexes, 1):
        name = idx.get("name", "unknown")
        q = idx.get("queryable", False)
        idx_type = idx.get("type", "unknown")
        print(f" {i}. {name} | type: {idx_type} | queryable: {q}")


def delete_index_by_name(index_name: str):
    try:
        papers_col.drop_search_index(index_name)
        print(f"🗑️  Index '{index_name}' has been deleted.")
    except Exception as e:
        print(f"⚠ Failed to delete index '{index_name}': {e}")


def create_vector_index(index_name="embedding_vector_index"):
    """
    Create a new vector index on field `embedding`, inferring the dimension from a sample document.
    If an index with the same name exists, prompt the user to delete and recreate it.
    """
    # Check if index with the same name already exists
    existing_indexes = list(papers_col.list_search_indexes())
    if any(idx.get("name") == index_name for idx in existing_indexes):
        print(f"⚠ Index '{index_name}' already exists.")
        confirm = input("Do you want to delete and recreate it? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("⏎ Skipped index creation.")
            return
        try:
            papers_col.drop_search_index(index_name)
            print(f"🗑️  Deleted index '{index_name}'.")
        except Exception as e:
            print(f"❌ Failed to delete index: {e}")
            return

    sample_doc = papers_col.find_one({"embedding": {"$exists": True}}, {"embedding": 1})
    if not sample_doc or "embedding" not in sample_doc:
        print("✘ No document with field 'embedding' found.")
        return

    vector = sample_doc["embedding"]
    if not isinstance(vector, list):
        print("✘ Field 'embedding' must be a list of floats.")
        return

    num_dimensions = len(vector)
    print(f"ℹ Detected embedding dimension: {num_dimensions}")

    index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type":         "vector",
                    "path":         "embedding",
                    "numDimensions": num_dimensions,
                    "similarity":   "dotProduct",
                    "quantization": "scalar"
                }
            ]
        },
        name=index_name,
        type="vectorSearch"
    )

    print(f"➤ Creating vector search index '{index_name}'...")
    try:
        created_name = papers_col.create_search_index(model=index_model)
        print(f"✔ Index '{created_name}' is being built.")
    except Exception as e:
        print(f"✘ Failed to create index: {e}")
        return

    # Wait until queryable
    print("⏳ Waiting for index to become queryable...")
    while True:
        status = list(papers_col.list_search_indexes(created_name))
        if status and status[0].get("queryable", False):
            print(f"✔ Index '{created_name}' is now ready for querying.")
            break
        time.sleep(5)



def main():
    while True:
        print("\n=== Index Management Menu ===")
        print("1) List existing vector indexes")
        print("2) Delete an index by name")
        print("3) Create new vector index on 'embedding'")
        print("4) Exit")
        choice = input("Your choice (1–4): ").strip()

        if choice == "1":
            list_vector_indexes()

        elif choice == "2":
            list_vector_indexes()
            index_name = input("Enter index name to delete: ").strip()
            if index_name:
                delete_index_by_name(index_name)

        elif choice == "3":
            create_vector_index()

        elif choice == "4":
            print("👋 Goodbye.")
            break

        else:
            print("❗ Invalid option. Please enter a number between 1 and 4.")


if __name__ == "__main__":
    main()
    client.close()
