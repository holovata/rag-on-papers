#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Utility for:
 1) Ingesting Markdown from local files.
 2) Ingesting cleaned Markdown from GridFS.
 3) Creating/updating vector indexes for abstracts and chunks.
 4) New functionality: Ingesting multiple PDF files at once:
    4.1) Convert each PDF → Markdown (remove tables).
    4.2) Save "clean" Markdown and PDF into a separate GridFS bucket.
    4.3) Store metadata of Markdown files in the pdf_markdowns collection.
    4.4) Read cleaned Markdown from GridFS, chunk, vectorize, and write into
         file_buffer_db.paper_chunks.
    4.5) Create/update a vector index for file_buffer_db.paper_chunks.
 5) Support for two GridFS buckets:
    • default (fs.files/fs.chunks) — for single-PDF pipeline.
    • pipeline_fs (pipeline_fs.files/pipeline_fs.chunks) — for multi-PDF pipeline.
 6) Ability to clear each bucket and the pdf_markdowns collection via the menu.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from bson import ObjectId
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel
from pymongo.errors import DuplicateKeyError, OperationFailure

from gridfs import GridFS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Assume these functions are implemented in your modules:
from src.data_processing.pdf_to_md import convert_pdf_to_md
from src.data_processing.removetables import remove_markdown_tables

load_dotenv()  # Load MONGODB_URI from .env

# ────────────────────── CONSTANTS ──────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI")
FILE_BUFFER_DB  = "file_buffer_db"
ABSTRACT_DB     = "arxiv_db"
ABS_COL         = "arxiv_metadata"
CHUNK_COL       = "paper_chunks"           # Collection in file_buffer_db for single-PDF chunks
MULTI_CHUNK_COL = "multi_paper_chunks"     # Collection in arxiv_db for multi-PDF chunks
MULTI_MD_COL    = "pdf_markdowns"          # Collection in file_buffer_db for Markdown metadata

DEFAULT_MD_FOLDER  = Path(__file__).resolve().parent.parent / "data" / "dl_papers_md"
DEFAULT_PDF_FOLDER = Path(r"C:\Work\diplom2\rag_on_papers\data\downloaded_papers")

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200

ABS_INDEX        = "embedding_vector_index"
CHNK_INDEX       = "chunk_embedding_index"
MULTI_CHNK_INDEX = "multi_chunk_embedding_index"
# ────────────────────────────────────────────────────────


def mongo_client() -> MongoClient:
    """
    Establish and return a MongoDB client. Exit on failure.
    """
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
        print(f"❌ MongoDB connection failed: {e}")
        sys.exit(1)


def get_gridfs_bucket(
    mongo_uri: str,
    db_name: str,
    bucket: str | None = None
) -> Tuple[MongoClient, GridFS]:
    """
    Return (MongoClient, GridFS) for a specified bucket.
    bucket=None  → default fs.files/fs.chunks
    bucket="pipeline_fs" → pipeline_fs.files/pipeline_fs.chunks
    """
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    db = client[db_name]
    fs = GridFS(db, collection=bucket) if bucket else GridFS(db)
    return client, fs


def save_to_bucket(
    fs: GridFS,
    file_path: Path,
    filename_in_db: str,
    overwrite: bool = True
) -> ObjectId:
    """
    Save a file into the specified GridFS bucket.
    If overwrite=True, delete older versions with the same filename first.
    Returns the ObjectId of the saved file.
    """
    if overwrite:
        try:
            for old in fs.find({"filename": filename_in_db}):
                fs.delete(old._id)
        except Exception:
            pass
    with file_path.open("rb") as fh:
        file_id = fs.put(fh, filename=filename_in_db)
    return file_id


# ────────────────────── Markdown Helpers ──────────────────────

def list_md(folder: Path) -> List[Path]:
    """
    List all .md files in a local folder.
    """
    return [p for p in folder.iterdir() if p.suffix.lower() == ".md" and p.is_file()]


def clear_chunk_collection(db, col_name: str):
    """
    Delete all documents in the specified collection.
    """
    coll = db[col_name]
    result = coll.delete_many({})
    print(f"[clear_chunk_collection] Cleared '{col_name}', deleted {result.deleted_count} documents.")


def list_pdf_files(folder: Path) -> List[Path]:
    """
    List all .pdf files in a local folder.
    """
    return [p for p in folder.iterdir() if p.suffix.lower() == ".pdf" and p.is_file()]


def clear_collection(db, name: str):
    """
    Delete all documents in the specified collection.
    """
    res = db[name].delete_many({})
    print(f"[clear_collection] {name}: {res.deleted_count} documents removed")


# ─────────────────────── Ingest: Markdown from Local Folder ───────────────────────

def process_md_file(coll, md_path: Path,
                    embedder: OpenAIEmbeddings,
                    splitter: RecursiveCharacterTextSplitter):
    """
    Read a local Markdown file, split into chunks, generate embeddings,
    and insert into the given collection.
    """
    filename = md_path.stem
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"   ✘ Cannot read {md_path}: {e}")
        return

    docs = splitter.create_documents([text])
    chunks = [d.page_content for d in docs]
    vecs = embedder.embed_documents(chunks)

    for i, (chunk, emb) in enumerate(zip(chunks, vecs)):
        doc_id = f"{filename}_chunk_{i}"
        rec = {
            "_id":         doc_id,
            "source":      filename,
            "chunk_index": i,
            "content":     chunk,
            "embedding":   emb
        }
        try:
            coll.insert_one(rec)
            print(f"   ✔ saved chunk #{i} (id={doc_id})")
        except DuplicateKeyError:
            print(f"   ⚠ chunk #{i} already exists, skipped")
        except Exception as e:
            print(f"   ✘ insert error on chunk #{i}: {e}")


def ingest_folder(db, folder: Path):
    """
    Ingest local Markdown files:
      1) Clear file_buffer_db.paper_chunks
      2) Split each .md into chunks, embed, and store in paper_chunks
    """
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return
    files = list_md(folder)
    if not files:
        print("⚠ No .md files in folder.")
        return

    clear_chunk_collection(db, CHUNK_COL)

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    coll = db[CHUNK_COL]
    print(f"ℹ Ingesting {len(files)} markdown files …")
    for md in files:
        print(f"\n→ {md.name}")
        process_md_file(coll, md, embedder, splitter)
    print("\n✅ All markdown ingested.")


# ─────────────────────── Ingest: PDF → GridFS (pipeline_fs) + Metadata ───────────────────────

def ingest_pdfs_to_markdowns(db_buffer, pdf_folder: Path):
    """
    1) Iterate over all PDFs in pdf_folder.
    2) Convert each PDF → raw.md, then remove tables → clean.md.
    3) Save PDF and clean.md into the 'pipeline_fs' GridFS bucket.
    4) Insert metadata into file_buffer_db.pdf_markdowns:
       { _id: "<stem>", filename: "<stem>.pdf", md_file_id: ObjectId }
    """
    if not pdf_folder.exists():
        print(f"❌ PDF folder not found: {pdf_folder}")
        return

    pdfs = list_pdf_files(pdf_folder)
    if not pdfs:
        print("⚠ No PDF files found.")
        return

    # Get the multi-PDF GridFS bucket
    _, fs_multi = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket="pipeline_fs")

    # Metadata collection
    coll_md = db_buffer[MULTI_MD_COL]
    clear_collection(db_buffer, MULTI_MD_COL)

    temp_dir = Path("temp_md")
    temp_dir.mkdir(exist_ok=True)

    print(f"ℹ Processing {len(pdfs)} PDF files …")

    for pdf_path in pdfs:
        name = pdf_path.stem
        print(f"→ {pdf_path.name}")

        raw_md = temp_dir / f"{name}.md"
        clean_md = temp_dir / f"{name}_clean.md"

        # 1) PDF → raw Markdown
        try:
            convert_pdf_to_md(pdf_path, raw_md, page_chunks=True, show_progress=False)
        except Exception as e:
            print(f"   ✘ convert_pdf_to_md failed: {e}")
            continue

        # 2) Remove tables from raw Markdown → clean Markdown
        try:
            remove_markdown_tables(str(raw_md), str(clean_md))
        except Exception as e:
            print(f"   ✘ remove_markdown_tables failed: {e}")
            continue

        # 3) Read clean Markdown text
        try:
            md_text = clean_md.read_text(encoding="utf-8")
        except Exception as e:
            print(f"   ✘ cannot read clean MD: {e}")
            continue

        # 4) Save PDF and clean Markdown into 'pipeline_fs' GridFS
        try:
            pdf_oid = save_to_bucket(fs_multi, pdf_path, pdf_path.name)
            md_oid  = save_to_bucket(fs_multi, clean_md, f"{name}_clean.md")
        except Exception as e:
            print(f"   ✘ GridFS store error: {e}")
            continue

        # 5) Insert metadata record
        rec = {
            "_id":        name,
            "filename":   pdf_path.name,
            "md_file_id": md_oid
        }
        try:
            coll_md.insert_one(rec)
            print("   ✔ Metadata saved to pdf_markdowns & files in pipeline_fs")
        except DuplicateKeyError:
            print("   ⚠ duplicate id in pdf_markdowns, skipped")
        except Exception as e:
            print(f"   ✘ insert error in pdf_markdowns: {e}")

        # Clean up temp files
        try:
            raw_md.unlink()
            clean_md.unlink()
        except Exception:
            pass

    print("\n✅ All PDFs ingested into GridFS.\n")


def ingest_from_gridfs(db_buffer):
    """
    Read all *_clean.md from default GridFS (file_buffer_db),
    split into chunks, embed, and store into file_buffer_db.paper_chunks.
    **Does NOT create a vector index.**
    """
    fs_default = GridFS(db_buffer)  # default bucket
    files = [f for f in fs_default.find() if f.filename.endswith("_clean.md")]
    if not files:
        print("⚠ No *_clean.md files found in default GridFS")
        return

    client = mongo_client()
    db_buf = client[FILE_BUFFER_DB]

    # Clear existing chunks before ingestion
    clear_chunk_collection(db_buf, CHUNK_COL)
    coll = db_buf[CHUNK_COL]

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    print(f"ℹ Vectorizing {len(files)} Markdown files from default GridFS …")
    for f in files:
        name = Path(f.filename).stem.replace("_clean", "")
        md_text = f.read().decode("utf-8")
        docs = splitter.create_documents([md_text])
        chunks = [d.page_content for d in docs]
        vecs = embedder.embed_documents(chunks)

        for i, (text, emb) in enumerate(zip(chunks, vecs)):
            rec = {
                "_id":         f"{name}_chunk_{i}",
                "source":      f.filename,
                "chunk_index": i,
                "content":     text,
                "embedding":   emb
            }
            try:
                coll.insert_one(rec)
                print(f"   ✔ Chunk #{i} inserted for {f.filename}")
            except DuplicateKeyError:
                print(f"   ⚠ Chunk #{i} already exists — skipped")
            except Exception as e:
                print(f"   ✘ Error inserting chunk #{i}: {e}")

    print("✅ All Markdown chunks vectorized and stored in file_buffer_db.paper_chunks.")
    client.close()


# ─────────────────────── Vectorize: Markdown from pdf_markdowns → multi_paper_chunks ───────────────────────

def ingest_multi_chunks(db_arxiv, db_buffer):
    """
    1) Query file_buffer_db.pdf_markdowns:
         { _id: name, filename: pdf, md_file_id: ObjectId }
    2) For each entry: read clean Markdown from 'pipeline_fs' GridFS,
       split into chunks, embed, and save into arxiv_db.multi_paper_chunks:
       { _id: "<name>_chunk_<i>", source_pdf: "<filename>.pdf", chunk_index: i, content, embedding }
    """
    coll_md = db_buffer[MULTI_MD_COL]
    docs = list(coll_md.find({}))
    if not docs:
        print("⚠ pdf_markdowns is empty — nothing to vectorize.")
        return

    clear_collection(db_arxiv, MULTI_CHUNK_COL)
    coll_chunks = db_arxiv[MULTI_CHUNK_COL]

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Get the 'pipeline_fs' GridFS bucket for reading clean Markdown
    _, fs_multi = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket="pipeline_fs")

    print(f"ℹ Vectorizing {len(docs)} Markdown files from pipeline_fs GridFS …")
    for doc in docs:
        pdf_id   = doc["_id"]          # stem
        pdf_name = doc["filename"]     # "<stem>.pdf"
        md_oid   = doc["md_file_id"]   # ObjectId for clean Markdown

        try:
            md_bytes = fs_multi.get(md_oid).read()
            md_text  = md_bytes.decode("utf-8")
        except Exception as e:
            print(f"   ⚠ cannot read MD from GridFS (id={md_oid}): {e}")
            continue

        chunks = [c.page_content for c in splitter.create_documents([md_text])]
        vecs = embedder.embed_documents(chunks)

        for i, (text, emb) in enumerate(zip(chunks, vecs)):
            rec = {
                "_id":         f"{pdf_id}_chunk_{i}",
                "source_pdf":  pdf_name,
                "chunk_index": i,
                "content":     text,
                "embedding":   emb
            }
            try:
                coll_chunks.insert_one(rec)
                print(f"   ✔ chunk #{i} saved (id={rec['_id']})")
            except DuplicateKeyError:
                print(f"   ⚠ chunk #{i} already exists, skipped")
            except Exception as e:
                print(f"   ✘ insert error on chunk #{i}: {e}")

    print("\n✅ All cleaned Markdown chunks vectorized.\n")


# ─────────────────────── Build Vector Index ───────────────────────

def build_vector_index(db, collection: str, idx_name: str, field: str = "embedding"):
    """
    Create or rebuild a vector search index on the specified collection.
    Assumes documents already contain a field 'embedding'.
    """
    import pymongo
    coll = db[collection]

    # 1) Drop existing index if present
    try:
        indexes = [idx["name"] for idx in coll.list_search_indexes()]
    except (OperationFailure, AttributeError):
        indexes = []
    if idx_name in indexes:
        print(f"⚠ Index '{idx_name}' exists — dropping …")
        try:
            coll.drop_search_index(name=idx_name)
            # Wait a moment for the index to drop
            time.sleep(2)
        except OperationFailure as e:
            print(f"✘ Failed to drop index '{idx_name}': {e}")

    # 2) Check that there's at least one document with the 'embedding' field
    sample = coll.find_one({field: {"$exists": True}}, {field: 1})
    if not sample:
        print(f"✘ Collection '{collection}' has no '{field}' field. Skipping index creation.")
        return

    dims = len(sample[field])
    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": field,
                    "numDimensions": dims,
                    "similarity": "cosine",
                    "quantization": "scalar"
                }
            ]
        },
        name=idx_name,
        type="vectorSearch"
    )

    try:
        created = coll.create_search_index(model=model)
        print(f"➤ Building index '{created}' …")
    except pymongo.errors.OperationFailure as e:
        if e.code == 68:  # IndexAlreadyExists
            print(f"ℹ Index '{idx_name}' already requested by another process; waiting …")
            created = idx_name
        else:
            print(f"✘ Cannot create index '{idx_name}': {e}")
            return

    # 3) Wait until the index is queryable
    print("⏳ Waiting for index to become ready …")
    while True:
        info = list(coll.list_search_indexes(created))
        if info and info[0].get("queryable", False):
            print(f"✔ Index '{created}' is ready.")
            break
        time.sleep(2)


def clear_all_before_ingest(db_buffer, bucket: str | None = None):
    """
    Clear the specified GridFS bucket and the pdf_markdowns collection.
    Arguments:
        db_buffer: file_buffer_db database handle
        bucket:
          - None → clear default GridFS (fs)
          - "pipeline_fs" → clear GridFS with prefix "pipeline_fs"
    """
    from gridfs import GridFS

    fs = GridFS(db_buffer, collection=bucket) if bucket else GridFS(db_buffer)
    label = f"{bucket or 'Default'} GridFS"

    clear_collection(db_buffer, MULTI_MD_COL)

    files = list(fs.find())
    for f in files:
        fs.delete(f._id)

    print(f"✔ {label} cleared ({len(files)} files)")


# ─────────────────────── CLI MENU ───────────────────────

def menu():
    client     = mongo_client()
    db_arxiv   = client[ABSTRACT_DB]
    db_buffer  = client[FILE_BUFFER_DB]
    # Default GridFS buckets:
    _, fs_default = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket=None)
    _, fs_multi   = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket="pipeline_fs")

    while True:
        print("\n=== Markdown/PDF → MongoDB Utility ===")
        print("1) Ingest markdown chunks from folder")
        print("2) Ingest single-PDF via GridFS → file_buffer_db.paper_chunks")
        print("3) Create/refresh vector index for abstracts (arxiv_db.arxiv_metadata)")
        print("4) Create/refresh vector index for single-PDF chunks (file_buffer_db.paper_chunks)")
        print("5) Ingest multiple PDFs → GridFS → multi_paper_chunks")
        print("6) Create/refresh vector index for multi_paper_chunks")
        print("7) Clear default GridFS bucket (fs.files/fs.chunks)")
        print("8) Clear pipeline_fs GridFS bucket (pipeline_fs.files/pipeline_fs.chunks)")
        print("9) Clear pdf_markdowns collection")
        print("0) Exit")

        choice = input("Select option (0-9): ").strip()

        if choice == "1":
            inp = input(f"MD folder [{DEFAULT_MD_FOLDER}]: ").strip()
            folder = Path(inp) if inp else DEFAULT_MD_FOLDER
            ingest_folder(db_buffer, folder)

        elif choice == "2":
            ingest_from_gridfs(db_buffer)

        elif choice == "3":
            build_vector_index(db_arxiv, ABS_COL, ABS_INDEX)

        elif choice == "4":
            build_vector_index(db_buffer, CHUNK_COL, CHNK_INDEX)

        elif choice == "5":
            inp_pdf = input(f"PDF folder [{DEFAULT_PDF_FOLDER}]: ").strip()
            pdf_folder = Path(inp_pdf) if inp_pdf else DEFAULT_PDF_FOLDER
            ingest_pdfs_to_markdowns(db_buffer, pdf_folder)
            ingest_multi_chunks(db_arxiv, db_buffer)

        elif choice == "6":
            build_vector_index(db_arxiv, MULTI_CHUNK_COL, MULTI_CHNK_INDEX)

        elif choice == "7":
            count_files = fs_default.find().count()
            for f in fs_default.find():
                fs_default.delete(f._id)
            print(f"✔ Default GridFS cleared ({count_files} files)")

        elif choice == "8":
            count_files = fs_multi.find().count()
            for f in fs_multi.find():
                fs_multi.delete(f._id)
            print(f"✔ Pipeline_fs GridFS cleared ({count_files} files)")

        elif choice == "9":
            clear_collection(db_buffer, MULTI_MD_COL)

        elif choice == "0":
            print("Bye!")
            break

        else:
            print("Invalid choice (0–9).")


if __name__ == "__main__":
    if not MONGODB_URI:
        print("✘ MONGODB_URI not set in environment (.env)")
        sys.exit(1)
    menu()
