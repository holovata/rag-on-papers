#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенный скрипт для:
 1) Ингеста Markdown из локальных файлов (как раньше).
 2) Ингеста очищённого Markdown из GridFS (как раньше).
 3) Создания/обновления векторных индексов для абстрактов и чанков (как раньше).
 4) Новая функциональность: Ингестирование нескольких PDF-файлов за раз:
    4.1) Конвертация каждого PDF→Markdown (убираем таблицы).
    4.2) Сохранение «очищённого» Markdown и PDF в отдельном GridFS-бакете.
    4.3) Христакание метаданных Markdown-файлов в коллекцию pdf_markdowns.
    4.4) Чтение очищенного Markdown из GridFS, чанкинг, векторизация и запись во
         коллекцию multi_paper_chunks (arxiv_db).
    4.5) Создание/обновление векторного индекса для multi_paper_chunks.
 5) Поддержка двух GridFS-бакетов:
    • default (fs.files/fs.chunks) — для «single-PDF» пайплайна при необходимости.
    • pipeline_fs (pipeline_fs.files/pipeline_fs.chunks) — для мульти-PDF пайплайна.
 6) Возможность очистки каждого бакета и коллекции pdf_markdowns через меню.
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
from pymongo.errors import DuplicateKeyError

from gridfs import GridFS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Предполагаем, что эти функции реализованы в ваших модулях:
from src.data_processing.pdf_to_md import convert_pdf_to_md
from src.data_processing.removetables import remove_markdown_tables

load_dotenv()  # подхватываем MONGODB_URI из .env

# ────────────────────── Константы ──────────────────────
MONGODB_URI     = os.getenv("MONGODB_URI")
DB_NAME         = "arxiv_db"
ABS_COL         = "arxiv_metadata"
CHUNK_COL       = "paper_chunks"
MULTI_CHUNK_COL = "multi_paper_chunks"   # коллекция для чанков мульти-PDF
MULTI_MD_COL    = "pdf_markdowns"        # коллекция метаданных Markdown
FILE_BUFFER_DB  = "file_buffer_db"       # БД-буфер для GridFS

DEFAULT_MD_FOLDER  = Path(__file__).resolve().parent.parent / "data" / "dl_papers_md"
DEFAULT_PDF_FOLDER = Path(r"C:\Work\diplom2\rag_on_papers\data\downloaded_papers")

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200

ABS_INDEX        = "embedding_vector_index"
CHNK_INDEX       = "chunk_embedding_index"
MULTI_CHNK_INDEX = "multi_chunk_embedding_index"
# ────────────────────────────────────────────────────────


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
        print(f"❌ MongoDB connection failed: {e}")
        sys.exit(1)


def get_gridfs_bucket(
    mongo_uri: str,
    db_name: str,
    bucket: str | None = None
) -> Tuple[MongoClient, GridFS]:
    """
    Возвращает (MongoClient, GridFS) для указанного бакета.
    bucket=None  → стандартные fs.files/fs.chunks
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
    Сохраняет файл в указанный GridFS-бакет (удаляет старую версию, если overwrite=True).
    Возвращает ObjectId сохранённого файла.
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


# ────────────────────── MD-helpers (существующие) ──────────────────────

def list_md(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() == ".md" and p.is_file()]

def clear_chunk_collection(db, col_name: str):
    """
    Очищает указанную коллекцию перед записью новых данных.
    """
    coll = db[col_name]
    result = coll.delete_many({})
    print(f"[clear_chunk_collection] Cleared '{col_name}', deleted {result.deleted_count} documents.")

def list_pdf_files(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.suffix.lower() == ".pdf" and p.is_file()]

def clear_collection(db, name: str):
    res = db[name].delete_many({})
    print(f"[clear_collection] {name}: {res.deleted_count} documents removed")


# ─────────────────────── Ingest: Markdown из локальной папки ───────────────────────

def process_md_file(coll, md_path: Path,
                    embedder: OpenAIEmbeddings,
                    splitter: RecursiveCharacterTextSplitter):
    """
    Берёт один локальный Markdown-файл, разбивает на чанки, получает эмбеддинги
    и сохраняет в коллекцию coll (имя коллекции уже должно быть установлено).
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
    Ингест локальных Markdown-файлов:
      1) Очищает коллекцию “paper_chunks”
      2) Разбивает каждый .md на чанки и сохраняет эмбеддинги
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


# ─────────────────────── Ingest: PDF → GridFS (pipeline_fs) + metadata ───────────────────────

def ingest_pdfs_to_markdowns(db_buffer, pdf_folder: Path):
    """
    1) Проходит по всем PDF-файлам в pdf_folder.
    2) Конвертирует каждый PDF→raw.md, затем удаляет таблицы→clean.md.
    3) Сохраняет PDF и clean.md в GridFS-бакете 'pipeline_fs'.
    4) В коллекцию file_buffer_db.pdf_markdowns записывает метаданные:
       { _id: "<имя>", filename: "<имя>.pdf", md_file_id: ObjectId }
    """
    if not pdf_folder.exists():
        print(f"❌ PDF folder not found: {pdf_folder}")
        return

    pdfs = list_pdf_files(pdf_folder)
    if not pdfs:
        print("⚠ No PDF files found.")
        return

    # Берём GridFS-бакет для мульти-PDF пайплайна
    _, fs_multi = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket="pipeline_fs")

    # Коллекция метаданных Markdown
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

        # 1. PDF → Markdown
        try:
            convert_pdf_to_md(pdf_path, raw_md, page_chunks=True, show_progress=False)
        except Exception as e:
            print(f"   ✘ convert_pdf_to_md failed: {e}")
            continue

        # 2. Удаление таблиц (чтобы не сохранять их в чистом MD)
        try:
            remove_markdown_tables(str(raw_md), str(clean_md))
        except Exception as e:
            print(f"   ✘ table-removal failed: {e}")
            continue

        # 3. Чтение очищенного Markdown
        try:
            md_text = clean_md.read_text(encoding="utf-8")
        except Exception as e:
            print(f"   ✘ cannot read clean MD: {e}")
            continue

        # 4. Сохранение PDF и MD в GridFS-бакете 'pipeline_fs'
        try:
            pdf_oid = save_to_bucket(fs_multi, pdf_path, pdf_path.name)
            md_oid  = save_to_bucket(fs_multi, clean_md, f"{name}_clean.md")
        except Exception as e:
            print(f"   ✘ GridFS store error: {e}")
            continue

        # 5. Вставляем метаданные в коллекцию pdf_markdowns
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

        # Удаляем временные файлы
        try:
            raw_md.unlink()
            clean_md.unlink()
        except Exception:
            pass

    print("\n✅ All PDFs ingested into GridFS.\n")

def ingest_from_gridfs(db):
    """
    Читає усі *_clean.md з default GridFS, ділить на чанки,
    робить embedding і зберігає у arxiv_db.paper_chunks.
    """
    fs_default = GridFS(db)                  # default bucket
    files = [f for f in fs_default.find() if f.filename.endswith("_clean.md")]
    if not files:
        print("⚠ No *_clean.md files found in default GridFS")
        return

    clear_chunk_collection(db, CHUNK_COL)    # видаляємо старі чанки
    coll = db[CHUNK_COL]

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    print(f"ℹ Vectorizing {len(files)} Markdown files from default GridFS …")
    for f in files:
        name = Path(f.filename).stem.replace("_clean", "")
        md_text = f.read().decode("utf-8")
        chunks = [c.page_content for c in splitter.create_documents([md_text])]
        vecs = embedder.embed_documents(chunks)

        for i, (text, emb) in enumerate(zip(chunks, vecs)):
            rec = {
                "_id":         f"{name}_chunk_{i}",
                "source":      f.filename,          # з якого MD-файлу
                "chunk_index": i,
                "content":     text,
                "embedding":   emb,
            }
            try:
                coll.insert_one(rec)
            except DuplicateKeyError:
                pass
    print("✅ All Markdown chunks vectorized and stored.")

# ─────────────────────── Vectorize: Markdown из pdf_markdowns → multi_paper_chunks ───────────────────────

def ingest_multi_chunks(db_arxiv, db_buffer):
    """
    1) Берёт все документы из file_buffer_db.pdf_markdowns:
         { _id: name, filename: pdf, md_file_id: ObjectId }
    2) Для каждого: читает чистый Markdown из GridFS-бакета 'pipeline_fs' по md_file_id,
       делит на чанки, делает embedding и сохраняет в arxiv_db.multi_paper_chunks:
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

    # Берём GridFS-бакет 'pipeline_fs' для чтения чистого Markdown
    _, fs_multi = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket="pipeline_fs")

    print(f"ℹ Vectorizing {len(docs)} Markdown files from pipeline_fs GridFS …")

    for doc in docs:
        pdf_id   = doc["_id"]          # stem
        pdf_name = doc["filename"]     # "<stem>.pdf"
        md_oid   = doc["md_file_id"]   # ObjectId для чистого Markdown

        # Читаем Markdown из GridFS
        try:
            md_bytes = fs_multi.get(md_oid).read()
            md_text  = md_bytes.decode("utf-8")
        except Exception as e:
            print(f"   ⚠ cannot read MD from GridFS (id={md_oid}): {e}")
            continue

        # Чанкинг и векторизация
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


# ─────────────────────── Build vector index (не меняется) ───────────────────────

def build_vector_index(db, collection: str, idx_name: str, field: str = "embedding"):
    import pymongo
    coll = db[collection]

    # 1) Дроп, если существует
    exists = any(idx["name"] == idx_name for idx in coll.list_search_indexes())
    if exists:
        print(f"⚠ Index '{idx_name}' exists — dropping …")
        coll.drop_search_index(name=idx_name)
        while any(idx["name"] == idx_name for idx in coll.list_search_indexes()):
            time.sleep(2)
        print(f"✔ Dropped index '{idx_name}'")

    # 2) Проверка, что есть хотя бы один документ с embedding
    sample = coll.find_one({field: {"$exists": True}}, {field: 1})
    if not sample:
        print(f"✘ Collection '{collection}' has no '{field}' field.")
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
            print(f"ℹ Index already recreated by another process; waiting …")
            created = idx_name
        else:
            print(f"✘ Cannot create index: {e}")
            return

    # 3) Ждём готовности
    print("⏳ Waiting index to become ready …")
    while True:
        info = list(coll.list_search_indexes(created))
        if info and info[0].get("queryable", False):
            print(f"✔ Index '{created}' is ready.")
            break
        time.sleep(4)


def clear_all_before_ingest(db_buffer, bucket: str | None = None):
    """
    Очищает указанный GridFS-бакет и коллекцию markdown-метаданных.

    Аргументы:
        db_buffer: об'єкт бази даних (file_buffer_db)
        bucket:
          - None → очистить default GridFS (fs)
          - "pipeline_fs" → очистить GridFS с префиксом "pipeline_fs"
    """
    from gridfs import GridFS

    fs = GridFS(db_buffer, collection=bucket) if bucket else GridFS(db_buffer)
    label = f"{bucket or 'Default'} GridFS"

    clear_collection(db_buffer, MULTI_MD_COL)

    files = list(fs.find())
    for f in files:
        fs.delete(f._id)

    print(f"✔ {label} cleared ({len(files)} files)")


# ─────────────────────── CLI loop (расширенный) ───────────────────────

def menu():
    client     = mongo_client()
    db_arxiv   = client[DB_NAME]
    db_buffer  = client[FILE_BUFFER_DB]
    # Бакеты по умолчанию:
    _, fs_default = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket=None)
    _, fs_multi   = get_gridfs_bucket(MONGODB_URI, FILE_BUFFER_DB, bucket="pipeline_fs")

    while True:
        print("\n=== Markdown/PDF → MongoDB Utility ===")
        print("1) Ingest markdown chunks from folder (single-PDF)")
        print("3) Create/refresh vector index for abstracts")
        print("4) Create/refresh vector index for markdown chunks")
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
            ingest_folder(db_arxiv, folder)

        elif choice == "3":
            build_vector_index(db_arxiv, ABS_COL, ABS_INDEX)

        elif choice == "4":
            build_vector_index(db_arxiv, CHUNK_COL, CHNK_INDEX)

        elif choice == "5":
            inp_pdf = input(f"PDF folder [{DEFAULT_PDF_FOLDER}]: ").strip()
            pdf_folder = Path(inp_pdf) if inp_pdf else DEFAULT_PDF_FOLDER
            ingest_pdfs_to_markdowns(db_buffer, pdf_folder)
            ingest_multi_chunks(db_arxiv, db_buffer)

        elif choice == "6":
            build_vector_index(db_arxiv, MULTI_CHUNK_COL, MULTI_CHNK_INDEX)

        elif choice == "7":
            # Удаляем все документы из default GridFS
            count_files = fs_default.find().count()
            for f in fs_default.find():
                fs_default.delete(f._id)
            print(f"✔ Default GridFS cleared ({count_files} files)")

        elif choice == "8":
            # Удаляем все документы из pipeline_fs GridFS
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
