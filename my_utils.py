# C:\Work\diplom2\rag_on_papers\my_utils.py

import os
import sys
import tempfile
import shutil
import pathlib
import streamlit as st

from functools import lru_cache
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import gridfs

from langchain_openai import ChatOpenAI

from src.data_processing.pdf_to_md import convert_pdf_to_md
from src.data_processing.removetables import remove_markdown_tables
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()


# ---------- Health-check helpers ---------- #

@lru_cache  # avoid a full round-trip on every Streamlit rerun
def check_mongo_connection() -> str:
    """
    Ping MongoDB (using MONGODB_URI) and report status.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return "❌ MongoDB: MONGODB_URI not set"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3_000)
        client.admin.command("ping")
        count_msg = ""
        try:
            # Попытка прочитать коллекцию arxiv_metadata в базе arxiv_db
            count = client["arxiv_db"]["arxiv_metadata"].count_documents({})
            count_msg = f"  •  docs loaded: {count}"
        except Exception:
            pass
        return f"✅ Connected{count_msg}"
    except ConnectionFailure as ce:
        return f"❌ {ce}"
    except Exception as e:
        return f"❌ {e}"

def check_mongo_connection_pdfs() -> str:
    """
    Ping MongoDB (using MONGODB_URI) and report status.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return "❌ MongoDB: MONGODB_URI not set"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3_000)
        client.admin.command("ping")
        return "✅ Connected"
    except ConnectionFailure as ce:
        return f"❌ {ce}"
    except Exception as e:
        return f"❌ {e}"

@lru_cache
def check_llm_connection() -> str:
    """
    One-token round-trip to verify OpenAI (ChatOpenAI) credentials.
    """
    try:
        _ = ChatOpenAI(model="gpt-4o-mini").invoke("ping")
        return "✅ Connected"
    except Exception as e:
        return f"❌ {e}"


def render_status(label: str, status_msg: str):
    if status_msg.startswith("✅"):
        st.markdown(f"🟢 **{label}**: {status_msg[2:].strip()}")
    elif status_msg.startswith("❌"):
        st.markdown(f"🔴 **{label}**: {status_msg[2:].strip()}")
    else:
        st.markdown(f"🟡 **{label}**: {status_msg.strip()}")


def clear_folder(folder_path: str, preserve_files: str = None) -> str:
    """
    Удаляет содержимое папки folder_path, оставляя только файлы из comma-separated строки preserve_files.
    Если папки нет, создаёт её.
    Возвращает путь folder_path (или "" при ошибке).
    """
    preserve_list = []
    if preserve_files:
        preserve_list = [x.strip() for x in preserve_files.split(",")]

    try:
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename in preserve_list:
                    continue
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"[clear_folder] Folder `{folder_path}` cleared (preserved: {preserve_list}).")
        else:
            os.makedirs(folder_path, exist_ok=True)
            print(f"[clear_folder] Folder `{folder_path}` created.")
        return folder_path
    except Exception as e:
        print(f"[clear_folder] Error: {e}")
        return ""




class StreamlitLogger:
    """
    Перехватывает print-выводы и пишет их в st.session_state с ключом protocol_output_{page_id}.
    """

    def __init__(self, page_id):
        self.original_stdout = sys.stdout
        self.log = ""
        self.page_id = page_id

    def write(self, message):
        self.log += message
        key = f'protocol_output_{self.page_id}'
        if key not in st.session_state:
            st.session_state[key] = ""
        st.session_state[key] += message

    def flush(self):
        pass

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout


def save_uploaded_file(uploaded_file, save_path: str) -> bool:
    """
    Сохраняет uploaded_file (Streamlit UploadedFile) в save_path.
    Возвращает True, если успешно, иначе False.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


# ---------- GridFS helpers ---------- #

def get_gridfs_collections():
    """
    Возвращает (client, db, fs) для GridFS, всегда используя БД 'file_buffer_db',
    независимо от того, что указано в URI.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not set in environment")

    client = MongoClient(uri)
    db = client["file_buffer_db"]  # ← всегда явно указываем нужную БД
    fs = gridfs.GridFS(db)
    return client, db, fs



def check_gridfs_connection() -> str:
    """
    Пробует подключиться к GridFS и подсчитать, сколько файлов уже сохранено.
    Возвращает строку статуса.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return "❌ GridFS: MONGODB_URI not set"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3_000)
        client.admin.command("ping")
        db = client["file_buffer_db"]
        count = db.fs.files.count_documents({})
        return f"✅ GridFS: Connected. Stored files: {count}"
    except Exception as e:
        return f"❌ GridFS: {e}"


def save_pdf_to_gridfs(pdf_bytes: bytes, filename: str) -> str:
    """
    Сохраняет PDF (bytes) в GridFS, возвращает строку с ObjectId.
    """
    if not pdf_bytes:
        raise ValueError("No PDF bytes to save")
    client, db, fs = get_gridfs_collections()
    file_id = fs.put(pdf_bytes, filename=filename, contentType="application/pdf")
    return str(file_id)


def save_text_to_gridfs(text: str, filename: str) -> str:
    """
    Сохраняет текст (Markdown) в GridFS, возвращает строку с ObjectId.
    """
    if text is None:
        raise ValueError("No text to save")
    client, db, fs = get_gridfs_collections()
    file_id = fs.put(text.encode("utf-8"), filename=filename, contentType="text/markdown")
    return str(file_id)

def clear_all_files_from_gridfs() -> int:
    """
    Удаляет все файлы из GridFS (fs.files и fs.chunks).
    Возвращает количество удалённых файлов.
    """
    client, db, fs = get_gridfs_collections()
    files = list(db.fs.files.find({}))
    deleted_count = 0
    for file_doc in files:
        fs.delete(file_doc["_id"])
        deleted_count += 1
    return deleted_count

def download_all_files_from_gridfs(target_folder="downloaded_from_gridfs") -> int:
    """
    Скачивает все файлы из GridFS в указанную локальную папку.
    Возвращает количество загруженных файлов.
    """
    client, db, fs = get_gridfs_collections()
    os.makedirs(target_folder, exist_ok=True)
    files = list(db.fs.files.find({}))
    saved_count = 0

    for file_doc in files:
        file_id = file_doc["_id"]
        filename = file_doc.get("filename", f"file_{file_id}.bin")
        local_path = os.path.join(target_folder, filename)

        try:
            with open(local_path, "wb") as f_out:
                f_out.write(fs.get(file_id).read())
            print(f"✔️ Saved: {filename}")
            saved_count += 1
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")

    print(f"\n📁 Files saved to: {os.path.abspath(target_folder)}")
    return saved_count


# ---------- Console testing (main) ---------- #

def process_pdf_via_console(pdf_path: str):
    """
    Консольный метод: принимает путь к PDF, выполняет следующие шаги:
      ...
    """
    if not os.path.isfile(pdf_path):
        print(f"❌ Файл не найден: {pdf_path}")
        return

    # 🔥 Удаляем все старые файлы из GridFS
    print("🧹 Очищаем GridFS от предыдущих файлов...")
    try:
        deleted = clear_all_files_from_gridfs()
        print(f"   ✔️ Удалено файлов: {deleted}")
    except Exception as e:
        print(f"❌ Не удалось очистить GridFS: {e}")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    base_name = pathlib.Path(pdf_path).stem
    orig_filename = f"{base_name}.pdf"

    print("1) Сохраняем оригинальный PDF в GridFS...")
    try:
        pdf_id = save_pdf_to_gridfs(pdf_bytes, orig_filename)
    except Exception as e:
        print(f"❌ Ошибка при сохранении PDF в GridFS: {e}")
        return
    print(f"   ✔️ PDF saved, id = {pdf_id}")

    # Создаём временную папку
    tmp_dir = tempfile.mkdtemp(prefix="pdf_to_md_")
    tmp_pdf_path = os.path.join(tmp_dir, orig_filename)
    with open(tmp_pdf_path, "wb") as f_tmp:
        f_tmp.write(pdf_bytes)
    print(f"   📂 Временный PDF для конвертации: {tmp_pdf_path}")

    # Конвертируем PDF → Markdown
    tmp_md_path = os.path.join(tmp_dir, f"{base_name}.md")
    print("2) Конвертируем PDF → Markdown...")
    try:
        convert_pdf_to_md(tmp_pdf_path, tmp_md_path, show_progress=False)
    except Exception as e:
        print(f"❌ Ошибка при конвертации PDF → MD: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Markdown сохранён локально: {tmp_md_path}")

    # Сохраняем Markdown в GridFS
    with open(tmp_md_path, "r", encoding="utf-8") as f_md:
        md_text = f_md.read()
    md_filename = f"{base_name}.md"
    print("3) Сохраняем Markdown в GridFS...")
    try:
        md_id = save_text_to_gridfs(md_text, md_filename)
    except Exception as e:
        print(f"❌ Ошибка при сохранении Markdown в GridFS: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Markdown saved, id = {md_id}")

    # «Очищаем» Markdown от таблиц
    tmp_md_clean_path = os.path.join(tmp_dir, f"{base_name}_clean.md")
    print("4) Удаляем таблицы из Markdown...")
    try:
        remove_markdown_tables(tmp_md_path, tmp_md_clean_path)
    except Exception as e:
        print(f"❌ Ошибка при удалении таблиц из Markdown: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Clean Markdown saved локально: {tmp_md_clean_path}")

    # Сохраняем Clean Markdown в GridFS
    with open(tmp_md_clean_path, "r", encoding="utf-8") as f_clean:
        clean_md_text = f_clean.read()
    clean_md_filename = f"{base_name}_clean.md"
    print("5) Сохраняем Clean Markdown в GridFS...")
    try:
        clean_md_id = save_text_to_gridfs(clean_md_text, clean_md_filename)
    except Exception as e:
        print(f"❌ Ошибка при сохранении Clean Markdown в GridFS: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Clean Markdown saved, id = {clean_md_id}")

    # Удаляем временную папку
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"6) Временные файлы удалены: {tmp_dir}")

    # Выводим все ID
    print("\n=== РЕЗУЛЬТАТЫ ===")
    print(f"• PDF ObjectId:         {pdf_id}")
    print(f"• Markdown ObjectId:    {md_id}")
    print(f"• Clean Markdown ObjectId: {clean_md_id}")
    print("=======================")


# ───── ВСТАВЬТЕ / ЗАМЕНИТЕ "main" В КОНЦЕ  my_utils.py ─────
def main():
    print("\n=== PDF → GridFS Test Utility ===")
    while True:
        print("\n1) Check MongoDB connection")
        print("2) Check GridFS connection")
        print("3) Process PDF and save (PDF + MD + Clean MD) to GridFS")
        print("4) Download all files from GridFS to local folder")
        print("0) Exit")
        choice = input("Select option (0-4): ").strip()

        if choice == "1":
            print(check_mongo_connection())

        elif choice == "2":
            print(check_gridfs_connection())

        elif choice == "3":
            pdf_path = input("Enter full path to PDF: ").strip().strip('"').strip("'")
            process_pdf_via_console(pdf_path)

        elif choice == "4":
            count = download_all_files_from_gridfs()
            print(f"\n✅ Downloaded {count} files from GridFS.")

        elif choice == "0":
            print("Bye!")
            break

        else:
            print("Invalid option. Please choose 0–4.")



if __name__ == "__main__":
    main()
