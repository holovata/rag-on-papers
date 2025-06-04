# C:\Work\diplom2\rag_on_papers\my_utils.py

import os
import sys
import tempfile
import shutil
import pathlib
import io
import contextlib
import streamlit as st

from functools import lru_cache
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import gridfs

from langchain_openai import ChatOpenAI

from src.data_processing.pdf_to_md import convert_pdf_to_md
from src.data_processing.removetables import remove_markdown_tables
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env


# ---------- Health-check Helpers ---------- #

@lru_cache  # Cache result to avoid re-pinging on every rerun
def check_mongo_connection() -> str:
    """
    Ping MongoDB using MONGODB_URI and report status.
    Attempts to count documents in arxiv_db.arxiv_metadata for extra info.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return "❌ MongoDB: MONGODB_URI not set"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        count_msg = ""
        try:
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
    Ping MongoDB using MONGODB_URI and report status (for PDF pipeline).
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return "❌ MongoDB: MONGODB_URI not set"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return "✅ Connected"
    except ConnectionFailure as ce:
        return f"❌ {ce}"
    except Exception as e:
        return f"❌ {e}"


@lru_cache
def check_llm_connection() -> str:
    """
    Perform a minimal round-trip with the ChatOpenAI client to verify OpenAI credentials.
    """
    try:
        _ = ChatOpenAI(model="gpt-4o-mini").invoke("ping")
        return "✅ Connected"
    except Exception as e:
        return f"❌ {e}"


def render_status(label: str, status_msg: str):
    """
    Display a status indicator in Streamlit based on status_msg prefix.
    """
    if status_msg.startswith("✅"):
        st.markdown(f"🟢 **{label}**: {status_msg[2:].strip()}")
    elif status_msg.startswith("❌"):
        st.markdown(f"🔴 **{label}**: {status_msg[2:].strip()}")
    else:
        st.markdown(f"🟡 **{label}**: {status_msg.strip()}")


# ---------- Folder Utilities ---------- #

def clear_folder(folder_path: str, preserve_files: str = None) -> str:
    """
    Delete all contents of folder_path except for files listed in preserve_files.
    If the folder does not exist, create it.
    Returns the path on success, or an empty string on failure.
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


# ---------- Streamlit Logger ---------- #

class StreamlitLogger:
    """
    Capture print outputs and write them directly into st.session_state
    under the key 'protocol_output_{page_id}'.
    """

    def __init__(self, page_id: str):
        self.original_stdout = sys.stdout
        self.log = ""
        self.page_id = page_id

    def write(self, message: str):
        """
        Append message to both an internal buffer and Streamlit session state.
        """
        self.log += message
        key = f"protocol_output_{self.page_id}"
        if key not in st.session_state:
            st.session_state[key] = ""
        st.session_state[key] += message

    def flush(self):
        """
        No-op to satisfy file-like interface.
        """
        pass

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout


# ---------- File Upload Helpers ---------- #

def save_uploaded_file(uploaded_file, save_path: str) -> bool:
    """
    Save a Streamlit UploadedFile to a local path.
    Returns True on success, False on failure.
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


# ---------- GridFS Helpers ---------- #

def get_gridfs_collections():
    """
    Return (client, db, fs) for the 'file_buffer_db' database,
    regardless of which database is specified in the URI.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not set in environment")

    client = MongoClient(uri)
    db = client["file_buffer_db"]
    fs = gridfs.GridFS(db)
    return client, db, fs


def check_gridfs_connection() -> str:
    """
    Attempt to connect to GridFS and count stored files.
    Return a status string.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return "❌ GridFS: MONGODB_URI not set"

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client["file_buffer_db"]
        count = db.fs.files.count_documents({})
        return f"✅ GridFS: Connected. Stored files: {count}"
    except Exception as e:
        return f"❌ GridFS: {e}"


def save_pdf_to_gridfs(pdf_bytes: bytes, filename: str) -> str:
    """
    Save a PDF (in bytes) into GridFS, returning the ObjectId as a string.
    """
    if not pdf_bytes:
        raise ValueError("No PDF bytes to save")
    client, db, fs = get_gridfs_collections()
    file_id = fs.put(pdf_bytes, filename=filename, contentType="application/pdf")
    return str(file_id)


def save_text_to_gridfs(text: str, filename: str) -> str:
    """
    Save text (Markdown) into GridFS, returning the ObjectId as a string.
    """
    if text is None:
        raise ValueError("No text to save")
    client, db, fs = get_gridfs_collections()
    file_id = fs.put(text.encode("utf-8"), filename=filename, contentType="text/markdown")
    return str(file_id)


def clear_all_files_from_gridfs() -> int:
    """
    Delete all files from the default GridFS bucket (fs.files and fs.chunks).
    Return the number of deleted files.
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
    Download all files from GridFS into a local folder.
    Return the number of files downloaded.
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


# ---------- Console Testing (Main) ---------- #

def process_pdf_via_console(pdf_path: str):
    """
    Console utility: given a path to a PDF, perform:
      1) Clear previous files from GridFS
      2) Save the original PDF into GridFS
      3) Convert PDF → Markdown (raw .md file)
      4) Save raw Markdown into GridFS
      5) Clean Markdown (remove tables) → save clean .md locally
      6) Save clean Markdown into GridFS
      7) Delete temporary files
      8) Print resulting ObjectIds
    """
    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    # 1) Clear existing files from GridFS
    print("🧹 Clearing GridFS of previous files...")
    try:
        deleted = clear_all_files_from_gridfs()
        print(f"   ✔️ Deleted files: {deleted}")
    except Exception as e:
        print(f"❌ Failed to clear GridFS: {e}")
        return

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    base_name = pathlib.Path(pdf_path).stem
    orig_filename = f"{base_name}.pdf"

    # 2) Save original PDF into GridFS
    print("1) Saving original PDF to GridFS...")
    try:
        pdf_id = save_pdf_to_gridfs(pdf_bytes, orig_filename)
    except Exception as e:
        print(f"❌ Error saving PDF to GridFS: {e}")
        return
    print(f"   ✔️ PDF saved, id = {pdf_id}")

    # Create a temporary folder for conversion
    tmp_dir = tempfile.mkdtemp(prefix="pdf_to_md_")
    tmp_pdf_path = os.path.join(tmp_dir, orig_filename)
    with open(tmp_pdf_path, "wb") as f_tmp:
        f_tmp.write(pdf_bytes)
    print(f"   📂 Temporary PDF for conversion: {tmp_pdf_path}")

    # 3) Convert PDF → raw Markdown
    tmp_md_path = os.path.join(tmp_dir, f"{base_name}.md")
    print("2) Converting PDF → Markdown...")
    try:
        convert_pdf_to_md(tmp_pdf_path, tmp_md_path, show_progress=False)
    except Exception as e:
        print(f"❌ Error converting PDF to MD: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Markdown saved locally: {tmp_md_path}")

    # 4) Save raw Markdown into GridFS
    with open(tmp_md_path, "r", encoding="utf-8") as f_md:
        md_text = f_md.read()
    md_filename = f"{base_name}.md"
    print("3) Saving Markdown to GridFS...")
    try:
        md_id = save_text_to_gridfs(md_text, md_filename)
    except Exception as e:
        print(f"❌ Error saving Markdown to GridFS: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Markdown saved, id = {md_id}")

    # 5) Clean Markdown by removing tables
    tmp_md_clean_path = os.path.join(tmp_dir, f"{base_name}_clean.md")
    print("4) Removing tables from Markdown...")
    try:
        remove_markdown_tables(tmp_md_path, tmp_md_clean_path)
    except Exception as e:
        print(f"❌ Error removing tables from Markdown: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Clean Markdown saved locally: {tmp_md_clean_path}")

    # 6) Save clean Markdown into GridFS
    with open(tmp_md_clean_path, "r", encoding="utf-8") as f_clean:
        clean_md_text = f_clean.read()
    clean_md_filename = f"{base_name}_clean.md"
    print("5) Saving clean Markdown to GridFS...")
    try:
        clean_md_id = save_text_to_gridfs(clean_md_text, clean_md_filename)
    except Exception as e:
        print(f"❌ Error saving clean Markdown to GridFS: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    print(f"   ✔️ Clean Markdown saved, id = {clean_md_id}")

    # 7) Remove temporary folder
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"6) Temporary files removed: {tmp_dir}")

    # 8) Print results
    print("\n=== RESULTS ===")
    print(f"• PDF ObjectId:           {pdf_id}")
    print(f"• Markdown ObjectId:      {md_id}")
    print(f"• Clean Markdown ObjectId: {clean_md_id}")
    print("========================")


# ───── Main Menu for my_utils.py ───── #

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
