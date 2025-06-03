import os
import shutil
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data_processing.removetables import remove_markdown_tables

# Пути к файлам
input_md = r"C:\Work\diplom2\rag_on_papers\data\user_manual\user_manual.md"
cleaned_md = r"C:\Work\diplom2\rag_on_papers\data\user_manual\user_manual_clean.md"
embeddings_folder = r"C:\Work\diplom2\rag_on_papers\data\user_manual\embeddings"

def process_and_store_chunks(md_path: str, persist_directory: str) -> None:
    """Обработка Markdown файла: очистка, разбиение на чанки и сохранение эмбеддингов в ChromaDB."""
    try:
        # 1) Очистка Markdown от таблиц
        remove_markdown_tables(md_path, cleaned_md)
        print(f"[INFO] Cleaned Markdown saved to: {cleaned_md}")

        # 2) Чтение очищенного текста
        with open(cleaned_md, "r", encoding="utf-8") as f:
            content = f.read()

        # 3) Настройка эмбединг-модели
        embed_model = OllamaEmbeddings(model="nomic-embed-text:latest")

        # 4) Инициализация RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # размер чанка в символах
            chunk_overlap=200,    # перекрытие между чанками
            length_function=len
        )

        # 5) Разбиение текста на чанки
        split_documents = text_splitter.create_documents([content])

        # 6) Генерация эмбеддингов для каждого чанка
        embeddings = embed_model.embed_documents(
            [doc.page_content for doc in split_documents]
        )

        # 7) Подготовка папки для хранения эмбеддингов
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)

        # 8) Сохранение в ChromaDB
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(name="user_manual_chunks")

        for idx, (doc, emb) in enumerate(zip(split_documents, embeddings)):
            print(f"[INFO] Processing chunk {idx}: {doc.page_content[:60].replace(os.linesep, ' ')}...")
            collection.add(
                ids=[f"{os.path.basename(md_path)}_chunk_{idx}"],
                embeddings=[emb],
                documents=[doc.page_content],
                metadatas=[{
                    "chunk_index": idx,
                    "source": os.path.basename(md_path)
                }]
            )
            print(f"[INFO] Chunk {idx} saved.")

        print(f"[SUCCESS] All chunks of {md_path} have been embedded and stored in {persist_directory}")

    except Exception as e:
        print(f"[ERROR] Failed to process {md_path}: {e}")
        exit(1)

if __name__ == "__main__":
    process_and_store_chunks(input_md, embeddings_folder)
