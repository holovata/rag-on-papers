import os
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.retrieval.abstracts_retrieval import generate_query_embedding

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))  # Коренева директорія проекту
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "vectorized_papers")
MD_FOLDER_PATH = os.path.join(BASE_DIR, "data", "dl_papers_md")  # Папка з Markdown-файлами

# Переконуємося, що директорії існують
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(MD_FOLDER_PATH, exist_ok=True)

# print(f"Ембедінги будуть збережені в: {CHROMA_DB_PATH}")

# Ініціалізація ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="papers_contents")

def get_md_files(folder_path):
    """Отримати список усіх Markdown-файлів у вказаній папці."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".md")]


def get_top_relevant_documents(query, top_n=5, embeddings_folder=None):
    """Извлекает наиболее релевантные документы из базы эмбеддингов.

    Если параметр embeddings_folder задан, клиент будет инициализирован с этим путём.
    """
    query_embedding = generate_query_embedding(query)
    if query_embedding is None:
        print("Не удалось сгенерировать ембеддинг для запроса.")
        return []

    try:
        if embeddings_folder:
            # Инициализируем клиент с указанной директорией для эмбеддингов
            client = chromadb.PersistentClient(path=embeddings_folder)
            collection_local = client.get_or_create_collection(name="papers_contents")
        else:
            # Если embeddings_folder не указан, используем глобальную коллекцию
            global collection
            collection_local = collection

        # Получение всех записей из коллекции
        results = collection_local.get(include=["embeddings", "documents", "metadatas"])

        if len(results["embeddings"]) == 0:
            print("База данных пуста.")
            return []

        # Вычисление косинусной схожести
        embeddings = np.array(results["embeddings"])
        query_vector = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vector, embeddings).flatten()

        # Получение топ-N индексов
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Формирование списка релевантных документов с указанием источника (если он есть)
        top_articles = []
        for idx in top_indices:
            metadata = results["metadatas"][idx]
            document = results["documents"][idx]
            similarity = similarities[idx]
            top_articles.append({
                "chunk_index": metadata.get("chunk_index"),
                "content": document,
                "similarity": similarity,
                "source": metadata.get("source", "N/A")
            })

        return top_articles

    except Exception as e:
        print(f"Ошибка при извлечении релевантных документов: {e}")
        return []



def process_and_store_chunks(md_path: str, persist_directory: str) -> None:
    """Обработка Markdown файла и сохранение в ChromaDB в указанную директорию."""
    try:
        with open(md_path, "r", encoding="utf-8") as file:
            content = file.read()

        embed_model = OllamaEmbeddings(model="nomic-embed-text:latest")

        # Инициализация RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Размер чанка
            chunk_overlap=200,  # Перекрытие между чанками
            length_function=len
        )

        # Разбиение текста на чанки
        split_documents = text_splitter.create_documents([content])

        # Генерация эмбеддингов для всех чанков
        embeddings = embed_model.embed_documents([doc.page_content for doc in split_documents])

        # Инициализация ChromaDB с указанием директории для сохранения
        os.makedirs(persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(name="papers_contents")

        for idx, (doc, embedding) in enumerate(zip(split_documents, embeddings)):
            print(f"Обработка чанка {idx} из файла {os.path.basename(md_path)}: {doc.page_content[:50]}...")

            collection.add(
                ids=[f"{os.path.basename(md_path)}_chunk_{idx}"],
                embeddings=[embedding],
                documents=[doc.page_content],
                metadatas=[{"chunk_index": idx, "source": os.path.basename(md_path)}]
            )

            print(f"Чанк {idx} успешно сохранен в {persist_directory}.")

        print(f"Markdown {md_path} успешно обработан и сохранен в ChromaDB.")

    except Exception as e:
        print(f"Ошибка при обработке Markdown {md_path}: {e}")


def print_all_chunks():
    """Виведення тексту всіх чанків у консоль."""
    try:
        # Отримання всіх записів із колекції
        results = collection.get(include=["documents", "metadatas"])

        # Якщо база порожня
        if len(results["documents"]) == 0:
            print("База даних порожня.")
            return

        # Виведення тексту всіх чанків
        for idx, (document, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"Чанк {metadata.get('chunk_index', idx)} з файлу {metadata.get('source', 'unknown')} :")
            print(document)
            print("-" * 80)

    except Exception as e:
        print(f"Помилка при виведенні всіх чанків: {e}")

if __name__ == "__main__":
    '''    # Отримати список усіх Markdown-файлів
    md_files = get_md_files(MD_FOLDER_PATH)
    if not md_files:
        print(f"У папці {MD_FOLDER_PATH} немає Markdown-файлів.")
    else:
        for md_file in md_files:
            process_and_store_chunks(md_file)

    # Виведення всіх чанків у консоль
    print_all_chunks()

    user_query = input("Введіть ваш запит: ")
    top_articles = get_top_relevant_documents(user_query, top_n=5)

    if top_articles:
        print("\nТоп-5 релевантних документів:")
        for i, article in enumerate(top_articles, start=1):
            print(f"\n#{i}")
            print(f"Індекс чанка: {article['chunk_index']}")
            print(f"Релевантність: {article['similarity']:.4f}")
            print(f"Зміст: {article['content'][:200]}...")
    else:
        print("Не вдалося знайти релевантні статті.")'''
    # === Виведення одного прикладу запису з векторної бази ===
    print("\n=== Приклад одного запису з векторної бази ===")
    try:
        example_result = collection.get(include=["embeddings", "documents", "metadatas"], ids=None)

        if example_result["documents"]:
            doc = example_result["documents"][10]
            metadata = example_result["metadatas"][10]
            embedding = example_result["embeddings"][10]

            print("→ Текст чанка:")
            print(doc[:300] + "..." if len(doc) > 300 else doc)
            print("\n→ Метадані:")
            print(f"- Джерело: {metadata.get('source', 'невідомо')}")
            print(f"- Індекс чанка: {metadata.get('chunk_index', 'N/A')}")
            print("\n→ Векторне представлення (обрізано до 10 значень):")
            print(embedding[:10], "...")

        else:
            print("База даних порожня, немає жодного запису.")

    except Exception as e:
        print(f"Помилка при виведенні прикладу запису: {e}")
