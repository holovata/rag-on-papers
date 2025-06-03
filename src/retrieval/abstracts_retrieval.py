import chromadb
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.retrieval.download_papers import download_arxiv_papers

# Шляхи та налаштування
CHROMA_DB_PATH = "C:\\Work\\diplom2\\rag_on_papers\\data\\vectorized_metadata"

# Ініціалізація ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="arxiv_data")

def generate_query_embedding(query):
    """Генерація ембеддинга для запиту користувача."""
    try:
        response = ollama.embeddings(model="nomic-embed-text:latest", prompt=query)
        return response["embedding"]
    except Exception as e:
        print(f"Помилка під час генерації ембеддинга для запиту: {e}")
        return None


def get_top_relevant_articles(query, top_n=3, min_similarity=0.7):
    """
    Получает топ-N релевантных статей из коллекции по заданному запросу,
    фильтруя по порогу косинусной схожести.
    Возвращает список словарей с информацией о статье, включая:
      - id
      - title (название)
      - authors (авторы)
      - abstract (аннотация)
      - similarity (значение схожести)
      - metadata (полные метаданные)
    """
    query_embedding = generate_query_embedding(query)
    print(f"[DEBUG] Query embedding sample: {query_embedding[:3]}")
    if query_embedding is None:
        print("Не вдалося згенерувати ембеддинг для запиту.")
        return []

    try:
        # Получаем данные из коллекции с необходимыми ключами
        results = collection.get(include=["embeddings", "documents", "metadatas"])
        # Проверка доступности коллекции
        print(f"Collection contains {len(results['ids'])} items")
        print(f"First item metadata: {results['metadatas'][0]}")

        # Проверка наличия ключей
        if not all(key in results for key in ["ids", "embeddings", "documents", "metadatas"]):
            print("Колекція містить неповні дані")
            return []

        if len(results["ids"]) == 0:
            print("База даних порожня.")
            return []

        # Проверка согласованности размеров данных
        data_len = len(results["ids"])
        if (len(results["embeddings"]) != data_len or
                len(results["documents"]) != data_len or
                len(results["metadatas"]) != data_len):
            print("Невідповідність розмірів даних у колекції")
            return []

        # Вычисляем косинусную схожесть
        embeddings = np.array(results["embeddings"])
        query_vector = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vector, embeddings).flatten()

        # Фильтрация индексов по порогу схожести
        valid_indices = np.where(similarities >= min_similarity)[0]
        if len(valid_indices) == 0:
            print(f"Немає статей зі схожістю ≥ {min_similarity}")
            return []

        # Сортировка по убыванию схожести и выбор top_n статей
        top_indices = valid_indices[similarities[valid_indices].argsort()[-top_n:][::-1]]

        top_articles = []
        for idx in top_indices:
            try:
                metadata = results["metadatas"][idx]
                # Извлечение названия и авторов из метаданных
                title = metadata.get("title", "N/A")
                authors = metadata.get("authors", "N/A")
                article_data = {
                    "id": metadata.get("id", "unknown_id"),
                    "title": title,
                    "authors": authors,
                    "abstract": results["documents"][idx][:500] + "..." if results["documents"][idx] else "Немає анотації",
                    "similarity": float(similarities[idx]),
                    "metadata": metadata
                }
                top_articles.append(article_data)
            except IndexError as e:
                print(f"Помилка індексу {idx}: {str(e)}")
                continue

        # Отладочный вывод
        print(f"[DEBUG] Перший елемент: {top_articles[0] if top_articles else 'пусто'}")
        print(f"[DEBUG] Max similarity: {np.max(similarities):.2f}")
        return top_articles

    except Exception as e:
        print(f"Критична помилка: {str(e)}")
        return []




if __name__ == "__main__":
    user_query = input("Введіть ваш запит: ")
    top_articles = get_top_relevant_articles(user_query, top_n=5)

    if top_articles:
        print("\nТоп-5 релевантних статей:")
        for i, article in enumerate(top_articles, start=1):
            print(f"\n#{i}")
            print(f"ID: {article['id']}")
            print(f"Релевантність: {article['similarity']:.4f}")
            print(f"Анотація: {article['abstract'][:500]}...")

        download_arxiv_papers(top_articles)
    else:
        print("Не вдалося знайти релевантні статті.")
