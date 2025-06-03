import os
import chromadb
import json
import ollama

# Шляхи та налаштування
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))  # Коренева директорія проекту
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw_metadata", "arxiv-metadata-oai-snapshot.json")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "vectorized_metadata")
PROCESSED_IDS_FILE = "C:\\Work\\diplom2\\rag_on_papers\\data\\vectorized_metadata\\processed_ids.txt"

# Переконуємося, що директорія для ChromaDB існує
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

print(f"Ембедінги будуть збережені в: {CHROMA_DB_PATH}")

# Ініціалізація ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="arxiv_data")


# Обробка JSON файлу частинами
def process_large_json(file_path, batch_size=1000):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Пропускаємо початкові та кінцеві дужки, якщо це JSON масив
            f.seek(0)
            f.readline()  # Зчитуємо перший рядок (початкова дужка "[")

            batch = []
            for line in f:
                # Прибираємо кому та закриваючу дужку, якщо це останній рядок
                line = line.strip().rstrip(",").rstrip("]")
                if not line:
                    continue

                try:
                    entry = json.loads(line)  # Перетворюємо рядок у JSON
                    batch.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Помилка декодування рядка: {e}")
                    continue

                if len(batch) >= batch_size:
                    yield batch  # Надсилаємо пакет записів
                    batch = []

            # Надсилаємо залишок
            if batch:
                yield batch
    except FileNotFoundError:
        print(f"Файл {file_path} не знайдено.")
    except Exception as e:
        print(f"Помилка при обробці файлу: {e}")


# Підрахунок кількості пакетів
def count_batches(file_path, batch_size=1000):
    total_lines = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().endswith("}") or line.strip().endswith("},"):
                    total_lines += 1
        return (total_lines // batch_size) + (1 if total_lines % batch_size else 0)
    except Exception as e:
        print(f"Помилка підрахунку рядків у файлі: {e}")
        return 0


# Завантаження оброблених ID
def load_processed_ids(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f)
    return set()


# Збереження оброблених ID
def save_processed_ids(file_path, ids):
    print(f"Функція save_processed_ids викликана з {len(ids)} ID.")
    if not ids:
        print("Немає нових ID для збереження.")
        return
    with open(file_path, "a", encoding="utf-8") as f:
        for entry_id in ids:
            f.write(f"{entry_id}\n")
    print(f"Збережено {len(ids)} нових ID у файл {file_path}.")


# Додавання даних до ChromaDB з відладкою
def add_to_chroma(batch, processed_ids):
    newly_processed_ids = set()
    for entry in batch:
        paper_id = entry.get("id")
        abstract = entry.get("abstract")
        if not paper_id or not abstract:
            print(f"Пропущено запись с некорректными данными: {entry}")
            continue  # Пропускаем записи без ID или аннотации

        if paper_id in processed_ids:
            print(f"Запись с ID {paper_id} уже добавлена, пропускаем.")
            continue

        try:
            response = ollama.embeddings(model="nomic-embed-text:latest", prompt=abstract)
            embedding = response["embedding"]

            # Создаем словарь метаданных и заменяем None на пустую строку
            metadata = {
                "id": paper_id,
                "title": entry.get("title") or "",
                "authors": entry.get("authors") or "",
                "abstract": abstract,
                "categories": entry.get("categories") or "",
                "comments": entry.get("comments") or "",
                "journal_ref": entry.get("journal-ref") or "",
                "doi": entry.get("doi") or ""
            }

            collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                documents=[abstract],
                metadatas=[metadata]
            )
            print(f"Успешно добавлена запись с ID: {paper_id}")
            newly_processed_ids.add(paper_id)
        except Exception as e:
            print(f"Ошибка при обработке записи {paper_id}: {e}")

    print(f"Новые обработанные ID в этом пакете: {newly_processed_ids}")
    return newly_processed_ids



LAST_BATCH_FILE = os.path.join(CHROMA_DB_PATH, "last_processed_batch.txt")


# Завантаження номера останнього обробленого пакету
def load_last_processed_batch(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0  # Якщо файл не існує, починаємо з першого пакету


# Збереження номера останнього обробленого пакету
def save_last_processed_batch(file_path, batch_number):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(batch_number))
    print(f"Збережено номер останнього обробленого пакету: {batch_number}")


# Функція перевірки вмісту ChromaDB
def check_chroma_content(test_ids):
    print("Перевірка вмісту ChromaDB для тестових ID...")
    for test_id in test_ids:
        try:
            result = collection.get(ids=[test_id])
            if result and len(result['metadatas']) > 0:
                metadata = result['metadatas'][0]
                document = result['documents'][0]
                print(f"ID: {test_id} знайдено в базі.")
                print(f"Метадані: {metadata}")
                print(f"Документ: {document[:100]}...")  # Виведення перших 100 символів
            else:
                print(f"ID: {test_id} не знайдено в базі.")
        except Exception as e:
            print(f"Помилка при перевірці ID {test_id}: {e}")


# Основний процес
if __name__ == "__main__":
    # Завантажуємо вже оброблені ID
    processed_ids = load_processed_ids(PROCESSED_IDS_FILE)
    print(f"Завантажено {len(processed_ids)} вже оброблених ID.")

    # Завантажуємо номер останнього обробленого пакету
    last_batch = load_last_processed_batch(LAST_BATCH_FILE)
    print(f"Починаємо з пакету {last_batch + 1}...")

    # Змінна для підрахунку загальної кількості доданих ID
    total_new_ids = 0

    # Обробка файлу пакетами
    for i, batch in enumerate(process_large_json(DATASET_PATH, batch_size=1000), start=1):
        if i <= last_batch:
            print(f"Пакет {i} вже оброблено, пропускаємо.")
            continue

        print(f"Обробка пакету {i}...")
        newly_processed_ids = add_to_chroma(batch, processed_ids)

        # Перевірка перед збереженням
        if newly_processed_ids:
            print(f"Перед збереженням: {newly_processed_ids}")

        # Зберігаємо нові ID у файл
        save_processed_ids(PROCESSED_IDS_FILE, newly_processed_ids)

        # Оновлюємо список оброблених ID
        processed_ids.update(newly_processed_ids)
        total_new_ids += len(newly_processed_ids)

        print(f"Оброблено {len(newly_processed_ids)} нових записів у пакеті {i}.")

        # Зберігаємо номер останнього обробленого пакету
        save_last_processed_batch(LAST_BATCH_FILE, i)

        # Обмеження на обробку (наприклад, обробити лише 20 пакетів за раз)
        if i >= last_batch + 20:  # Обробляємо 20 пакетів за запуск
            print("Обробка тимчасово зупинена. Ви можете продовжити пізніше.")
            break

    print(f"Всього додано {total_new_ids} нових записів.")

    # Перевіряємо вміст ChromaDB для тестових ID
    #test_ids = ["0704.0001", "0704.1000", "0704.2000"]  # Приклади тестових ID
    #check_chroma_content(test_ids)