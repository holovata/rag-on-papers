import kaggle
import os

# Параметри
DATASET = 'Cornell-University/arxiv'  # Посилання на датасет
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))  # Коренева директорія проекту
DOWNLOAD_PATH = os.path.join(BASE_DIR, 'data/raw_metadata')  # Відносний шлях для збереження даних

def download_dataset():
    """
    Завантажує та розпаковує датасет у вказану папку.
    """
    print(f"Завантаження датасету в {DOWNLOAD_PATH}...")
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)
    kaggle.api.dataset_download_files(DATASET, path=DOWNLOAD_PATH, unzip=True)
    print(f"Датасет успішно завантажено та розпаковано в {DOWNLOAD_PATH}")

if __name__ == "__main__":
    download_dataset()