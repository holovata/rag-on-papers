import nltk
from nltk.tokenize.texttiling import TextTilingTokenizer

# 1) Загружаем необходимые ресурсы NLTK
nltk.download('punkt')
nltk.download('stopwords')

def read_and_tile(filepath: str) -> list[str]:
    """
    Читает файл, нормалізує переходи рядків і вставляє подвійні \n\n
    для коректного виявлення абзацних розривів, потім повертає
    список сегментів TextTiling.
    """
    # 2) Читаємо весь текст
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 3) Нормалізуємо всі види переносу рядків у '\n'
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    # 4) Якщо в тексті немає подвійних '\n\n', замінюємо кожен одиночний '\n' на '\n\n'
    if '\n\n' not in text:
        text = text.replace('\n', '\n\n')

    # 5) Ініціалізуємо та запускаємо TextTiling
    tokenizer = TextTilingTokenizer()
    tiles = tokenizer.tokenize(text)
    return tiles

if __name__ == '__main__':
    # приклад використання
    tiles = read_and_tile(r'C:\Work\diplom2\rag_on_papers\data\dl_papers_md\0704.0002.md')
    for i, tile in enumerate(tiles, start=1):
        print(f"\n--- Tile {i} ---")
        print(tile[:200].replace('\n', ' '), '…')  # перші 200 символів кожного сегмента
