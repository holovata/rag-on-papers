import os
from src.data_processing.pdf_to_md import pymupdf4llm
from src.data_processing.removetables import remove_markdown_tables
from src.data_processing.papers_vectorization import process_and_store_chunks

def convert_pdf_to_md(pdf_path: str, md_output: str) -> str:
    """
    Конвертирует PDF в Markdown с помощью pymupdf4llm.to_markdown.
    Возвращает путь к полученному MD-файлу.
    """
    try:
        md_text = pymupdf4llm.to_markdown(
            doc=pdf_path,
            page_chunks=True,
            write_images=False,
            show_progress=False
        )
        md_content = "\n\n".join(
            chunk.get("text", "") for chunk in md_text if isinstance(chunk, dict)
        )
        with open(md_output, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[convert_pdf_to_md] Markdown сохранён: {md_output}")
        return md_output
    except Exception as e:
        print(f"[convert_pdf_to_md] Ошибка: {e}")
        return ""

def remove_markdown_tables_wrapper(input_file: str, output_file: str) -> str:
    """
    Удаляет таблицы из Markdown-файла и сохраняет результат в output_file.
    """
    try:
        remove_markdown_tables(input_file, output_file)
        print(f"[remove_markdown_tables_wrapper] Очищенный markdown сохранён: {output_file}")
        return output_file
    except Exception as e:
        print(f"[remove_markdown_tables_wrapper] Ошибка: {e}")
        return ""

def perform_text_tiling(input_file: str, output_file: str, segment_size: int = 5) -> str:
    """
    Разбивает текст на сегменты (text tiling) по умолчанию по 5 предложений в сегмент.
    Результат сохраняется в output_file.
    Возвращает путь к исходному (очищенному) Markdown.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = sent_tokenize(text)
        segments = [
            ' '.join(sentences[i:i+segment_size])
            for i in range(0, len(sentences), segment_size)
        ]
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, 1):
                f.write(f"=== Segment {idx} ===\n{seg}\n\n")
        print(f"[perform_text_tiling] Сегменты сохранены: {output_file}")
        return input_file
    except Exception as e:
        print(f"[perform_text_tiling] Ошибка: {e}")
        return ""

def full_file_processing_pipeline(input_folder: str, embeddings_folder: str) -> None:
    """
    Обрабатывает все PDF-файлы, лежащие в input_folder.
    Для каждого PDF:
      1) Конвертация в Markdown (<base_name>.md)
      2) Очистка от таблиц (<base_name>_clean.md)
      3) Сегментация текста (<base_name>_segments.txt)
      4) Векторизация – для каждого файла создаётся подпапка в embeddings_folder.
    Все эмбеддинги сохраняются в общей директории embeddings_folder.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]

            md_output = os.path.join(input_folder, f"{base_name}.md")
            cleaned_md = os.path.join(input_folder, f"{base_name}_clean.md")
            segments_path = os.path.join(input_folder, f"{base_name}_segments.txt")

            print(f"[INFO] Обработка файла: {filename}")

            # 1) Конвертация PDF -> Markdown
            md_result = convert_pdf_to_md(pdf_path, md_output)
            if not md_result:
                print(f"[ERROR] Конвертация файла {filename} не удалась.")
                continue

            # 2) Очистка Markdown от таблиц
            clean_result = remove_markdown_tables_wrapper(md_output, cleaned_md)
            if not clean_result:
                print(f"[ERROR] Очистка файла {filename} не удалась.")
                continue

            # 3) Сегментация текста (text tiling)
            segments_result = perform_text_tiling(clean_result, segments_path)
            if not segments_result:
                print(f"[ERROR] Сегментация файла {filename} не удалась.")
                continue

            # 4) Векторизация: создаём подпапку для данного файла внутри общей папки эмбеддингов
            file_embedding_folder = os.path.join(embeddings_folder, base_name)
            os.makedirs(file_embedding_folder, exist_ok=True)
            print(f"[INFO] Сохранение эмбеддингов для {filename} в: {file_embedding_folder}")
            process_and_store_chunks(cleaned_md, persist_directory=file_embedding_folder)
            print(f"[INFO] Файл {filename} обработан.\n")

    print("[INFO] Обработка всех PDF-файлов завершена.")


def full_file_processing_pipeline_single(input_file_path: str, target_folder: str, embeddings_folder: str) -> None:
    """
    Обрабатывает один PDF-файл.
    1) Конвертация PDF -> Markdown (<base_name>.md)
    2) Очистка от таблиц (<base_name>_clean.md)
    3) Сегментация текста (<base_name>_segments.txt)
    4) Векторизация: эмбеддинги сохраняются в embeddings_folder.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    md_output = os.path.join(target_folder, f"{base_name}.md")
    cleaned_md = os.path.join(target_folder, f"{base_name}_clean.md")
    segments_path = os.path.join(target_folder, f"{base_name}_segments.txt")

    print(f"[INFO] Processing file: {input_file_path}")

    # 1) PDF -> Markdown
    md_result = convert_pdf_to_md(input_file_path, md_output)
    if not md_result:
        print(f"[ERROR] Конвертация файла {input_file_path} не удалась.")
        return

    # 2) Очистка Markdown от таблиц
    clean_result = remove_markdown_tables_wrapper(md_output, cleaned_md)
    if not clean_result:
        print(f"[ERROR] Очистка файла {input_file_path} не удалась.")
        return

    # 3) Сегментация текста (text tiling)
    segments_result = perform_text_tiling(clean_result, segments_path)
    if not segments_result:
        print(f"[ERROR] Сегментация файла {input_file_path} не удалась.")
        return

    # 4) Векторизация – сохраняем эмбеддинги в указанную папку
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder, exist_ok=True)
    print(f"[INFO] Saving embeddings for {base_name} in: {embeddings_folder}")
    process_and_store_chunks(cleaned_md, persist_directory=embeddings_folder)
    print(f"[INFO] File {input_file_path} processed successfully.")

if __name__ == "__main__":
    # Пример использования:
    user_input_folder = input("Введите путь к папке с PDF-файлами: ").strip()
    embeddings_folder = input("Введите путь для сохранения эмбеддингов: ").strip()

    full_file_processing_pipeline(input_folder=user_input_folder, embeddings_folder=embeddings_folder)
