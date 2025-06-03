# src/data_processing/pdf_to_md.py
import pathlib
import pymupdf4llm


def convert_pdf_to_md(
    pdf_path: str | pathlib.Path,
    md_path: str | pathlib.Path,
    *,
    page_chunks: bool = True,
    write_images: bool = False,
    show_progress: bool = False,
) -> None:
    """
    Конвертує PDF-файл у Markdown за допомогою pymupdf4llm.

    Parameters
    ----------
    pdf_path : str | Path
        Шлях до PDF-файла.
    md_path : str | Path
        Куди зберегти Markdown.
    page_chunks : bool, default=True
        Розбивати PDF по сторінках (True) чи обробляти суцільним текстом.
    write_images : bool, default=False
        Чи експортувати зображення.
    show_progress : bool, default=False
        Показувати індикатор прогресу.
    """
    pdf_path = pathlib.Path(pdf_path)
    md_path = pathlib.Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"→ Обробляю: {pdf_path}")

    try:
        md_chunks = pymupdf4llm.to_markdown(
            doc=str(pdf_path),
            page_chunks=page_chunks,
            write_images=write_images,
            show_progress=show_progress,
        )

        markdown_content = [
            chunk["text"] for chunk in md_chunks if isinstance(chunk, dict) and "text" in chunk
        ]
        md_path.write_text("\n\n".join(markdown_content), encoding="utf-8")

        print(f"✅ Markdown збережено: {md_path}")
    except Exception as exc:
        print(f"❌ Помилка обробки {pdf_path}: {exc}")


# ─────────────────────────
# Приклад пакетної обробки
# ─────────────────────────
if __name__ == "__main__":
    pdf_folder = pathlib.Path("../../data/downloaded_papers")
    md_folder = pathlib.Path("../../data/dl_papers_md")

    for pdf_file in pdf_folder.glob("*.pdf"):
        target_md = md_folder / f"{pdf_file.stem}.md"
        convert_pdf_to_md(pdf_file, target_md, show_progress=True)
