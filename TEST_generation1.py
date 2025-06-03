#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import re
import nltk
from tqdm import tqdm
import chromadb

# LangChain
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NLTK
nltk.download("punkt", quiet=True)

# Папки
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
MD_FOLDER   = os.path.join(BASE_DIR, "data", "dl_papers_md")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Модели
embed_model = OllamaEmbeddings(model="nomic-embed-text:latest")
llm         = OllamaLLM(model="llama3.2:latest", temperature=0)

# Чанкир: рекурсивный, chunk_size=1000, overlap=200
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def slugify(text: str, max_len: int = 50) -> str:
    s = text.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')[:max_len] or "query"

def load_markdowns(md_folder: str) -> dict[str, str]:
    md_texts = {}
    for fn in os.listdir(md_folder):
        if fn.lower().endswith(".md"):
            path = os.path.join(md_folder, fn)
            md_texts[fn] = open(path, encoding="utf-8").read()
    if not md_texts:
        sys.exit("❌ Нет md-файлов для обработки")
    return md_texts

def chunk_recursive(text: str) -> list[str]:
    docs = splitter.create_documents([text])
    return [d.page_content.strip() for d in docs if d.page_content.strip()]

def build_collection(name: str, chunks: dict[str,str]) -> chromadb.api.models.Collection:
    client = chromadb.Client()
    # если коллекция с таким именем есть — удаляем
    if any(c.name == name for c in client.list_collections()):
        client.delete_collection(name)
    coll = client.create_collection(name=name)
    ids   = list(chunks.keys())
    docs  = list(chunks.values())
    embs  = embed_model.embed_documents(docs)
    coll.add(ids=ids, documents=docs, embeddings=embs)
    return coll

def generate_answer(query: str,
                    collection: chromadb.api.models.Collection,
                    chunks_map: dict[str,str],
                    top_k: int = 5) -> tuple[str, str]:
    """
    Возвращает (prompt, answer). Источники будут в формате:
      filename.md (chunk N)
    """

    # 1) Получаем embedding и делаем запрос в Chroma
    q_emb   = embed_model.embed_query(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    top_ids = results["ids"][0]  # список в порядке релевантности

    # вспомогательная функция для красивой подписи источника
    def fmt_source_id(sid: str) -> str:
        # разбиваем "0704.3843_clean.md#c9" → ("0704.3843_clean.md", "9")
        fn, _, idx = sid.partition("#c")
        return f"{fn} (chunk {idx})"

    # 2) Собираем контекст
    context = "\n\n".join(
        f"Source: {fmt_source_id(sid)}\nContent: {chunks_map[sid]}"
        for sid in top_ids
    )

    # 3) Новый, более «человечный» английский промпт
    prompt = f"""You are a knowledgeable assistant.  
Please provide a coherent, well-structured answer to the query below, weaving together the information from the context.  
For every factual claim you make, cite its source in parentheses using the format “(source: filename.md, chunk N)”.

Query:
{query}

Context:
{context}
"""

    # 4) Получаем ответ
    answer = llm.invoke(prompt)
    return prompt, answer


def main():
    # 1) Загружаем все md и чанкаем
    md_texts = load_markdowns(MD_FOLDER)

    # Одностратегийно — только Recursive
    chunks_map: dict[str,str] = {}
    for fn, text in md_texts.items():
        for i, chunk in enumerate(chunk_recursive(text)):
            chunks_map[f"{fn}#c{i}"] = chunk

    # 2) Строим хранилище embedding-ов
    coll = build_collection("recursive_chunks", chunks_map)

    # 3) Сначала вводим несколько запросов
    print("Enter your queries, one per line. Submit an empty line to finish:")
    queries: list[str] = []
    while True:
        q = input("Query: ").strip()
        if not q:
            break
        queries.append(q)
    if not queries:
        print("No queries entered. Exiting.")
        sys.exit(0)

    # 4) Обрабатываем их по порядку, сохраняем промпт и ответ в файлы
    for idx, query in enumerate(queries, start=1):
        prompt, answer = generate_answer(query, coll, chunks_map, top_k=5)

        # Вывод на экран
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n" + ("—" * 40) + "\n")

        # Сохраняем в человекочитаемом виде
        filename = f"{idx:02d}_{slugify(query)}.txt"
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt.strip() + "\n\n")
            f.write("=== ANSWER ===\n")
            f.write(answer.strip() + "\n")
        print(f"✅ Saved prompt and answer to {path}\n")

if __name__ == "__main__":
    main()
