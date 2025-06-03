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
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results2")
# Создаем подпапки 1..5 внутри RESULTS_DIR
for i in range(1, 6):
    os.makedirs(os.path.join(RESULTS_DIR, str(i)), exist_ok=True)

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
    if any(c.name == name for c in client.list_collections()):
        client.delete_collection(name)
    coll = client.create_collection(name=name)
    ids   = list(chunks.keys())
    docs  = list(chunks.values())
    embs  = embed_model.embed_documents(docs)
    coll.add(ids=ids, documents=docs, embeddings=embs)
    return coll

# Пять разных промптов для параллельного использования:
prompt_templates = [
    # 1: акцент на фактологическую точность
    """
You are a precise assistant.
Answer the question strictly based on the provided context. Do not add any external facts or assumptions.
Cite the source after every factual statement using the format (source: filename.md, chunk N).

Query:
{query}

Context:
{context}
""",

    # 2: акцент на фактологичность с доп.
    """
You are a knowledgeable assistant.
Provide a factually rich and accurate answer using the context below. If you are confident, you may include relevant background knowledge not found in the context.
Cite sources from the context when possible using (source: filename.md, chunk N).

Query:
{query}

Context:
{context}
""",

    # 3: сбалансированный подход
    """
You are an expert assistant.
Give a concise and informative answer, using both the context and your internal knowledge if helpful. Prefer clarity and completeness over citing every detail. Use citations only for key claims.

Query:
{query}

Context:
{context}
""",

    # 4: легкость восприятия
    """
You are a clear and helpful assistant.
Write an easy-to-read answer that summarizes the context naturally. Use your own knowledge to clarify if needed, and minimize direct citations unless they are critical.

Query:
{query}

Context:
{context}
""",

    # 5: расширенный разбор
    """
You are a friendly educator.
Provide a highly readable and intuitive explanation, suitable for non-experts. It’s okay to simplify complex concepts and use your own words. Use external facts when necessary, but avoid overloading with citations.

Query:
{query}

Context:
{context}
"""
]

def generate_answer(query: str,
                    collection: chromadb.api.models.Collection,
                    chunks_map: dict[str,str],
                    prompt_idx: int,
                    top_k: int = 5) -> tuple[str, str]:
    # 1) Получаем embedding и делаем запрос
    q_emb   = embed_model.embed_query(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    top_ids = results["ids"][0]

    # вспомогательная функция для красивой подписи источника
    def fmt_source_id(sid: str) -> str:
        fn, _, idx = sid.partition("#c")
        return f"{fn} (chunk {idx})"

    # 2) Собираем контекст
    context = "\n\n".join(
        f"Source: {fmt_source_id(sid)}\nContent: {chunks_map[sid]}"
        for sid in top_ids
    )

    # 3) Выбираем шаблон по индексу
    template = prompt_templates[prompt_idx]
    prompt = template.format(query=query, context=context)

    # 4) Получаем ответ
    answer = llm.invoke(prompt)
    return prompt, answer


def main():
    md_texts = load_markdowns(MD_FOLDER)
    chunks_map: dict[str,str] = {}
    for fn, text in md_texts.items():
        for i, chunk in enumerate(chunk_recursive(text)):
            chunks_map[f"{fn}#c{i}"] = chunk

    coll = build_collection("recursive_chunks", chunks_map)

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

    # Обработка: каждый запрос через все 5 промптов
    for query in queries:
        for idx, template in enumerate(prompt_templates, start=1):
            prompt, answer = generate_answer(query, coll, chunks_map, idx-1, top_k=5)

            print(f"\n=== ANSWER (Prompt {idx}) ===\n")
            print(answer)
            print("\n" + ("—" * 40) + "\n")

            # Сохраняем в подпапку idx
            subdir = os.path.join(RESULTS_DIR, str(idx))
            filename = f"{slugify(query)}.txt"
            path = os.path.join(subdir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write("=== PROMPT ===\n")
                f.write(prompt.strip() + "\n\n")
                f.write("=== ANSWER ===\n")
                f.write(answer.strip() + "\n")
            print(f"✅ Saved to {path}\n")

if __name__ == "__main__":
    main()