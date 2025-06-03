#!/usr/bin/env python3
# coding: utf-8
"""
Pipeline для багаторазової оцінки з Ollama-LLM
- Один раз генеруємо чанки та створюємо колекції
- Для списку з 10 запитів виконуємо LLM-оцінку та підрахунок метрик
- У кінці виводимо середні значення Precision, Recall, F1 для кожної стратегії
"""
import os
import sys
import nltk
import chromadb
from tqdm import tqdm

# Локальні утиліти
from src.data_processing.pdf_to_md import convert_pdf_to_md
from src.data_processing.removetables import remove_markdown_tables

# LangChain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# NLTK
nltk.download("punkt", quiet=True)

# Шляхи
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "data", "downloaded_papers")
MD_FOLDER  = os.path.join(BASE_DIR, "data", "dl_papers_md")
os.makedirs(MD_FOLDER, exist_ok=True)

# Моделі
embed_model    = OllamaEmbeddings(model="nomic-embed-text:latest")
llm_classifier = OllamaLLM(model="llama3.2:latest", temperature=0)

# TextTiling
from nltk.tokenize.texttiling import TextTilingTokenizer
_tt = TextTilingTokenizer()

# 1) PDF -> MD + чистка таблиць

def preprocess_all_pdfs():
    for fn in os.listdir(PDF_FOLDER):
        if not fn.lower().endswith(".pdf"): continue
        md_path = os.path.join(MD_FOLDER, fn[:-4] + ".md")
        if os.path.exists(md_path): continue
        try:
            convert_pdf_to_md(os.path.join(PDF_FOLDER, fn), md_path)
            print(f"✅ {fn} → {os.path.basename(md_path)}")
        except Exception as e:
            print(f"❌ {fn}: {e}")

# 2) Чанкінг-стратегії

def chunk_texttiling(text: str) -> list[str]:
    text = text.replace('\r\n','\n').replace('\r','\n').strip()
    if '\n\n' not in text:
        text = text.replace('\n','\n\n')
    try:
        segs = _tt.tokenize(text)
    except ValueError:
        segs = [text]
    return [s.strip() for s in segs if len(s.strip())>50]

chunk_recursive = lambda text: [d.page_content.strip() for d in RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([text])]
chunk_simple    = lambda text: [d.page_content.strip() for d in CharacterTextSplitter().create_documents([text]) if d.page_content.strip()]

STRATEGIES = {
    "TextTiling": chunk_texttiling,
    "Recursive":  chunk_recursive,
    "Simple":     chunk_simple,
}

# 3) LLM-класифікатор
_PROMPT = """
Ти бінарний класифікатор релевантності.

Запит:
«{query}»

Фрагмент:
«{chunk}»

Відповідай лише 1 (релевантно) або 0 (ні).
"""

def label_chunks_llm(chunks: list[str], query: str) -> list[int]:
    labels = []
    for ch in tqdm(chunks, desc="LLM labeling", leave=False):
        prompt = _PROMPT.format(query=query, chunk=ch[:500])
        try:
            resp = llm_classifier.invoke(prompt).strip()
        except:
            resp = "0"
        labels.append(1 if resp.startswith("1") else 0)
    return labels

# 4) Побудова колекцій та пошук Top-K

def build_collections(chunks_by_strategy: dict[str, dict[str,str]]) -> dict[str, chromadb.api.models.Collection]:
    client = chromadb.Client()
    colls = {}
    for name, chunks_map in chunks_by_strategy.items():
        if any(c.name == name for c in client.list_collections()):
            client.delete_collection(name)
        coll = client.create_collection(name=name)
        ids       = list(chunks_map.keys())
        docs      = list(chunks_map.values())
        embs      = embed_model.embed_documents(docs)
        coll.add(ids=ids, documents=docs, embeddings=embs)
        colls[name] = coll
    return colls

# 5) Оцінка одного запиту

def evaluate_query(query: str,
                   chunks_by_strategy: dict[str, dict[str,str]],
                   colls: dict[str, chromadb.api.models.Collection],
                   k: int = 20) -> dict[str, dict[str, float]]:
    metrics = {}
    print(f"\n=== Запит: {query} ===")
    for name, chunks_map in chunks_by_strategy.items():
        total = len(chunks_map)
        gt = label_chunks_llm(list(chunks_map.values()), query)
        ids = list(chunks_map.keys())
        gt_map = dict(zip(ids, gt))
        retrieved = set(colls[name].query(
            query_embeddings=[embed_model.embed_query(query)],
            n_results=k, include=[])["ids"][0])
        TP=TN=FP=FN=0
        for cid, lbl in gt_map.items():
            in_top = cid in retrieved
            if lbl==1 and in_top: TP+=1
            elif lbl==0 and not in_top: TN+=1
            elif lbl==0 and in_top: FP+=1
            elif lbl==1 and not in_top: FN+=1
        prec = TP/(TP+FP) if TP+FP else 0
        rec  = TP/(TP+FN) if TP+FN else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        print(f"{name:12s}: TP={TP:3d} FP={FP:3d} FN={FN:3d} TN={TN:3d}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        metrics[name] = {"precision": prec, "recall": rec, "f1": f1}
    return metrics

# 6) MAIN

def main():
    preprocess_all_pdfs()
    md_texts = [(md, open(os.path.join(MD_FOLDER, md), encoding="utf-8").read())
                for md in os.listdir(MD_FOLDER) if md.endswith(".md")]
    if not md_texts:
        print("❌ Немає Markdown-файлів."); sys.exit(1)
    chunks_by_strategy = {}
    for name, fn in STRATEGIES.items():
        mp = {}
        for md_fname, text in md_texts:
            for i, ch in enumerate(fn(text)):
                mp[f"{md_fname}#c{i}"] = ch
        chunks_by_strategy[name] = mp
    colls = build_collections(chunks_by_strategy)

    print("Введіть 10 запитів (по одному на рядок):")
    queries = [input(f"Запит {i+1}: ").strip() for i in range(10)]

    all_metrics = {name: {"precision": [], "recall": [], "f1": []}
                   for name in STRATEGIES}
    for q in queries:
        if not q:
            print("Порожній запит, пропуск.")
            continue
        m = evaluate_query(q, chunks_by_strategy, colls, k=20)
        for name, vals in m.items():
            all_metrics[name]["precision"].append(vals["precision"])
            all_metrics[name]["recall"].append(vals["recall"])
            all_metrics[name]["f1"].append(vals["f1"])

    # 7) Підсумкова статистика
    print("\n=== Середні значення по всіх запитах ===")
    for name, vals in all_metrics.items():
        avg_p = sum(vals["precision"]) / len(vals["precision"]) if vals["precision"] else 0
        avg_r = sum(vals["recall"])    / len(vals["recall"])    if vals["recall"]    else 0
        avg_f = sum(vals["f1"])        / len(vals["f1"])        if vals["f1"]        else 0
        print(f"{name:12s}: avg Precision={avg_p:.3f}, avg Recall={avg_r:.3f}, avg F1={avg_f:.3f}")

if __name__ == "__main__":
    main()