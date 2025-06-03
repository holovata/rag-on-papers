#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiPDFAnalysisPipeline с автоматическим использованием существующих функций в papers_vectorization_MONGO.py:

Шаги:
 1) Очищаем нужные коллекции:
    • GridFS‐бакет 'pipeline_fs' (file_buffer_db)
    • Коллекцию pdf_markdowns в file_buffer_db
    • Коллекцию multi_paper_chunks (arxiv_db)
 2) Ингестим весь указанный PDF-каталог:
    • PDF → raw.md → clean.md (remove tables)
    • Сохраняем PDF и clean.md в GridFS 'pipeline_fs'
    • Записываем метаданные (имя, md_file_id) в pdf_markdowns
 3) Vectorize:
    • Для каждого чистого Markdown из GridFS «pipeline_fs» делаем чанки + эмбеддинги,
      вставляем в arxiv_db.multi_paper_chunks
 4) Создаём/обновляем векторный индекс для multi_paper_chunks
 5) Retrieval: получаем топ-K чанков из multi_paper_chunks
 6) Reasoning: CoT-процесс по шаблону из YAML, используя retrieved chunks
 7) QA_over_pdf: финальный LLM‐запрос с retrieved chunks
 8) (по желанию) Synthesis
"""

import os
import sys
import yaml

from pathlib import Path
from typing import List, Any, Dict

from bson import ObjectId
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from langchain.schema import Document
from langchain_ollama import OllamaLLM

from dotenv import load_dotenv

# Импорт новых функций из papers_vectorization_MONGO.py
from src.data_processing.papers_vectorization_MONGO import (
    mongo_client as pv_mongo_client,
    ingest_pdfs_to_markdowns,
    ingest_multi_chunks,
    build_vector_index,
    clear_all_before_ingest,
)

# Импорт getter-функции из papers_retrieval_MONGO.py
from src.retrieval.papers_retrieval_MONGO import get_relevant_chunks

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Константы для баз данных / коллекций
# ──────────────────────────────────────────────────────────────

# БД, где хранятся chunks и сами документы
ARXIV_DB_NAME          = "arxiv_db"
MULTI_CHUNK_COL        = "multi_paper_chunks"
MULTI_CHNK_INDEX       = "multi_chunk_embedding_index"

# Буферная БД для GridFS
FILE_BUFFER_DB_NAME    = "file_buffer_db"
PIPELINE_FS_BUCKET     = "pipeline_fs"   # имя коллекции-бакета для мульти-PDF
PDF_MARKDOWNS_COL      = "pdf_markdowns" # коллекция с метаданами чистых MD

# LLM‐модель для CoT/QA
LLM_MODEL_NAME         = "llama3.2:latest"

# ──────────────────────────────────────────────────────────────

class MultiPDFAnalysisPipeline:
    def __init__(self, protocol_path: str):
        self.protocol = self._load_protocol(protocol_path)
        self.context: Dict[str, Any] = {}
        # Подключения к монго
        self.client_arxiv = pv_mongo_client()                 # клиент к 'arxiv_db'
        self.db_arxiv     = self.client_arxiv[ARXIV_DB_NAME]
        self.client_buf   = pv_mongo_client()                 # клиент к 'file_buffer_db'
        self.db_buffer    = self.client_buf[FILE_BUFFER_DB_NAME]

        # GridFS-бакеты
        _, self.fs_default  = pv_mongo_client()[FILE_BUFFER_DB_NAME], None  # не используем для мульти-PDF
        _, self.fs_pipeline = pv_mongo_client()[FILE_BUFFER_DB_NAME], None  # позже инициализируем

        # LLM-инстанс
        self.llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.4)

    def _load_protocol(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _resolve_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подставляет значения из self.context для шаблонов вида "{{var}}".
        """
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                var = v[2:-2].strip()
                resolved[k] = self.context.get(var, "")
            else:
                resolved[k] = v
        return resolved


    # ──────────────────────────────────────────────────────────
    # Шаг 1: очищаем все коллекции и GridFS (выбираем, что чистить)
    # ──────────────────────────────────────────────────────────
    def run_ingest_and_index(self, pdf_folder: str, skip_index: bool = False):
        """
        Выполняем полную очистку и затем:
         • ingest_pdfs_to_markdowns → заполняет file_buffer_db.pipeline_fs и pdf_markdowns
         • ingest_multi_chunks       → читает из file_buffer_db.pipeline_fs и собирает multi_paper_chunks
         • build_vector_index         → создаёт/пересоздаёт индекс в multi_paper_chunks
        """

        # 1) Очистка: GridFS-бакета 'pipeline_fs' + коллекции pdf_markdowns + multi_paper_chunks

        # Инициализируем GridFS для pipeline_fs
        _, self.fs_pipeline = pv_mongo_client()[FILE_BUFFER_DB_NAME], None
        # (GridFS actual creation при первом save_to_bucket внутри ingest_pdfs_to_markdowns)

        print("\n[STEP] Очищаем данные перед новым инжестом:")
        #   а) очистка коллекции pdf_markdowns
        from src.data_processing.papers_vectorization_MONGO import clear_collection
        clear_collection(self.db_buffer, PDF_MARKDOWNS_COL)

        #   б) очистка GridFS-бакета "pipeline_fs"
        clear_all_before_ingest(self.db_buffer, bucket=PIPELINE_FS_BUCKET)

        #   в) очистка коллекции multi_paper_chunks
        clear_collection(self.db_arxiv, MULTI_CHUNK_COL)

        # 2) Инжест PDF → GridFS → metadata
        print("\n[STEP] Инжестируем PDF-файлы в GridFS и записываем метаданные:")
        ingest_pdfs_to_markdowns(self.db_buffer, Path(pdf_folder))

        # 3) Векторизация Markdown из GridFS → multi_paper_chunks
        print("\n[STEP] Чанки и векторизация (multi_paper_chunks):")
        ingest_multi_chunks(self.db_arxiv, self.db_buffer)

        # 4) Построение/обновление векторного индекса
        if not skip_index:
            print("\n[STEP] Построение векторного индекса для multi_paper_chunks:")
            build_vector_index(self.db_arxiv, MULTI_CHUNK_COL, MULTI_CHNK_INDEX)

        print("\n✅ Шаги ingest+index completed.\n")
        return {
            "status": "ingested_and_indexed",
            "multi_chunk_count": self.db_arxiv[MULTI_CHUNK_COL].count_documents({})
        }


    # ──────────────────────────────────────────────────────────
    # Шаг 2: Retrieval по векторам (multi_paper_chunks)
    # ──────────────────────────────────────────────────────────
    def run_retrieval(self, query: str, top_n: int = 4, min_similarity: float = 0.0) -> List[Document]:
        """
        Делаем векторный поиск по multi_paper_chunks. Возвращаем список Document (page_content + metadata).
        """
        print(f"\n[STEP] Retrieval: ищем топ {top_n} релевантных чанков в multi_paper_chunks для запроса '{query}' ...")
        docs = get_relevant_chunks(
            query=query,
            top_n=top_n,
            min_similarity=min_similarity,
            chunk_collection=MULTI_CHUNK_COL,
            chunk_index_name=MULTI_CHNK_INDEX
        )
        print(f"[INFO] Найдено {len(docs)} релевантных чанков.")
        return docs


    # ──────────────────────────────────────────────────────────
    # Шаг 3: Reasoning (CoT) — уточнённый Chain-of-Thought
    # ──────────────────────────────────────────────────────────
    def run_reasoning(self, query: str, chunks: List[Document]) -> Dict[str, Any]:
        """
        Делаем multi-iteration CoT с LLM: последовательно задаём уточняющие вопросы,
        каждый раз передавая предыдущие ответы и куски (с метаданными source).
        Возвращаем dict:
          {
            "initial_answer": str,
            "reasoning_steps": [ {step: str, analysis: str, iteration: int}, ... ],
            "final_answer": str
          }
        """
        print(f"\n[STEP] Reasoning (CoT) for query: {query}")
        # Собираем первый контекст: первые 5 чанков (или сколько есть)
        context_chunks = "\n".join(
            f"Chunk {i+1} (Source: {doc.metadata.get('source')}): {doc.page_content[:200].strip()}..."
            for i, doc in enumerate(chunks[:5])
        )

        # Шаг 1: Initial Answer
        initial_prompt = f"""Initial Answer Generation:
Query: {query}

Context:
{context_chunks}

Please provide your initial answer to the query in a concise manner.
Indicate the source (filename) of each key piece of information.
"""
        initial_answer = self.llm.invoke(initial_prompt)
        print(f"\n[LLM] Initial answer:\n{initial_answer}\n")
        reasoning_steps = [ { "step": "Initial Answer", "analysis": initial_answer, "iteration": 0 } ]
        all_answers = [initial_answer]

        # Шаги CoT (с уточняющими вопросами)
        cot_questions = [
            "What are the main ideas from the retrieved chunks? Indicate sources.",
            "Are there any contradictions or differing perspectives among the sources? If so, which chunks?",
            "How can information from different chunks be combined to create a more comprehensive answer? Indicate sources.",
            "Which chunks (sources) seem most reliable and why? Provide brief justification referencing the chunk IDs.",
            "Is any relevant information missing? Which potential chunks or sources could fill that gap?",
            "Summarize the final answer by integrating all relevant information and clearly indicating sources."
        ]

        for i, question in enumerate(cot_questions, start=1):
            prev_text = "\n".join([f"Answer {j+1}: {ans}" for j, ans in enumerate(all_answers)])
            iteration_prompt = f"""Chain-of-Thought Iteration {i}:
Query: {query}

Context Chunks:
{context_chunks}

Previous Answers:
{prev_text}

Refinement Question: {question}

Please analyze the previous answers, refine your response, and clearly indicate the source of each piece of information (filename or chunk ID).
"""
            iteration_answer = self.llm.invoke(iteration_prompt)
            print(f"\n[LLM] Iteration {i} ({question}):\n{iteration_answer}\n")
            reasoning_steps.append({
                "step": question,
                "analysis": iteration_answer,
                "iteration": i
            })
            all_answers.append(iteration_answer)

        final_answer = all_answers[-1]
        print(f"\n[LLM] Final Answer:\n{final_answer}\n")

        return {
            "initial_answer": initial_answer,
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer
        }


    # ──────────────────────────────────────────────────────────
    # Шаг 4: QA_over_pdf — ещё один запрос LLM с retrieved chunks
    # ──────────────────────────────────────────────────────────
    def run_qa_over_pdf(self, query: str, chunks: List[Document]) -> str:
        """
        Формируем единый prompt, куда кладём top‐chunks и задаём прямой вопрос LLM.
        В ответе LLM обязан упомянуть source (filename) для каждого упомянутого факта.
        """
        print(f"\n[STEP] QA_over_pdf: задаём LLM финальный вопрос по top_chunks ...")
        context_text = "\n\n".join(
            f"Source: {doc.metadata.get('source')}  (Chunk ID: {doc.metadata.get('mongo_id')})\n"
            f"{doc.page_content}"
            for doc in chunks
        )

        prompt = f"""
Based on the following chunks extracted from multiple PDF documents, please answer the query:
{query}

Context (chunks with sources):
{context_text}

Answer and clearly indicate the source (filename) of each piece of information in your answer.
"""
        answer = self.llm.invoke(prompt)
        print(f"\n[LLM] QA_over_pdf Answer:\n{answer}\n")
        return answer


    # ──────────────────────────────────────────────────────────
    # Шаг 5: Synthesis (краткое объединение результатов, при необходимости)
    # ──────────────────────────────────────────────────────────
    def run_synthesis(self, reasoning_output: Dict[str, Any], qa_output: str) -> str:
        """
        Можно реализовать сводный шаг, который объединяет reasoning и qa_output.
        Здесь просто конкатенируем, при необходимости можем доработать.
        """
        print("\n[STEP] Synthesis (объединение Reasoning + QA):")
        initial = reasoning_output.get("initial_answer", "")
        final  = reasoning_output.get("final_answer", "")
        combined = (
            f"Initial reasoning answer:\n{initial}\n\n"
            f"Final reasoning answer:\n{final}\n\n"
            f"QA_over_pdf answer:\n{qa_output}"
        )
        print(f"\n[Combined Output]:\n{combined}\n")
        return combined


    # ──────────────────────────────────────────────────────────
    # Основной метод для пошагового запуска с прогресс‐callback
    # ──────────────────────────────────────────────────────────
    def run_pipeline_with_progress(
        self,
        user_query: str,
        pdf_folder: str,
        progress_callback = None
    ):
        """
        Последовательно выполняет все стадии протокола, отдаёт на каждом шаге:
            (stage_name, status, full_context_dict)
        Если передан progress_callback, то он вызывается на in_progress и done.
        """
        self.context.clear()
        self.context["user_query"] = user_query
        self.context["pdf_folder"] = pdf_folder

        # 1) Ingest + Index
        stage = "ingest_and_index"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        ingest_result = self.run_ingest_and_index(pdf_folder)
        self.context.update(ingest_result)
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # 2) Retrieval
        stage = "retrieval"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        top_chunks: List[Document] = self.run_retrieval(user_query, top_n=5)
        self.context["retrieval_output"] = top_chunks
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # 3) Reasoning
        stage = "reasoning"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        reasoning_out = self.run_reasoning(user_query, top_chunks)
        self.context["reasoning_output"] = reasoning_out
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # 4) QA_over_pdf
        stage = "qa_over_pdf"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        qa_out = self.run_qa_over_pdf(user_query, top_chunks)
        self.context["qa_over_pdf_output"] = qa_out
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # 5) Synthesis (опционально)
        stage = "synthesis"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        synth_out = self.run_synthesis(reasoning_out, qa_out)
        self.context["synthesis_output"] = synth_out
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # Финальный вывод
        yield "pipeline_complete", "done", self.context.copy()


    # ──────────────────────────────────────────────────────────
    # Упрощённый запуск (без прогресс-callback)
    # ──────────────────────────────────────────────────────────
    def run_pipeline(self, user_query: str, pdf_folder: str):
        """
        «Быстрый» запуск без пошагового отслеживания:
        просто выполняет ingest→retrieval→reasoning→qa_over_pdf→synthesis и возвращает context.
        """
        for stage, status, ctx in self.run_pipeline_with_progress(user_query, pdf_folder):
            pass
        return self.context


# ──────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    protocol_file = "n_papers_protocol.yaml"
    pipeline = MultiPDFAnalysisPipeline(protocol_file)

    user_query  = input("Введите ваш запрос для анализа документов: ").strip()
    pdf_folder  = input(f"Введите путь к папке с PDF (или Enter для дефолта): ").strip() or r"C:\Work\diplom2\rag_on_papers\data\n_papers"

    # Пошаговый callback-пример
    def progress_callback(stage_name, status, ctx):
        print(f"[{stage_name}] {status}")

    for stage, status, ctx in pipeline.run_pipeline_with_progress(user_query, pdf_folder, progress_callback):
        print(f"[{stage}] {status}")

    # После завершения можно посмотреть финальный результат:
    final_answer = pipeline.context.get("synthesis_output", "")
    print("\n=== Pipeline finished ===")
    print(final_answer)
