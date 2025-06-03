#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDFAnalysisPipeline с OpenAI и стриминговым выводом, где тексты промптов хранятся в YAML.

Шаги:
 1) Берёт очищенный Markdown из GridFS (file_buffer_db → filename_clean.md).
 2) Переиспользует ingest_from_gridfs из papers_vectorization_MONGO
    (очищает paper_chunks, разбивает MD на чанки и сохраняет эмбеддинги OpenAI).
 3) Переиспользует build_vector_index из papers_vectorization_MONGO для создания/обновления векторного индекса.
 4) Для ретривала использует get_relevant_chunks из papers_retrieval_MONGO.
 5) Генеративные этапы (retrieval_prompt, reasoning, qa_over_pdf) читают “action” из YAML
    и используют ChatOpenAI.stream для стриминга ответа.
 6) Все промежуточные шаги reasoning выводятся на экран.
"""

import os
import sys
import tempfile
from pathlib import Path

import yaml
import gridfs
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from langchain_openai import ChatOpenAI

# Переиспользуемые функции
from src.retrieval.papers_retrieval_MONGO import get_relevant_chunks
from src.data_processing.papers_vectorization_MONGO import clear_all_before_ingest, mongo_client

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Константы / Настройки
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("✘ MONGODB_URI not set in environment (.env)", file=sys.stderr)
    sys.exit(1)

ARXIV_DB_NAME    = "arxiv_db"
CHUNKS_COLL_NAME = "paper_chunks"
GRIDFS_DB_NAME   = "file_buffer_db"

# Имя векторного индекса (совпадает с CHNK_INDEX из papers_vectorization_MONGO)

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательная функция: получить путь к очищенному MD из GridFS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_clean_markdown_from_gridfs(pdf_path: str) -> str:
    """
    Ищет в GridFS (file_buffer_db) файл "<stem>_clean.md", сохраняет локально
    во временную папку и возвращает путь. Если не найден, возвращает "".
    """
    stem = Path(pdf_path).stem
    clean_filename = f"{stem}_clean.md"

    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    db = client[GRIDFS_DB_NAME]
    fs = gridfs.GridFS(db)

    grid_out = fs.find_one({"filename": clean_filename})
    if not grid_out:
        print(f"[fetch_clean_markdown] ✘ '{clean_filename}' not found in GridFS.", file=sys.stderr)
        client.close()
        return ""

    content_bytes = grid_out.read()
    client.close()

    tmp_dir = tempfile.mkdtemp(prefix="clean_md_")
    tmp_path = Path(tmp_dir) / clean_filename
    try:
        tmp_path.write_bytes(content_bytes)
    except Exception as e:
        print(f"[fetch_clean_markdown] ✘ Cannot write to {tmp_path}: {e}", file=sys.stderr)
        return ""
    return str(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# PDFAnalysisPipeline
# ──────────────────────────────────────────────────────────────────────────────

class PDFAnalysisPipeline:
    def __init__(self, protocol_path: str):
        self.protocol = self._load_protocol(protocol_path)
        self.context = {}
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    def _load_protocol(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _resolve_parameters(self, params: dict) -> dict:
        """
        Подставляет значения из self.context для шаблонов вида "{{var}}".
        """
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                key = v[2:-2].strip()
                resolved[k] = self.context.get(key, "")
            else:
                resolved[k] = v
        return resolved

    def _stream_llm(self, prompt: str) -> str:
        """
        Стримим ответ LLM, печатаем построчно и накапливаем в одну строку.
        """
        streamed = ""
        for chunk in self.llm.stream([("user", prompt)]):
            print(chunk.content, end="", flush=True)
            streamed += chunk.content
        print()
        return streamed

    def run_pipeline_with_progress(self, user_query: str, user_pdf: str, progress_callback=None):
        """
        Итерация по шагам протокола: отдаёт (stage, status, context).
        Если передан progress_callback, его вызывают в шаге reasoning для каждого CoT-подшага.
        """
        self.context.update({"user_query": user_query, "user_pdf": user_pdf})

        for step in self.protocol["pipeline"]:
            stage  = step["stage"]
            action = step.get("action", "").strip()
            print(f"\n>> [{stage}]")  # Заголовок этапа
            yield stage, "in_progress", dict(self.context)

            params = self._resolve_parameters(step.get("parameters", {}))
            self.context.update(params)

            if stage == "retrieval":
                # Формируем prompt из action: подставляем параметры
                # prompt = action.format(**self.context)
                # print(f"-- Prompt for retrieval:\n{prompt}\n")
                # Здесь не стримим, просто выводим запрос и получаем чанки
                docs = get_relevant_chunks(self.context["user_query"], top_n=params.get("top_n", 3))
                self.context["retrieval_output"] = docs
                print(f"-- Retrieved {len(docs)} chunks.")

            elif stage == "reasoning":
                # Берём шаблон prompt из action
                # base_prompt = action.format(**self.context)
                # print(f"-- Prompt for reasoning:\n{base_prompt}\n")

                # Разбиваем на подшага CoT: здесь предполагаем, что action даёт базовый запрос
                # Мы всё ещё делаем multi-step CoT, но первый шаг – это base_prompt
                # Шаг 0: initial answer
                # initial_answer = self._stream_llm(step.get("prompt_template", "").strip())
                reasoning_steps = []
                # all_answers = [initial_answer]
                # if progress_callback:
                #    progress_callback(0, "Initial Answer", initial_answer)

                # ───── внутри run_pipeline_with_progress ─────
                for step in self.protocol["pipeline"]:
                    stage = step["stage"]
                    prompt = step.get("prompt_template", "").strip()  # вместо action
                    cot_qs = step.get("cot_questions")  # список вопросов, если есть
                    ...
                    # — retrieval —
                    '''if stage == "retrieval":
                        print(prompt.format(**self.context), "\n")
                        docs = get_relevant_chunks(...)'''

                    # — reasoning —
                    if stage == "reasoning":
                        base_prompt = prompt.format(**self.context)
                        print("-- Prompt for reasoning:\n", base_prompt, "\n")

                        initial_answer = self._stream_llm(base_prompt)
                        reasoning_steps, all_answers = [], [initial_answer]

                        questions = cot_qs or [  # если в YAML нет списка – дефолт
                            "Identify potential weaknesses or gaps in your answer.",
                            "How can you address those gaps?",
                            "What extra details would improve your answer?",
                            "Does your answer fully align with the query?",
                            "How confident are you, and what would increase confidence?",
                            "Summarize an improved final answer."
                        ]

                        context_chunks = "\n".join(
                            f"Chunk {i + 1}: {ch.page_content}"
                            for i, ch in enumerate(self.context.get("retrieval_output", [])[:5])
                        )

                        for i, question in enumerate(questions, 1):
                            prev = "\n".join(f"Answer {j + 1}: {ans}"
                                             for j, ans in enumerate(all_answers))
                            iteration_prompt = f"""Iteration {i}:
                Query: {self.context['user_query']}

                Context:
                {context_chunks}

                Previous Answers:
                {prev}

                Refinement Question: {question}

                Please refine your response."""
                            analysis = self._stream_llm(iteration_prompt)
                            reasoning_steps.append({"step": question,
                                                    "analysis": analysis,
                                                    "iteration": i})
                            all_answers.append(analysis)
                            if progress_callback:
                                progress_callback(i, question, analysis)

                final_answer = all_answers[-1]
                if progress_callback:
                    progress_callback(0, "CoT Complete", final_answer)

                self.context.update({
                    "initial_answer":   initial_answer,
                    "reasoning_steps":  reasoning_steps,
                    "final_answer":     final_answer
                })

            elif stage == "qa_over_pdf":
                prompt = action.format(**self.context)
                print(f"-- Prompt for QA:\n{prompt}\n")
                answer = self._stream_llm(prompt)
                self.context["qa_over_pdf_output"] = answer

            else:
                func = getattr(self, step["function"], None)
                if func:
                    output = func(**params)
                    self.context[f"{step['function']}_output"] = output
                else:
                    print(f"[ERROR] Function '{step['function']}' not found", file=sys.stderr)
                    yield stage, "error", dict(self.context)
                    continue

            yield stage, "done", dict(self.context)

        yield "pipeline_complete", "", dict(self.context)

    def run_pipeline(self, *args, **kwargs) -> dict:
        for _ in self.run_pipeline_with_progress(*args, **kwargs):
            pass
        return self.context


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    protocol_file = "one_paper_protocol.yaml"
    pipeline = PDFAnalysisPipeline(protocol_file)

    FILE_BUFFER_DB = "file_buffer_db"  # БД-буфер для GridFS
    client = mongo_client()
    db_buffer  = client[FILE_BUFFER_DB]

    clear_all_before_ingest(db_buffer)
    user_pdf = input("Enter path to PDF: ").strip()
    user_query = input("Enter your query: ").strip()

    def progress_callback(step_num, question, analysis):
        print(f"\n--- CoT Step {step_num}: {question} ---")
        print(analysis)
        print("----------------------------\n")

    for stage, status, ctx in pipeline.run_pipeline_with_progress(user_query, user_pdf, progress_callback):
        print(f"[{stage}] {status}")

    final = pipeline.context.get("final_answer")
    if final:
        print("\n🎯 Final Answer:")
        print(final)
