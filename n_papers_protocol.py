#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MultiPDFAnalysisPipeline:
  1) Clears collections и GridFS buckets.
  2) Ingests PDFs → GridFS → metadata.
  3) Vectorizes в arxiv_db.multi_paper_chunks.
  4) Builds vector index.
  5) Uses get_relevant_chunks_with_totals() for retrieval.
  6) Reasoning (CoT) через llm.invoke.
  7) QA_over_pdf с потоковой передачей.
  8) (Optional) Synthesis.
  9) stream_final_reasoning() — стримит **только** финальный шаг CoT.
"""

import yaml
from pathlib import Path
from typing import List, Any, Dict

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Утилиты из papers_vectorization_MONGO.py
from src.data_processing.papers_vectorization_MONGO import (
    mongo_client as pv_mongo_client,
    ingest_pdfs_to_markdowns,
    ingest_multi_chunks,
    build_vector_index,
    clear_all_before_ingest,
    clear_collection,
)

# Новая функция Retrieval
from src.retrieval.papers_retrieval_MONGO import get_relevant_chunks_with_totals

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────────────────

ARXIV_DB_NAME       = "arxiv_db"
MULTI_CHUNK_COL     = "multi_paper_chunks"
MULTI_CHNK_INDEX    = "multi_chunk_embedding_index"

FILE_BUFFER_DB_NAME = "file_buffer_db"
PIPELINE_FS_BUCKET  = "pipeline_fs"
PDF_MARKDOWNS_COL   = "pdf_markdowns"

LLM_MODEL_NAME      = "gpt-4o-mini"   # ChatGPT-4o mini

# ──────────────────────────────────────────────────────────────

class MultiPDFAnalysisPipeline:
    def __init__(self, protocol_path: str):
        """
        Загружает YAML-протокол, инициализирует MongoDB, GridFS и LLM.
        """
        self.protocol = self._load_protocol(protocol_path)
        self.context: Dict[str, Any] = {}

        # MongoDB clients
        self.client_arxiv = pv_mongo_client()
        self.db_arxiv     = self.client_arxiv[ARXIV_DB_NAME]

        self.client_buf   = pv_mongo_client()
        self.db_buffer    = self.client_buf[FILE_BUFFER_DB_NAME]

        # ChatGPT-4o mini (streaming=False для invoke)
        self.llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.4, streaming=False)

    def _load_protocol(self, path: str) -> Dict[str, Any]:
        """
        Загружает YAML-протокол.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Форматирует шаблонную строку вида "{user_query}", "{context_chunks}", "{context_text}".
        """
        return template.format(**variables)

    # ──────────────────────────────────────────────────────────
    # Step 1: Ingest PDFs и build index
    # ──────────────────────────────────────────────────────────
    def run_ingest_and_index(self, pdf_folder: str, skip_index: bool = False) -> Dict[str, Any]:
        """
        1) Clear: GridFS 'pipeline_fs', коллекция 'pdf_markdowns', 'multi_paper_chunks'.
        2) ingest_pdfs_to_markdowns → GridFS + pdf_markdowns.
        3) ingest_multi_chunks → multi_paper_chunks.
        4) build_vector_index → строит векторный индекс.
        """
        print("\n[STEP] Clearing data before ingestion:")
        clear_collection(self.db_buffer, PDF_MARKDOWNS_COL)
        print(f"  • Cleared collection '{PDF_MARKDOWNS_COL}'.")
        clear_all_before_ingest(self.db_buffer, bucket=PIPELINE_FS_BUCKET)
        print(f"  • Cleared GridFS bucket '{PIPELINE_FS_BUCKET}'.")
        clear_collection(self.db_arxiv, MULTI_CHUNK_COL)
        print(f"  • Cleared collection '{MULTI_CHUNK_COL}'.")

        print("\n[STEP] Ingesting PDFs into GridFS and saving metadata:")
        ingest_pdfs_to_markdowns(self.db_buffer, Path(pdf_folder))

        print("\n[STEP] Chunking and vectorizing (multi_paper_chunks):")
        ingest_multi_chunks(self.db_arxiv, self.db_buffer)

        if not skip_index:
            print("\n[STEP] Building vector index for multi_paper_chunks:")
            build_vector_index(self.db_arxiv, MULTI_CHUNK_COL, MULTI_CHNK_INDEX)

        print("\n✅ Ingestion and indexing completed.\n")
        return {
            "status": "ingested_and_indexed",
            "multi_chunk_count": self.db_arxiv[MULTI_CHUNK_COL].count_documents({})
        }

    # ──────────────────────────────────────────────────────────
    # Step 2: Retrieval
    # ──────────────────────────────────────────────────────────
    def run_retrieval(self, query: str, top_n: int = 5, min_similarity: float = 0.0) -> List[Document]:
        """
        Векторный поиск через get_relevant_chunks_with_totals.
        """
        print(f"\n[STEP] Retrieval: searching top {top_n} relevant chunks for '{query}' …")
        docs = get_relevant_chunks_with_totals(
            query=query,
            top_n=top_n,
            min_similarity=min_similarity,
            num_candidates=150,
            db_name=ARXIV_DB_NAME,
            chunk_collection=MULTI_CHUNK_COL,
            chunk_index_name=MULTI_CHNK_INDEX
        )
        print(f"[INFO] Found {len(docs)} relevant chunks.")
        return docs

    # ──────────────────────────────────────────────────────────
    # Step 3: Reasoning (CoT) через llm.invoke
    # ──────────────────────────────────────────────────────────
    def run_reasoning(self, query: str, chunks: List[Document]) -> Dict[str, Any]:
        """
        1) Формируем context_chunks (первые 5 чанков, “Chunk i/total”).
        2) Initial Answer через llm.invoke → result.content.
        3) По каждому cot_question — llm.invoke → result.content.
        Возвращаем dict с initial_answer, reasoning_steps, final_answer.
        """
        print(f"\n[STEP] Reasoning (CoT) for query: {query}")

        # Собираем первые 5 чанков в одну строку
        context_lines = []
        for doc in chunks[:5]:
            fn      = doc.metadata["source"]
            idx     = doc.metadata["chunk_index"]
            total   = doc.metadata["total_chunks"]
            snippet = doc.page_content[:200].replace("\n", " ").strip()
            human_i = idx + 1
            label   = f"Chunk {human_i}/{total} (Source: {fn})"
            context_lines.append(f"{label}: {snippet}…")
        context_chunks = "\n".join(context_lines)
        self.context["context_chunks"] = context_chunks

        # Достаём из YAML шаблон и список CoT-вопросов
        reasoning_stage = next((s for s in self.protocol["pipeline"] if s["stage"] == "reasoning"), None)
        if reasoning_stage is None:
            raise RuntimeError("Reasoning stage not found in protocol.")

        prompt_template = reasoning_stage.get("prompt_template", "").strip()
        cot_questions   = reasoning_stage.get("cot_questions", [])

        # —————————————
        # Step 0: Initial Answer
        initial_prompt = self._render_template(prompt_template, {
            "user_query":     query,
            "context_chunks": context_chunks
        })
        print("\n[LLM] Invoking Initial Answer…")
        chat0 = self.llm.invoke(initial_prompt)
        initial_answer = chat0.content

        reasoning_steps = [
            {"step": "Initial Answer", "analysis": initial_answer, "iteration": 0}
        ]
        all_answers = [initial_answer]

        # —————————————
        # Итерации CoT
        for i, question in enumerate(cot_questions, start=1):
            prev_answers = "\n\n".join(f"Answer {j+1}: {ans}" for j, ans in enumerate(all_answers))
            iteration_prompt = (
                f"Chain-of-Thought Iteration {i}:\n"
                f"Query: {query}\n\n"
                f"Context Chunks:\n{context_chunks}\n\n"
                f"Previous Answers:\n{prev_answers}\n\n"
                f"Refinement Question: {question}\n\n"
                f"Пожалуйста, проанализируйте предыдущие ответы, уточните ответ "
                f"и укажите источники (имена файлов или chunk_ID)."
            )
            print(f"\n[LLM] Invoking Iteration {i} ({question})…")
            chat_i = self.llm.invoke(iteration_prompt)
            iter_answer = chat_i.content

            reasoning_steps.append({
                "step":      question,
                "analysis":  iter_answer,
                "iteration": i
            })
            all_answers.append(iter_answer)

        # —————————————
        final_answer = all_answers[-1]
        print(f"\n[LLM] Final Answer (Reasoning):\n{final_answer}\n")

        # Сохраняем reasoning_output в context
        self.context["reasoning_output"] = {
            "initial_answer": initial_answer,
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer
        }

        return {
            "initial_answer": initial_answer,
            "reasoning_steps": reasoning_steps,
            "final_answer": final_answer
        }

    # ──────────────────────────────────────────────────────────
    # Генератор «пошагового» CoT
    # ──────────────────────────────────────────────────────────
    def run_reasoning_step_by_step(self, query: str, chunks: List[Document]):
        """
        Generator, который возвращает на каждом шаге:
        (iteration_number, question_text, full_answer_for_step).
        Использует llm.invoke для каждого шага, и сохраняет reasoning_output в context.
        """
        # 1) Извлекаем из YAML CoT-вопросы и шаблон
        with open("n_papers_protocol.yaml", "r", encoding="utf-8") as f:
            pipeline = yaml.safe_load(f)["pipeline"]
        reasoning_stage = next((s for s in pipeline if s["stage"] == "reasoning"), None)
        cot_questions   = reasoning_stage.get("cot_questions", [])
        prompt_template = reasoning_stage.get("prompt_template", "").strip()

        # 2) Собираем первые 5 чанков
        context_lines = []
        for doc in chunks[:5]:
            fn      = doc.metadata["source"]
            idx     = doc.metadata["chunk_index"]
            total   = doc.metadata["total_chunks"]
            snippet = doc.page_content[:200].replace("\n", " ").strip()
            human_i = idx + 1
            label   = f"Chunk {human_i}/{total} (Source: {fn})"
            context_lines.append(f"{label}: {snippet}…")
        context_chunks = "\n".join(context_lines)
        self.context["context_chunks"] = context_chunks

        # —————————————
        # Step 0: Initial Answer
        initial_prompt = prompt_template.format(
            user_query=query,
            context_chunks=context_chunks
        )
        chat0 = self.llm.invoke(initial_prompt)
        answer_0 = chat0.content
        reasoning_steps = [
            {"step": "Initial Answer", "analysis": answer_0, "iteration": 0}
        ]
        all_answers = [answer_0]
        yield (0, "Initial Answer", answer_0)

        # —————————————
        # Итерации CoT
        for i, question in enumerate(cot_questions, start=1):
            prev_answers = "\n\n".join(f"Answer {j+1}: {ans}" for j, ans in enumerate(all_answers))
            iteration_prompt = (
                f"Chain-of-Thought Iteration {i}:\n"
                f"Query: {query}\n\n"
                f"Context Chunks:\n{context_chunks}\n\n"
                f"Previous Answers:\n{prev_answers}\n\n"
                f"Refinement Question: {question}\n\n"
                f"Пожалуйста, проанализируйте предыдущие ответы, уточните ответ "
                f"и укажите источники (имена файлов или chunk_ID)."
            )
            chat_i = self.llm.invoke(iteration_prompt)
            answer_i = chat_i.content
            reasoning_steps.append({
                "step":      question,
                "analysis":  answer_i,
                "iteration": i
            })
            all_answers.append(answer_i)
            yield (i, question, answer_i)

        # —————————————
        # Финальный ответ (последний шаг)
        final_answer = all_answers[-1]
        reasoning_steps.append({
            "step": "CoT Complete",
            "analysis": final_answer,
            "iteration": 0
        })

        # Сохраняем весь вывод CoT-шагов в context:
        self.context["reasoning_output"] = {
            "initial_answer":  reasoning_steps[0]["analysis"],
            "reasoning_steps": reasoning_steps,
            "final_answer":    final_answer
        }

        # Шаг "CoT Complete" не возвращаем на сайт (только в context),
        # поэтому мы здесь НЕ yield (0, "CoT Complete", final_answer)

    # ──────────────────────────────────────────────────────────
    # Step 4: QA_over_pdf (струминг)
    # ──────────────────────────────────────────────────────────
    def run_qa_over_pdf(self, query: str, chunks: List[Document]) -> str:
        """
        Формируем контекст из всех найденных чанков (с маркировкой “chunk i/total”),
        затем стримим ответ LLM.
        """
        print(f"\n[STEP] QA_over_pdf: invoking LLM…")

        context_lines = []
        for doc in chunks:
            fn      = doc.metadata["source"]
            idx     = doc.metadata["chunk_index"]
            total   = doc.metadata["total_chunks"]
            human_i = idx + 1
            label   = f"Source: {fn} (chunk {human_i}/{total})"
            content = doc.page_content.replace("\n", " ").strip()
            context_lines.append(f"{label}\n{content}")
        context_text = "\n\n".join(context_lines)
        self.context["context_text"] = context_text

        qa_stage = next((s for s in self.protocol["pipeline"] if s["stage"] == "qa_over_pdf"), None)
        if qa_stage is None:
            raise RuntimeError("QA_over_pdf stage not found in protocol.")

        qa_template = qa_stage.get("prompt_template", "").strip()
        qa_prompt = self._render_template(qa_template, {
            "user_query":   query,
            "context_text": context_text
        })

        print("\n[LLM STREAM] QA_over_pdf:")
        streamed_qa = ""
        for chunk in self.llm.stream([("user", qa_prompt)]):
            print(chunk.content, end="", flush=True)
            streamed_qa += chunk.content
        print()
        return streamed_qa

    # ──────────────────────────────────────────────────────────
    # Step 5: Synthesis (объединяем reasoning + QA)
    # ──────────────────────────────────────────────────────────
    def run_synthesis(self, reasoning_output: Dict[str, Any], qa_output: str) -> str:
        """
        Объединяет reasoning_output и qa_output в единый текст.
        """
        print("\n[STEP] Synthesis: merging reasoning + QA…")
        initial = reasoning_output.get("initial_answer", "")
        final   = reasoning_output.get("final_answer", "")
        combined = (
            f"Initial reasoning answer:\n{initial}\n\n"
            f"Final reasoning answer:\n{final}\n\n"
            f"QA_over_pdf answer:\n{qa_output}"
        )
        print(f"\n[Combined Output]:\n{combined}\n")
        return combined

    # ──────────────────────────────────────────────────────────
    # Метод для стриминга **только последнего** (финального) шага CoT
    # ──────────────────────────────────────────────────────────
    def stream_final_reasoning(self, query: str, chunks: List[Document]):
        """
        Стримит только финальный шаг CoT-итерации:
        берёт reasoning_output из self.context,
        восстанавливает prompt последней итерации и запускает llm.stream.
        """
        reasoning_output = self.context.get("reasoning_output", {})
        if not reasoning_output:
            raise RuntimeError("No reasoning_output found in context. Call run_reasoning or run_reasoning_step_by_step first.")

        # Собираем контекст снова
        context_chunks  = self.context.get("context_chunks", "")
        reasoning_steps = reasoning_output.get("reasoning_steps", [])
        if len(reasoning_steps) < 2:
            # Если CoT состояла только из Initial Answer, стримим его заново
            last_question   = "Initial Answer"
            prev_answers    = ""
            iteration_index = 0
        else:
            # Последний шаг CoT (не считая “CoT Complete”, который мы не выводили)
            iteration_index = len(reasoning_steps) - 1
            last_step       = reasoning_steps[-1]
            last_question   = last_step["step"]
            all_prev        = [step["analysis"] for step in reasoning_steps[:-1]]
            prev_answers    = "\n\n".join(f"Answer {i+1}: {ans}" for i, ans in enumerate(all_prev))

        iteration_prompt = (
            f"Chain-of-Thought Iteration {iteration_index}:\n"
            f"Query: {query}\n\n"
            f"Context Chunks:\n{context_chunks}\n\n"
            f"Previous Answers:\n{prev_answers}\n\n"
            f"Refinement Question: {last_question}\n\n"
            f"Пожалуйста, проанализируйте предыдущие ответы, уточните ответ "
            f"и укажите источники (имена файлов или chunk_ID)."
        )

        for chunk in self.llm.stream([("user", iteration_prompt)]):
            yield chunk.content

    # ──────────────────────────────────────────────────────────
    # Основной generator с callback
    # ──────────────────────────────────────────────────────────
    def run_pipeline_with_progress(
        self,
        user_query: str,
        pdf_folder: str,
        progress_callback=None
    ):
        """
        1) Ingest+Index
        2) Retrieval
        3) Reasoning
        4) QA_over_pdf
        5) Synthesis
        Yields (stage, status, full_context_dict).
        """
        self.context.clear()
        self.context["user_query"] = user_query
        self.context["pdf_folder"]  = pdf_folder

        # — 1) Ingest + Index
        stage = "ingest_and_index"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        ingest_result = self.run_ingest_and_index(pdf_folder)
        self.context.update(ingest_result)
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # — 2) Retrieval
        stage = "retrieval"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        top_chunks: List[Document] = self.run_retrieval(user_query, top_n=5)
        self.context["retrieval_output"] = top_chunks
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # — 3) Reasoning
        stage = "reasoning"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        reasoning_out = self.run_reasoning(user_query, top_chunks)
        # run_reasoning уже записал reasoning_output в context
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # — 4) QA_over_pdf
        stage = "qa_over_pdf"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        qa_out = self.run_qa_over_pdf(user_query, top_chunks)
        self.context["qa_over_pdf_output"] = qa_out
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # — 5) Synthesis
        stage = "synthesis"
        if progress_callback:
            progress_callback(stage, "in_progress", self.context.copy())
        synth_out = self.run_synthesis(reasoning_out, qa_out)
        self.context["synthesis_output"] = synth_out
        if progress_callback:
            progress_callback(stage, "done", self.context.copy())

        # Финальный yield
        yield "pipeline_complete", "done", self.context.copy()

    # ──────────────────────────────────────────────────────────
    # Quick run без callback
    # ──────────────────────────────────────────────────────────
    def run_pipeline(self, user_query: str, pdf_folder: str) -> Dict[str, Any]:
        for stage, status, ctx in self.run_pipeline_with_progress(user_query, pdf_folder):
            pass
        return self.context

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    protocol_file = "n_papers_protocol.yaml"
    pipeline = MultiPDFAnalysisPipeline(protocol_file)

    user_query = input("Enter your query: ").strip()
    pdf_folder = input("Path to folder with PDFs (или Enter для пути по умолчанию): ").strip() \
                 or r"C:\Work\diplom2\rag_on_papers\data\n_papers"

    def progress_callback(stage_name, status, ctx):
        print(f"[{stage_name}] {status}")

    for stage, status, ctx in pipeline.run_pipeline_with_progress(user_query, pdf_folder, progress_callback):
        print(f"[{stage}] {status}")

    print("\n=== Final Answer ===")
    print(pipeline.context.get("synthesis_output", ""))
