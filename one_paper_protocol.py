#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDFAnalysisPipeline with separate upload_and_vectorize and query methods:

1) upload_and_vectorize_pdf(pdf_path):
   - Clears default GridFS in file_buffer_db.
   - Saves the PDF to GridFS.
   - Converts PDF → raw Markdown, removes tables → clean Markdown.
   - Saves clean Markdown to GridFS.
   - Ingests all *_clean.md files from GridFS into file_buffer_db.paper_chunks.
   - Rebuilds the vector index on paper_chunks.

2) run_query_with_progress(user_query, progress_callback):
   - Uses already-vectorized chunks in file_buffer_db.paper_chunks.
   - Performs retrieval → chain-of-thought reasoning → QA over PDF.
   - Streams intermediate LLM outputs via ChatOpenAI.
   - Yields (stage, status, context) for frontend progress tracking.
"""

import os
import sys
import tempfile
from pathlib import Path

import yaml

from langchain_openai import ChatOpenAI

from src.retrieval.papers_retrieval_MONGO import get_relevant_chunks
from src.data_processing.papers_vectorization_MONGO import (
    ingest_from_gridfs,
    build_vector_index,
    mongo_client
)
from my_utils import (
    clear_all_files_from_gridfs,
    save_pdf_to_gridfs,
    save_text_to_gridfs,
    remove_markdown_tables,
    convert_pdf_to_md
)

from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("✘ MONGODB_URI not set in environment (.env)", file=sys.stderr)
    sys.exit(1)

# Database and collection constants
ARXIV_DB_NAME     = "arxiv_db"
PAPER_CHUNK_COL   = "paper_chunks"
FILE_BUFFER_DB    = "file_buffer_db"
VECTOR_INDEX_NAME = "chunk_embedding_index"


class PDFAnalysisPipeline:
    def __init__(self, protocol_path: str):
        """
        Initialize pipeline with a YAML protocol defining retrieval, reasoning, and QA steps.
        """
        self.protocol = self._load_protocol(protocol_path)
        self.context = {}
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    def _load_protocol(self, path: str) -> dict:
        """
        Load the pipeline definition from a YAML file.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _stream_llm(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and stream the response to stdout, accumulating it into one string.
        """
        streamed = ""
        for chunk in self.llm.stream([("user", prompt)]):
            print(chunk.content, end="", flush=True)
            streamed += chunk.content
        print()
        return streamed

    def upload_and_vectorize_pdf(self, pdf_path: str) -> bool:
        """
        1) Clear default GridFS (file_buffer_db).
        2) Save the PDF to GridFS.
        3) Convert PDF → raw Markdown, remove tables → clean Markdown, save clean Markdown to GridFS.
        4) Ingest all *_clean.md files from GridFS into file_buffer_db.paper_chunks.
        5) Rebuild vector index 'chunk_embedding_index' on paper_chunks.
        Returns True on success, False on any error.
        """
        # 1) Clear existing files in default GridFS of file_buffer_db
        deleted_count = clear_all_files_from_gridfs()
        print(f"[upload_and_vectorize] Cleared default GridFS: {deleted_count} files removed.")

        if not os.path.isfile(pdf_path):
            print(f"✘ PDF not found at: {pdf_path}", file=sys.stderr)
            return False

        # Read PDF bytes
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        base_name = Path(pdf_path).stem
        pdf_filename = f"{base_name}.pdf"
        clean_md_filename = f"{base_name}_clean.md"

        # 2) Save PDF into GridFS
        try:
            pdf_oid = save_pdf_to_gridfs(pdf_bytes, pdf_filename)
        except Exception as e:
            print(f"✘ Error saving PDF to GridFS: {e}", file=sys.stderr)
            return False
        print(f"[upload_and_vectorize] ✔ PDF saved to GridFS (id={pdf_oid}).")

        # 3) Convert PDF → raw Markdown, then remove tables → clean Markdown
        tmp_dir = tempfile.mkdtemp(prefix="pdf_to_md_")
        tmp_pdf_path = os.path.join(tmp_dir, pdf_filename)
        with open(tmp_pdf_path, "wb") as f_tmp:
            f_tmp.write(pdf_bytes)

        raw_md_path = os.path.join(tmp_dir, f"{base_name}.md")
        print("[upload_and_vectorize] ⏳ Converting PDF → Markdown …")
        try:
            convert_pdf_to_md(tmp_pdf_path, raw_md_path, show_progress=False)
        except Exception as e:
            print(f"✘ convert_pdf_to_md failed: {e}", file=sys.stderr)
            return False
        print(f"[upload_and_vectorize] ✔ Markdown saved locally: {raw_md_path}")

        clean_md_path = os.path.join(tmp_dir, clean_md_filename)
        print("[upload_and_vectorize] ⏳ Removing tables from Markdown …")
        try:
            remove_markdown_tables(raw_md_path, clean_md_path)
        except Exception as e:
            print(f"✘ remove_markdown_tables failed: {e}", file=sys.stderr)
            return False
        print(f"[upload_and_vectorize] ✔ Clean Markdown saved locally: {clean_md_path}")

        # 4) Save clean Markdown into GridFS
        with open(clean_md_path, "r", encoding="utf-8") as f_clean:
            clean_md_text = f_clean.read()
        try:
            clean_md_oid = save_text_to_gridfs(clean_md_text, clean_md_filename)
        except Exception as e:
            print(f"✘ Error saving clean Markdown to GridFS: {e}", file=sys.stderr)
            return False
        print(f"[upload_and_vectorize] ✔ Clean Markdown saved to GridFS (id={clean_md_oid}).")

        # 5) Remove temporary files and directory
        try:
            os.remove(raw_md_path)
            os.remove(clean_md_path)
            os.remove(tmp_pdf_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass
        print(f"[upload_and_vectorize] ✔ Temporary files removed: {tmp_dir}")

        # 6) Ingest all *_clean.md from GridFS → file_buffer_db.paper_chunks
        print("[upload_and_vectorize] ⏳ Clearing and vectorizing clean Markdown into paper_chunks …")
        client = mongo_client()
        db_buffer = client[FILE_BUFFER_DB]
        ingest_from_gridfs(db_buffer)
        print("[upload_and_vectorize] ✔ All clean Markdown chunks vectorized and stored.")

        # 7) Rebuild vector index on paper_chunks in file_buffer_db
        print(f"[upload_and_vectorize] ⏳ Building vector index '{VECTOR_INDEX_NAME}' …")
        build_vector_index(db_buffer, PAPER_CHUNK_COL, VECTOR_INDEX_NAME)
        print(f"[upload_and_vectorize] ✔ Vector index '{VECTOR_INDEX_NAME}' is ready.")

        client.close()
        return True

    def run_query_with_progress(self, user_query: str, progress_callback=None):
        """
        Perform retrieval → reasoning → qa_over_pdf using already-vectorized chunks.
        Yields (stage, status, context) for UI progress.
        """
        # Store the query in context
        self.context["user_query"] = user_query

        # 1) Retrieval
        print("\n>> [retrieval]")
        yield "retrieval", "in_progress", dict(self.context)

        docs = get_relevant_chunks(
            query=user_query,
            top_n=self.protocol["pipeline"][0].get("parameters", {}).get("top_n", 3),
            db_name=FILE_BUFFER_DB,
            chunk_collection=PAPER_CHUNK_COL,
            chunk_index_name=VECTOR_INDEX_NAME
        )
        self.context["retrieval_output"] = docs

        print(f"-- Retrieved {len(docs)} chunks:")
        for i, ch in enumerate(docs, 1):
            print(f"\n--- Chunk {i} ---")
            print(ch.page_content.strip())
            print("---------------------")

        yield "retrieval", "done", dict(self.context)

        # 2) Reasoning (chain-of-thought)
        reasoning_step = next((s for s in self.protocol["pipeline"] if s["stage"] == "reasoning"), None)
        if reasoning_step:
            print("\n>> [reasoning]")
            yield "reasoning", "in_progress", dict(self.context)

            prompt_template = reasoning_step.get("prompt_template", "").strip()
            base_prompt = prompt_template.format(**self.context)
            print(f"-- Prompt for reasoning:\n{base_prompt}\n")

            # Initial answer from LLM
            initial_answer = self._stream_llm(base_prompt)
            reasoning_steps = []
            all_answers = [initial_answer]
            if progress_callback:
                progress_callback(0, "Initial Answer", initial_answer)

            cot_qs = reasoning_step.get("cot_questions")
            questions = cot_qs or [
                "Identify potential weaknesses or gaps in your answer.",
                "How can you address those weaknesses?",
                "What additional details would improve your answer?",
                "Does your answer fully address the query?",
                "How confident are you, and what would increase confidence?",
                "Summarize an improved final answer."
            ]

            # Prepare up to 5 retrieved chunks for context
            context_chunks = "\n".join(
                f"Chunk {i+1}: {ch.page_content}"
                for i, ch in enumerate(self.context.get("retrieval_output", [])[:5])
            )

            # Iterate over CoT refinement questions
            for i, question in enumerate(questions, 1):
                prev_answers = "\n".join(f"Answer {j+1}: {ans}" for j, ans in enumerate(all_answers))
                iteration_prompt = (
                    f"Iteration {i}:\n"
                    f"Query: {self.context['user_query']}\n\n"
                    f"Context:\n{context_chunks}\n\n"
                    f"Previous Answers:\n{prev_answers}\n\n"
                    f"Refinement Question: {question}\n\n"
                    f"Please refine your response."
                )
                analysis = self._stream_llm(iteration_prompt)
                reasoning_steps.append({
                    "step": question,
                    "analysis": analysis,
                    "iteration": i
                })
                all_answers.append(analysis)
                if progress_callback:
                    progress_callback(i, question, analysis)

            # Final answer is the last refinement
            final_answer = all_answers[-1]
            if progress_callback:
                progress_callback(0, "CoT Complete", final_answer)

            self.context.update({
                "initial_answer":  initial_answer,
                "reasoning_steps": reasoning_steps,
                "final_answer":    final_answer
            })

            yield "reasoning", "done", dict(self.context)

        # 3) QA Over PDF
        qa_step = next((s for s in self.protocol["pipeline"] if s["stage"] == "qa_over_pdf"), None)
        if qa_step:
            print("\n>> [qa_over_pdf]")
            yield "qa_over_pdf", "in_progress", dict(self.context)

            prompt_text = qa_step.get("action", "").format(**self.context)
            print(f"-- Prompt for QA:\n{prompt_text}\n")
            answer = self._stream_llm(prompt_text)
            self.context["qa_over_pdf_output"] = answer

            yield "qa_over_pdf", "done", dict(self.context)

        # Final pipeline complete
        yield "pipeline_complete", "", dict(self.context)


# Entry point for command-line usage: full upload+query in one run
if __name__ == "__main__":
    protocol_file = "one_paper_protocol.yaml"
    pipeline = PDFAnalysisPipeline(protocol_file)

    # 1) Upload and vectorize PDF
    user_pdf = input("Enter path to PDF: ").strip()
    ok = pipeline.upload_and_vectorize_pdf(user_pdf)
    if not ok:
        print("🚨 upload_and_vectorize_pdf failed.", file=sys.stderr)
        sys.exit(1)

    # 2) Run query interactively
    user_query = input("Enter your query: ").strip()

    def progress_callback(step_num, question, analysis):
        print(f"\n--- CoT Step {step_num}: {question} ---")
        print(analysis)
        print("----------------------------\n")

    for stage, status, ctx in pipeline.run_query_with_progress(user_query, progress_callback):
        print(f"[{stage}] {status}")

    final = pipeline.context.get("final_answer")
    if final:
        print("\n🎯 Final Answer:")
        print(final)
