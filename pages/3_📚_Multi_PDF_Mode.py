# streamlit: name = 📚 Multi-PDF Mode

import streamlit as st
import os
from pathlib import Path

from src.data_processing.papers_vectorization_MONGO import (
    clear_all_before_ingest,
    clear_collection,
    ingest_pdfs_to_markdowns,
    ingest_multi_chunks,
    build_vector_index,
)
from n_papers_protocol import MultiPDFAnalysisPipeline

from my_utils import (
    save_uploaded_file,
    clear_folder,
    check_mongo_connection_pdfs,
    check_llm_connection,
    render_status,
)

# ───────────────────────── Session helpers ─────────────────────────
def init_session():
    page_id = "MultiPDF"
    defaults = {
        f"uploaded_files_{page_id}": [],
        f"query_{page_id}": "",
        f"pipeline_instance_{page_id}": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ───────────────────────── Main App ─────────────────────────
def main():
    try:
        page_id = "MultiPDF"
        init_session()
        st.set_page_config(page_title="Multi-PDF Analysis", layout="wide")
        st.title("📚 Multi-PDF Document Analysis")

        # ───────────────────── Sidebar ─────────────────────
        with st.sidebar:
            st.header("⚙️ Document Settings")
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=["pdf"],
                help="Upload one or more PDFs for analysis",
                accept_multiple_files=True,
            )
            st.session_state[f"uploaded_files_{page_id}"] = uploaded_files

            st.header("⚙️ Info")
            render_status("MongoDB", check_mongo_connection_pdfs())
            render_status("OpenAI API", check_llm_connection())

        # Фиксированные пути
        BASE_DIR = Path(__file__).resolve().parents[1]
        target_folder = BASE_DIR / "data" / "n_papers"
        embeddings_folder = os.path.join(target_folder, "embeddings")

        st.subheader("Step 1: Document Processing")
        st.markdown(
            "Click the button below to prepare your PDF files: convert them to Markdown, "
    "remove tables, save them to GridFS, split into chunks, and build the vector index."
        )

        if st.button("Process Documents"):
            if not st.session_state[f"uploaded_files_{page_id}"]:
                st.warning("Please upload at least one PDF file before processing.")
            else:
                # 1) Сохраняем в папку target_folder
                for uploaded_file in st.session_state[f"uploaded_files_{page_id}"]:
                    file_path = os.path.join(target_folder, uploaded_file.name)
                    if not save_uploaded_file(uploaded_file, file_path):
                        st.error(f"Could not save {uploaded_file.name}.")

                # 2) Очистка (оставляем только загруженные PDF)
                preserve = ",".join(
                    [f.name for f in st.session_state[f"uploaded_files_{page_id}"]]
                )
                clear_folder(str(target_folder), preserve_files=preserve)
                clear_folder(embeddings_folder)

                # 3) Создаём экземпляр pipeline и сохраняем в сессии
                processor = MultiPDFAnalysisPipeline("n_papers_protocol.yaml")
                st.session_state[f"pipeline_instance_{page_id}"] = processor

                # 4) Отображаем прогресс-бар
                progress_bar = st.progress(0)
                status_text  = st.empty()

                # — 1: Clear
                status_text.text("Clearing previous data…")
                clear_collection(processor.db_buffer, "pdf_markdowns")
                clear_all_before_ingest(processor.db_buffer, bucket="pipeline_fs")
                clear_collection(processor.db_arxiv, "multi_paper_chunks")
                progress_bar.progress(20)

                # — 2: Ingest PDF → GridFS
                status_text.text("Ingesting PDFs into GridFS…")
                ingest_pdfs_to_markdowns(processor.db_buffer, Path(target_folder))
                progress_bar.progress(50)

                # — 3: Chunk + Vectorize
                status_text.text("Chunking and vectorizing Markdown…")
                ingest_multi_chunks(processor.db_arxiv, processor.db_buffer)
                progress_bar.progress(70)

                # — 4: Build Vector Index
                status_text.text("Building vector index for multi_paper_chunks…")
                build_vector_index(
                    processor.db_arxiv,
                    "multi_paper_chunks",
                    "multi_chunk_embedding_index",
                )
                progress_bar.progress(100)

                status_text.text("Document processing completed.")
                st.success(
                    f"Processing done! Inserted {processor.db_arxiv['multi_paper_chunks'].count_documents({})} chunks."
                )

        if st.session_state.get(f"pipeline_instance_{page_id}") is None:
            st.subheader("Step 2: Document Analysis")
            st.info("Please complete **Step 1** to process the PDFs before querying.")
            return

        st.subheader("Step 2: Document Analysis")
        st.markdown("Enter your query after clicking “Process Documents”.")

        st.session_state[f"query_{page_id}"] = st.text_input(
            "Enter your query:", key="multi_pdf_query"
        )

        if st.button("Analyze Documents"):
            if not st.session_state[f"query_{page_id}"].strip():
                st.warning("Please enter a query.")
            else:
                processor: MultiPDFAnalysisPipeline = st.session_state.get(
                    f"pipeline_instance_{page_id}"
                )
                if processor is None:
                    st.error("Please complete **Step 1** first.")
                else:
                    stage_container  = st.container()
                    stage_indicators = {}
                    STAGE_LABELS = {
                        "retrieval":        "Retrieving Relevant Chunks",
                        "reasoning":        "Chain-of-Thought Analysis",
                        "qa_over_pdf":      "Generating Final Answer",
                        "synthesis":        "Synthesis",
                        "pipeline_complete": "Pipeline Complete",
                    }

                    # — Retrieval
                    if "retrieval" not in stage_indicators:
                        stage_indicators["retrieval"] = stage_container.empty()
                        stage_indicators["retrieval"].markdown(
                            f"⏳ **{STAGE_LABELS['retrieval']}**: Waiting…"
                        )
                    stage_indicators["retrieval"].markdown(
                        f"🚀 **{STAGE_LABELS['retrieval']}**: In progress…"
                    )
                    top_chunks = processor.run_retrieval(
                        st.session_state[f"query_{page_id}"], top_n=5
                    )
                    stage_indicators["retrieval"].markdown(
                        f"✅ **{STAGE_LABELS['retrieval']}**: Completed"
                    )

                    # — Reasoning (пошаговый CoT)
                    if "reasoning" not in stage_indicators:
                        stage_indicators["reasoning"] = stage_container.empty()
                        stage_indicators["reasoning"].markdown(
                            f"⏳ **{STAGE_LABELS['reasoning']}**: Waiting…"
                        )
                    stage_indicators["reasoning"].markdown(
                        f"🚀 **{STAGE_LABELS['reasoning']}**: In progress…"
                    )

                    reasoning_iter = processor.run_reasoning_step_by_step(
                        st.session_state[f"query_{page_id}"], top_chunks
                    )

                    final_reasoning = ""
                    for iteration_number, question_text, step_content in reasoning_iter:
                        # Пропускаем “CoT Complete” шаг (его не выводим)
                        if question_text == "CoT Complete":
                            continue

                        header = f"Step {iteration_number}: {question_text}"
                        with st.expander(header):
                            st.markdown(step_content)
                        final_reasoning = step_content

                    stage_indicators["reasoning"].markdown(
                        f"✅ **{STAGE_LABELS['reasoning']}**: Completed"
                    )

                    # — QA_over_pdf
                    if "qa_over_pdf" not in stage_indicators:
                        stage_indicators["qa_over_pdf"] = stage_container.empty()
                        stage_indicators["qa_over_pdf"].markdown(
                            f"⏳ **{STAGE_LABELS['qa_over_pdf']}**: Waiting…"
                        )
                    stage_indicators["qa_over_pdf"].markdown(
                        f"🚀 **{STAGE_LABELS['qa_over_pdf']}**: In progress…"
                    )
                    qa_out = processor.run_qa_over_pdf(
                        st.session_state[f"query_{page_id}"], top_chunks
                    )
                    stage_indicators["qa_over_pdf"].markdown(
                        f"✅ **{STAGE_LABELS['qa_over_pdf']}**: Completed"
                    )

                    # — Synthesis
                    if "synthesis" not in stage_indicators:
                        stage_indicators["synthesis"] = stage_container.empty()
                        stage_indicators["synthesis"].markdown(
                            f"⏳ **{STAGE_LABELS['synthesis']}**: Waiting…"
                        )
                    stage_indicators["synthesis"].markdown(
                        f"🚀 **{STAGE_LABELS['synthesis']}**: In progress…"
                    )
                    synth_out = processor.run_synthesis(
                        {"final_answer": final_reasoning}, qa_out
                    )
                    stage_indicators["synthesis"].markdown(
                        f"✅ **{STAGE_LABELS['synthesis']}**: Completed"
                    )

                    # — Финальный индикатор
                    if "pipeline_complete" not in stage_indicators:
                        stage_indicators["pipeline_complete"] = stage_container.empty()
                    stage_indicators["pipeline_complete"].markdown(
                        f"✅ **{STAGE_LABELS['pipeline_complete']}**"
                    )

                    # — Stream Final Answer (Reasoning)
                    st.subheader("🎯 Final Answer")
                    container_stream = st.empty()
                    query_text = st.session_state[f"query_{page_id}"]

                    accumulated = ""
                    for token in processor.stream_final_reasoning(query_text, top_chunks):
                        accumulated += token
                        container_stream.markdown(accumulated)

    except Exception as err:
        st.exception(err)  # Полный traceback в браузере
        raise

if __name__ == "__main__":
    main()
