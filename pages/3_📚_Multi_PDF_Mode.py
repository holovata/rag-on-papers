# streamlit: name = 📚 Multi-PDF Mode

import streamlit as st
import os
import io
import contextlib

from pathlib import Path

from src.data_processing.papers_vectorization_MONGO import (
    ingest_pdfs_to_markdowns,
    ingest_multi_chunks,
    build_vector_index,
    clear_all_before_ingest,
)
from src.retrieval.papers_retrieval_MONGO import get_relevant_chunks
from n_papers_protocol import MultiPDFAnalysisPipeline

from my_utils import save_uploaded_file, check_chroma_connection, clear_folder

def init_session():
    page_id = "MultiPDF"
    session_defaults = {
        f"uploaded_files_{page_id}": [],
        f"query_{page_id}": "",
        f"pipeline_instance_{page_id}": None,
        f"protocol_output_{page_id}": ""
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_stage_indicator(
    stage, status, data,
    container, indicators, labels,
    cot_container=None, cot_indicators=None
):
    """Обновляет статус каждой стадии, включая CoT-подшаги."""
    stage_label = labels.get(stage, stage)
    if stage not in indicators:
        indicators[stage] = container.empty()
        indicators[stage].markdown(f"⏳ **{stage_label}**: Waiting...")
    if status == "in_progress":
        if stage == "reasoning" and isinstance(data, dict) and "reasoning_output" in data:
            # после завершения reasoning, показываем каждый шаг CoT
            reasoning_out = data["reasoning_output"]
            for step_info in reasoning_out.get("reasoning_steps", []):
                step_label = step_info["step"]
                analysis = step_info["analysis"]
                if step_label not in cot_indicators and cot_container is not None:
                    cot_indicators[step_label] = cot_container.expander(f"📝 {step_label}")
                    cot_indicators[step_label].markdown(f"{analysis}")
            indicators[stage].markdown(f"🚀 **{stage_label}**: In progress...")
        else:
            indicators[stage].markdown(f"🚀 **{stage_label}**: In progress...")
    elif status == "done":
        indicators[stage].markdown(f"✅ **{stage_label}**: Completed")
    elif status == "error":
        indicators[stage].markdown(f"❌ **{stage_label}**: Error")

def main():
    page_id = "MultiPDF"
    init_session()
    st.set_page_config(page_title="Multi-PDF Analysis", layout="wide")
    st.title("📚 Multi-PDF Document Analysis")

    # Sidebar: загрузка и проверка подключений
    with st.sidebar:
        st.header("⚙️ Document Settings")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            help="Upload one or more PDFs for analysis",
            accept_multiple_files=True
        )
        st.session_state[f"uploaded_files_{page_id}"] = uploaded_files

        persist_dir = r"C:\Work\diplom2\rag_on_papers\data\n_papers\embeddings"
        connection_status = check_chroma_connection(persist_dir)
        st.markdown(f"**Embedding Database Status:** {connection_status}")

    # Фиксированные пути
    target_folder = r"C:\Work\diplom2\rag_on_papers\data\n_papers"
    embeddings_folder = os.path.join(target_folder, "embeddings")

    st.subheader("Step 1: Document Processing")
    st.markdown(
        "Click the button below to prepare your PDF files: "
        "convert them to Markdown, remove tables, save them to GridFS, "
        "split into chunks, and build the vector index."
    )

    if st.button("Process Documents"):
        if not st.session_state[f"uploaded_files_{page_id}"]:
            st.warning("Please upload at least one PDF file before processing.")
        else:
            # Сохраняем загруженные файлы в папку target_folder
            for uploaded_file in st.session_state[f"uploaded_files_{page_id}"]:
                file_path = os.path.join(target_folder, uploaded_file.name)
                if not save_uploaded_file(uploaded_file, file_path):
                    st.error(f"Failed to save {uploaded_file.name}.")
            # Сохраняем список имён, чтобы не удалять их при очистке
            preserve = ",".join([f.name for f in st.session_state[f"uploaded_files_{page_id}"]])
            clear_folder(target_folder, preserve_files=preserve)
            # Очищаем старые embeddings
            clear_folder(embeddings_folder)

            # Запускаем ingestion + indexing через MultiPDFAnalysisPipeline
            processor = MultiPDFAnalysisPipeline("n_papers_protocol.yaml")
            # Сохраняем экземпляр pipeline в session_state, чтобы переиспользовать его для анализа
            st.session_state[f"pipeline_instance_{page_id}"] = processor

            with st.spinner("Processing documents..."):
                ingest_result = processor.run_ingest_and_index(pdf_folder=target_folder)
            st.success(f"Document processing completed! Inserted {ingest_result.get('multi_chunk_count', 0)} chunks.")

    if st.session_state.get(f"pipeline_instance_{page_id}") is None:
        st.subheader("Step 2: Document Analysis")
        st.info("Please complete Step 1 to process the PDFs before querying.")
        return

    st.subheader("Step 2: Document Analysis")
    st.markdown("Enter a query to analyze the documents (after you have clicked “Process Documents”).")
    st.session_state[f"query_{page_id}"] = st.text_input(
        "Enter your query:", key="multi_pdf_query"
    )

    if st.button("Analyze Documents"):
        if not st.session_state[f"query_{page_id}"].strip():
            st.warning("Please enter a query.")
        else:
            processor = st.session_state.get(f"pipeline_instance_{page_id}")
            if processor is None:
                st.error("Сначала нажмите «Process Documents», чтобы загрузить и проиндексировать PDF.")
            else:
                # Контейнеры для статусов стадий и CoT
                stage_container = st.container()
                cot_container = st.container()
                stage_indicators = {}
                cot_indicators = {}
                STAGE_LABELS = {
                    "ingest_and_index": "Ingest & Index",
                    "retrieval": "Retrieving Relevant Chunks",
                    "reasoning": "Chain-of-Thought Analysis",
                    "qa_over_pdf": "Generating Final Answer",
                    "synthesis": "Synthesis",
                    "pipeline_complete": "Pipeline Complete"
                }

                # Логирование stdout
                log_stream = io.StringIO()
                try:
                    with st.spinner("Analyzing documents..."):
                        with contextlib.redirect_stdout(log_stream):
                            events = processor.run_pipeline_with_progress(
                                user_query=st.session_state[f"query_{page_id}"],
                                pdf_folder=target_folder,
                                progress_callback=lambda stage, status, data: update_stage_indicator(
                                    stage, status, data,
                                    stage_container, stage_indicators, STAGE_LABELS,
                                    cot_container, cot_indicators
                                )
                            )
                            # Перебираем все события, чтобы pipeline полностью завершился
                            for _ in events:
                                pass

                    # Сохраняем логи в session_state
                    st.session_state[f"protocol_output_{page_id}"] = log_stream.getvalue()
                    st.divider()

                    # Выводим финальный результат reasoning и QA
                    result = processor.context

                    # Если были CoT-шаги, показываем их под экспандерами
                    reasoning_out = result.get("reasoning_output", {})
                    if reasoning_out:
                        for step_info in reasoning_out.get("reasoning_steps", []):
                            with st.expander(f"📝 {step_info['step']}"):
                                st.markdown(step_info["analysis"])

                    # Показываем итоговый ответ
                    st.subheader("🎯 Final Answer")
                    final_synthesis = result.get("synthesis_output", "")
                    if final_synthesis:
                        st.markdown(final_synthesis)
                    else:
                        qa_out = result.get("qa_over_pdf_output", "")
                        if qa_out:
                            st.markdown(qa_out)
                        else:
                            st.info("No answer produced.")

                except Exception as e:
                    st.error(f"🚨 Document analysis failed: {str(e)}")
                finally:
                    st.divider()
                    with st.expander("📜 Execution Logs"):
                        logs = st.session_state.get(f"protocol_output_{page_id}", "")
                        if logs.strip():
                            st.code(logs, language="log")
                        else:
                            st.info("No logs available.")

if __name__ == "__main__":
    main()
