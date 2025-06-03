# streamlit: name = 📄 Single PDF Mode

import streamlit as st
import os
import tempfile
from pathlib import Path

from my_utils import process_pdf_via_console, check_llm_connection, check_mongo_connection_pdfs, render_status

from src.data_processing.papers_vectorization_MONGO import ingest_from_gridfs, build_vector_index
from one_paper_protocol import PDFAnalysisPipeline


def init_session():
    page_id = "PDF"
    session_defaults = {
        f'uploaded_file_{page_id}': None,
        f'query_{page_id}': "",
        f'processing_complete_{page_id}': False,
        f'protocol_output_{page_id}': ""
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Показываем логи CoT-этапов
def progress_update(step_number, question, analysis):
    if step_number == 0 and question == "CoT Complete":
        st.success("Chain-of-Thought reasoning completed.")
    else:
        with st.expander(f"Step {step_number}: {question}"):
            st.markdown(analysis)

def main():
    page_id = "PDF"
    init_session()
    st.set_page_config(page_title="PDF Analysis", layout="wide")
    st.title("📄 Single PDF Document Analysis")

    # Sidebar: загрузка PDF
    with st.sidebar:
        st.header("📝 Document Upload")
        uploaded_file = st.file_uploader("Upload one PDF file", type=["pdf"], accept_multiple_files=False)
        st.session_state[f'uploaded_file_{page_id}'] = uploaded_file

        st.header("⚙️ Info")
        render_status("MongoDB", check_mongo_connection_pdfs())
        render_status("OpenAI API", check_llm_connection())

    # Шаг 1: Сохранить PDF в GridFS, конвертировать, векторизовать и индексировать
    st.subheader("Step 1: Upload & Process PDF")
    st.markdown("Upload a PDF, and it will be automatically saved to MongoDB (GridFS), "
                "converted to Markdown, split into chunks, embedded, and the vector index will be updated.")

    if st.button("🗂 Process PDF"):
        if not st.session_state[f'uploaded_file_{page_id}']:
            st.warning("Please upload a PDF file first.")
        else:
            # Сохраняем загруженный файл во временный локальный путь
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            try:
                tmp.write(st.session_state[f'uploaded_file_{page_id}'].getbuffer())
                tmp.flush()
                tmp_path = tmp.name
            finally:
                tmp.close()

            st.session_state[f'processing_complete_{page_id}'] = False
            with st.spinner("Processing PDF—saving to GridFS, converting, ingesting…"):
                try:
                    # 1) Сохраняем PDF → GridFS, генерируем MD и чистый MD → GridFS
                    process_pdf_via_console(tmp_path)

                    # 2) Извлекаем clean.md из GridFS, разбиваем на чанки, сохраняем в paper_chunks
                    client = ingest_from_gridfs.__globals__['mongo_client']()
                    db = client["arxiv_db"]
                    ingest_from_gridfs(db)

                    # 3) Обновляем векторный индекс в paper_chunks
                    build_vector_index(db, "paper_chunks", "chunk_embedding_index")
                    client.close()

                    st.success("PDF processed and indexed successfully!")
                    st.session_state[f'processing_complete_{page_id}'] = True

                except Exception as e:
                    st.error(f"🚨 Failed to process PDF: {e}")
                finally:
                    # Удаляем временный файл
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    # Шаг 2: Анализ документа (после успешной обработки)
    st.subheader("Step 2: Analyze Document")
    if not st.session_state[f'processing_complete_{page_id}']:
        st.info("Please complete Step 1 to process the PDF before querying.")
        return

    st.markdown("Enter your query and click “Analyze Document” to get an answer based on the processed document.")
    st.session_state[f'query_{page_id}'] = st.text_input("Enter your document query:", key="pdf_query")

    stage_container = st.container()
    stage_indicators = {}
    STAGE_LABELS = {
        "convert_pdf_to_md":            "Load Clean Markdown",
        "remove_markdown_tables_wrapper":"Skip Table Removal",
        "chunk_and_vectorize":          "Chunk & Vectorize",
        "retrieval":                    "Retrieving Relevant Chunks",
        "reasoning":                    "Chain-of-Thought Reasoning",
        "qa_over_pdf":                  "Answering Based on Document",
        "create_index_chunks":          "Indexing Chunks",
        "pipeline_complete":            "Pipeline Complete"
    }

    if st.button("🧠 Run Query"):
        query = st.session_state[f'query_{page_id}'].strip()
        if not query:
            st.warning("Please enter a query.")
            return

        processor = PDFAnalysisPipeline("one_paper_protocol.yaml")
        try:
            with st.spinner("Analyzing document..."):
                pipeline_steps = processor.run_pipeline_with_progress(
                    user_query=query,
                    user_pdf="unused.pdf",  # PDF уже в GridFS; путь не важен
                    progress_callback=progress_update
                )

                for stage, status, context in pipeline_steps:
                    label = STAGE_LABELS.get(stage, stage)
                    if stage not in stage_indicators:
                        stage_indicators[stage] = stage_container.empty()
                        stage_indicators[stage].markdown(f"⏳ **{label}**: Waiting...")
                    if status == "in_progress":
                        stage_indicators[stage].markdown(f"🚀 **{label}**: In progress...")
                    elif status == "done":
                        stage_indicators[stage].markdown(f"✅ **{label}**: Completed")
                    elif status == "error":
                        stage_indicators[stage].markdown(f"❌ **{label}**: Error")

            st.divider()
            final_answer = processor.context.get("final_answer")
            if final_answer:
                st.subheader("🎯 Final Answer")
                st.markdown(final_answer)

        except Exception as e:
            st.error(f"🚨 Document analysis failed: {e}")

        finally:
            st.divider()
            with st.expander("📜 Execution Logs"):
                logs = st.session_state.get(f'protocol_output_{page_id}', "")
                if logs.strip():
                    st.code(logs, language="log")
                else:
                    st.info("No logs available.")

if __name__ == "__main__":
    main()
