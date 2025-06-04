# streamlit: name = 📄 Single PDF Mode
import streamlit as st
import os, tempfile, io, contextlib

from my_utils import (
    check_llm_connection, check_mongo_connection_pdfs, render_status
)
from one_paper_protocol import PDFAnalysisPipeline   # <-- новая версия pipeline

# ───────────────────────── Session helpers ─────────────────────────
def init_session():
    page_id = "PDF"
    defaults = {
        f"uploaded_file_{page_id}": None,
        f"processing_complete_{page_id}": False,
        f"log_buffer_{page_id}": io.StringIO(),    # храним вывод upload+query
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def append_log(text: str, page_id="PDF"):
    st.session_state[f"log_buffer_{page_id}"].write(text)


def progress_update(step_number, question, analysis, page_id="PDF"):
    """
    Показывает рассуждения Chain-of-Thought.
    Последний агрегирующий шаг («CoT Complete») выводим только в лог.
    """
    # не раскрываем финальный шаг
    if step_number == 0 and question == "CoT Complete":
        append_log(f"\n--- {question} ---\n{analysis}\n", page_id)
        return

    header = f"Step {step_number}: {question}"
    with st.expander(header):
        st.markdown(analysis)

    append_log(f"\n--- {header} ---\n{analysis}\n", page_id)



# ───────────────────────── Main App ─────────────────────────
def main():
    page_id = "PDF"
    init_session()

    st.set_page_config(page_title="PDF Analysis", layout="wide")
    st.title("📄 Single PDF Document Analysis")

    # ───────────── Sidebar ─────────────
    with st.sidebar:
        st.header("📝 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload one PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            key="pdf_upload"
        )
        st.session_state[f"uploaded_file_{page_id}"] = uploaded_file

        st.header("⚙️ Info")
        render_status("MongoDB", check_mongo_connection_pdfs())
        render_status("OpenAI API", check_llm_connection())

    # ───────────── Step 1: Process PDF ─────────────
    st.subheader("Step 1: Upload & Process PDF")
    st.markdown(
        "Upload a PDF file to begin. The system will automatically prepare the document for analysis by converting it into a readable format and indexing its content for fast search."    )

    if st.button("🗂 Process PDF"):
        pdf = st.session_state[f"uploaded_file_{page_id}"]
        if pdf is None:
            st.warning("Please upload exactly one PDF first.")
        else:
            # сохраняем во временный файл
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(pdf.getbuffer())
            tmp.flush()
            tmp_path = tmp.name
            tmp.close()

            pipeline = PDFAnalysisPipeline("one_paper_protocol.yaml")

            # перехватываем stdout, чтобы показать логи во время работы
            out_buf = st.session_state[f"log_buffer_{page_id}"] = io.StringIO()
            with contextlib.redirect_stdout(out_buf):
                with st.spinner("Uploading & vectorizing…"):
                    ok = pipeline.upload_and_vectorize_pdf(tmp_path)

            os.remove(tmp_path)

            if ok:
                st.success("PDF processed and indexed successfully!")
                st.session_state[f"processing_complete_{page_id}"] = True
            else:
                st.error("🚨 Processing failed – check logs below.")

    # ───────────── Step 2: Ask a question ─────────────
    st.subheader("Step 2: Ask a question about the document")
    if not st.session_state[f"processing_complete_{page_id}"]:
        st.info("Please complete **Step 1** to process the PDF before querying.")
        return

    query = st.text_input("Your question:", key="pdf_query")
    run_btn = st.button("🧠 Run Query", disabled=not query.strip())

    # контейнеры для этапов
    stage_placeholder = st.container()
    stage_widgets = {}
    STAGE_NAMES = {
        "retrieval": "Retrieving chunks",
        "reasoning": "Chain-of-Thought reasoning",
        "qa_over_pdf": "Answering",
        "pipeline_complete": "Done"
    }
    status_retr = st.empty()
    status_reasoning = st.empty()
    status_answering = st.empty()
    if run_btn:
        if not query.strip():
            st.warning("Please enter a query.")
            return

        processor = PDFAnalysisPipeline("one_paper_protocol.yaml")
        out_buf = st.session_state[f"log_buffer_{page_id}"]
        out_buf.seek(0);
        out_buf.truncate(0)

        try:
            with contextlib.redirect_stdout(out_buf):
                with st.spinner("Running query…"):
                    steps = processor.run_query_with_progress(
                        user_query=query,
                        progress_callback=lambda n, q, a: progress_update(n, q, a, page_id)
                    )

                    for stage, status, ctx in steps:
                        if stage == "retrieval":
                            if status == "in_progress":
                                status_retr.markdown("⏳ Retrieving relevant chunks…")
                            elif status == "done":
                                status_retr.markdown("✅ Relevant chunks retrieved")
                            elif status == "error":
                                status_retr.markdown("❌ Retrieval failed")

                        elif stage == "reasoning":
                            if status == "in_progress":
                                status_reasoning.markdown("⏳ Chain-of-Thought reasoning…")
                            elif status == "done":
                                status_reasoning.markdown("✅ Reasoning complete")
                            elif status == "error":
                                status_reasoning.markdown("❌ Reasoning failed")

                        elif stage == "qa_over_pdf":
                            if status == "in_progress":
                                status_answering.markdown("⏳ Generating answer from document…")
                            elif status == "done":
                                status_answering.markdown("✅ Final answer generated")
                            elif status == "error":
                                status_answering.markdown("❌ Answering failed")

        except Exception as e:
            st.error(f"🚨 Query failed: {e}")
        else:
            final_txt = (processor.context.get("final_answer")
                         or processor.context.get("qa_over_pdf_output"))

            if final_txt:
                st.divider()
                st.subheader("🎯 Final Answer")
                st.markdown(final_txt)


if __name__ == "__main__":
    main()
