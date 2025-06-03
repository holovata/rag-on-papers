# streamlit: name = 🆘 Help & Guide

import streamlit as st
import os
import io
import contextlib
import yaml
from help_protocol import UserGuidePipeline  # Класс обработки протокола из help_protocol.py

# Абсолютный путь к файлу руководства
MANUAL_PATH = r"C:\Work\diplom2\rag_on_papers\data\user_manual\user_manual.md"
# Путь к YAML-протоколу для QA (retrieval и qa_over_guide)
PROTOCOL_PATH = r"C:\Work\diplom2\rag_on_papers\help_protocol.yaml"


@st.cache_data(show_spinner=False)
def load_manual(manual_path: str) -> str:
    """Загружает содержимое Markdown-файла руководства."""
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading manual: {e}")
        return ""


def load_protocol_stages() -> list:
    """Извлекает список этапов из YAML-протокола."""
    try:
        with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
            protocol = yaml.safe_load(f)
        stages = [step["stage"] for step in protocol["pipeline"]]
        return stages
    except Exception as e:
        st.error(f"Error loading protocol: {e}")
        return []


# Словарь для отображения понятных пользователю надписей для этапов
STAGE_LABELS = {
    "query_refinement": "Query refinement",
    "retrieval": "Retrieval of relevant information",
    "qa_over_guide": "Final answer generation",
    "done": "Completed"
}


def initialize_pipeline():
    """Инициализирует объект UserGuidePipeline и сохраняет его в session_state."""
    if 'user_guide_pipeline' not in st.session_state:
        st.session_state.user_guide_pipeline = UserGuidePipeline(PROTOCOL_PATH)
    return st.session_state.user_guide_pipeline


def process_query_with_indicators(query: str) -> (str, str):
    """
    Processes the query using the pipeline and updates visual indicators for each stage.
    Intermediate results are not displayed, only stage statuses, while logs are saved.
    """
    if not query.strip():
        st.warning("Please enter a valid question")
        return "", ""

    pipeline = initialize_pipeline()
    stages = load_protocol_stages()

    # Create a container for stage indicators
    stage_container = st.container()
    indicators = {}
    for stage in stages:
        user_label = STAGE_LABELS.get(stage, stage)
        # Initially, all stages are marked as "Waiting..."
        indicators[stage] = stage_container.empty()
        indicators[stage].markdown(f"⏳ **{user_label}**: Waiting...")

    # Capture standard output for logs
    debug_buffer = io.StringIO()
    final_answer = ""

    with contextlib.redirect_stdout(debug_buffer):
        for stage, context in pipeline.run_pipeline(query):
            user_label = STAGE_LABELS.get(stage, stage)
            # Update indicator for the current stage
            if stage in indicators:
                # If the stage has just started, show "In progress..."
                indicators[stage].markdown(f"🚀 **{user_label}**: In progress...")
            # If this is the answer generation stage, save the final answer
            if stage == "qa_over_guide":
                if "answer_query_from_guide_output" in context:
                    final_answer = context["answer_query_from_guide_output"]
                elif "final_answer" in context:
                    final_answer = context["final_answer"]
            # After each step, immediately update the current stage status as "Completed"
            if stage in indicators:
                indicators[stage].markdown(f"✅ **{user_label}**: Completed")

    logs = debug_buffer.getvalue()
    st.session_state['protocol_output'] = logs
    return final_answer, logs


def main():
    st.set_page_config(
        page_title="Smart User Guide",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🆘 Help & Guide")
    st.markdown("Ask questions about our documentation using the AI-powered guide assistant.")

    # Sidebar: настройки и скачивание руководства
    with st.sidebar:
        st.header("📘 Manual Info")
        manual_content = load_manual(MANUAL_PATH)
        if manual_content:
            st.success("Manual loaded successfully")
        else:
            st.error("Manual file missing")
        st.download_button(
            label="Download Manual",
            data=manual_content,
            file_name="user_manual.md",
            mime="text/markdown"
        )
    # Основной контент: содержимое руководства спрятано в экспандере (по умолчанию свернуто)
    with st.expander("Show Manual", expanded=False):
        manual_content = load_manual(MANUAL_PATH)
        if manual_content:
            st.markdown(manual_content, unsafe_allow_html=True)
        else:
            st.warning("Manual content not available.")

    # Раздел для вопросов
    st.subheader("Ask a question about the guide")
    query = st.text_input("Your question:", placeholder="Type your question here...")
    if st.button("Ask"):
        final_answer, logs = process_query_with_indicators(query)
        st.markdown("### Final Answer:")
        st.write(final_answer)
        with st.expander("Show Debug Logs"):
            st.code(logs, language="log")


if __name__ == "__main__":
    main()
