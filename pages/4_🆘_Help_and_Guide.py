# streamlit: name = 🆘 Help & Guide
import streamlit as st
import os
import io
import yaml
from pathlib import Path

from help_protocol import UserGuidePipeline
from my_utils import check_llm_connection, check_mongo_connection, render_status

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parents[1]

# Paths to Markdown manual and protocol
MANUAL_PATH    = BASE_DIR / "data" / "user_manual" / "user_manual.md"
PROTOCOL_PATH  = BASE_DIR / "help_protocol.yaml"

# -------------------- Helper Functions -------------------- #
@st.cache_data(show_spinner=False)
def load_manual(md_path: Path) -> str:
    """
    Loads the manual markdown content as plain text.
    """
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Error loading manual: {e}")
        return ""

@st.cache_data(show_spinner=False)
def load_protocol_stages() -> list[str]:
    """
    Reads the YAML protocol and returns all stage names.
    """
    try:
        with open(PROTOCOL_PATH, encoding="utf-8") as f:
            proto = yaml.safe_load(f)
        return [s["stage"] for s in proto["pipeline"]]
    except Exception as e:
        st.error(f"Error loading protocol: {e}")
        return []

STAGE_LABELS = {
    "retrieval":     "Retrieval of relevant information",
    "qa_over_guide": "Final answer generation",
    "done":          "Completed",
}

def initialize_pipeline() -> UserGuidePipeline:
    """
    Initializes the UserGuidePipeline instance stored in session state.
    """
    if "user_guide_pipeline" not in st.session_state:
        st.session_state.user_guide_pipeline = UserGuidePipeline(PROTOCOL_PATH)
    return st.session_state.user_guide_pipeline

def process_query_with_stream(query: str):
    """
    Executes the guide-question pipeline with streaming LLM output.
    """
    if not query.strip():
        st.warning("Please enter a valid question")
        return

    pipeline = initialize_pipeline()
    stages = load_protocol_stages()

    # Create progress indicators
    stage_container = st.container()
    indicators = {s: stage_container.empty() for s in stages}
    for s, box in indicators.items():
        box.markdown(f"⏳ **{STAGE_LABELS.get(s, s)}**: Waiting...")

    final_answer = ""
    answer_container = st.container()

    for stage, ctx in pipeline.run_pipeline(query):
        lbl = STAGE_LABELS.get(stage, stage)
        if stage in indicators:
            indicators[stage].markdown(f"🚀 **{lbl}**: In progress...")

        if stage == "qa_over_guide" and "streamed_response" in ctx:
            # Stream and display the model's output
            answer_container.empty()
            acc_text = ""
            text_display = answer_container.empty()

            for chunk in ctx["streamed_response"]:
                acc_text += chunk.content
                text_display.text(acc_text)

            final_answer = acc_text

        if stage in indicators:
            indicators[stage].markdown(f"✅ **{lbl}**: Completed")

    return final_answer

# -------------------- Streamlit Page -------------------- #
st.set_page_config(page_title="Smart User Guide", page_icon="📚", layout="wide")

st.title("🆘 Help & Guide")
st.markdown("Ask questions about our documentation using the AI-powered guide assistant.")

# Sidebar information
with st.sidebar:
    st.header("⚙️ Info")
    render_status("MongoDB", check_mongo_connection())      # 🟢 / 🔴
    render_status("OpenAI API", check_llm_connection())     # 🟢 / 🔴

    st.header("📘 Manual Info")
    try:
        md_bytes = MANUAL_PATH.read_bytes()
        st.download_button(
            "📥 Download Manual",
            data=md_bytes,
            file_name="user_manual.md",
            mime="text/markdown"
        )
        st.success("Manual loaded successfully")
    except Exception as e:
        st.error(f"Manual file not found: {e}")

# Manual preview (Markdown only)
with st.expander("Show Manual", expanded=False):
    manual_text = load_manual(MANUAL_PATH)
    st.markdown(manual_text, unsafe_allow_html=True)

# User query input
st.subheader("Ask a question about the guide")
user_q = st.text_input("Your question:", placeholder="Type your question here…")

if st.button("❓ Ask"):
    process_query_with_stream(user_q)
