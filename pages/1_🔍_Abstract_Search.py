# streamlit: name = 🔍 Abstract Search

import io
import contextlib
import yaml
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from abstracts_protocol import (
    get_top_relevant_articles,
    generate_clarification,
    generate_ideas,
)
from my_utils import check_llm_connection, check_mongo_connection, render_status

PROTOCOL_PATH = "abstracts_protocol.yaml"


# ────────────────────────── Helpers ──────────────────────────
@st.cache_data(show_spinner=False)
def load_pipeline() -> list[dict]:
    """Load pipeline definition only once."""
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["pipeline"]


PIPELINE = load_pipeline()


def get_prompt(stage: str) -> str | None:
    """Return the prompt_template for a given stage."""
    for step in PIPELINE:
        if step["stage"] == stage:
            return step.get("prompt_template")
    return None


def format_articles(arts: list[dict]) -> str:
    """Nicely format articles for the LLM prompt."""
    return "\n".join(
        f"{i+1}. {a['title']} (authors: {a.get('authors','N/A')})\n"
        f"   Abstract: {a['abstract'][:100]}...\n"
        f"   PDF: {a['pdf_url']}"
        for i, a in enumerate(arts)
    )



# ──────────────────────── Page layout  ───────────────────────
st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("🔍 Abstract Search")

with st.sidebar:
    st.header("⚙️ Info and Settings")
    render_status("MongoDB", check_mongo_connection())
    render_status("OpenAI API", check_llm_connection())
    enable_refinement = st.checkbox("Enable refinement mode", value=False)

# ───── Reset on refinement toggle ─────
if "prev_enable_refinement" not in st.session_state:
    st.session_state.prev_enable_refinement = enable_refinement

if st.session_state.prev_enable_refinement != enable_refinement:
    # Очистити всі пов’язані поля та скинути інтерфейс
    for key in [
        "initial_done", "initial_context",
        "refined_done", "refined_context",
        "is_generating", "is_refining",
        "log_initial", "log_refined",
        "query", "extra"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.prev_enable_refinement = enable_refinement
    st.rerun()


query = st.text_input(
    "Enter your research query:",
    key="query",
    placeholder="e.g. What is sparsity in graph decompositions?"
)


# ─────────────────── Session-state defaults ──────────────────
for key, default in {
    "initial_done": False,
    "initial_context": {},
    "refined_done": False,
    "refined_context": {},
    # button locks
    "is_generating": False,
    "is_refining": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# ═════════════════ 1 ) INITIAL RUN ═════════════════

# Run-Query button FIRST, disabled while generating
run_btn = st.button(
    "🧠 Run Query",
    key="btn_initial",
    disabled=st.session_state.is_generating,
)

# Progress placeholders
status_retr = st.empty()
status_synth = st.empty()
status_idea = st.empty()
answer_container = st.empty()

if run_btn:
    if not query.strip():
        st.warning("Please enter a query before proceeding.")
        st.stop()

    # lock the button
    st.session_state.is_generating = True

    # reset session output
    st.session_state.initial_done = False
    st.session_state.refined_done = False
    st.session_state.initial_context = {}
    st.session_state.refined_context = {}

    log_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
            # ── Retrieval ──
            status_retr.markdown("⏳ Retrieving relevant articles…")
            arts = get_top_relevant_articles(query=query, top_n=7, min_similarity=0.6)
            status_retr.markdown(f"✅ Retrieved {len(arts)} articles")

            # ── First synthesis ──
            status_synth.markdown("⏳ Generating report…")
            prompt = PromptTemplate.from_template(get_prompt("first_synthesis")).format(
                query=query,
                num_articles=len(arts),
                articles_data=format_articles(arts),
            )

            first_out = ""
            for chunk in llm.stream([("user", prompt)]):
                first_out += chunk.content
                answer_container.markdown(first_out)
            status_synth.markdown("✅ Report generation complete")

            # ── Ideas ──
            status_idea.markdown("⏳ Generating research ideas…")
            ideas = generate_ideas(arts, query, get_prompt("ideation"))
            status_idea.markdown("✅ Research ideas generated")

            # ── Clarification ──
            clar_questions = []
            if enable_refinement:
                clar = generate_clarification(
                    arts, first_out, query, get_prompt("clarification")
                )
                clar_questions = clar.get("questions", [])
    finally:
        st.session_state.is_generating = False  # always unlock

    # save
    st.session_state.initial_context = {
        "arts": arts,
        "first_out": first_out,
        "clar_questions": clar_questions,
        "ideas": ideas,
    }
    st.session_state.initial_done = True
    st.session_state.log_initial = log_buf.getvalue()

# ─── Display initial extras ───
if st.session_state.initial_done:
    if enable_refinement and st.session_state.initial_context["clar_questions"]:
        with st.expander("📝 Clarification Questions"):
            # show from the second question onward
            for q in st.session_state.initial_context["clar_questions"][1:]:
                st.markdown(f"- {q}")

    with st.expander("💡 Research Ideas"):
        for idea in st.session_state.initial_context["ideas"]["ideas"]:
            st.markdown(f"- {idea}")

    # with st.expander("📜 Initial Logs"):
    #     st.code(st.session_state.log_initial, language="log")

# ═════════════════ 2 ) REFINED RUN ═════════════════
if enable_refinement and st.session_state.initial_done:
    st.markdown("### 2️⃣ Refined Answer")

    extra = st.text_input(
        "Your clarification (answer to the questions above):",
        key="extra",
    )

    refine_btn = st.button(
        "🔄 Get Refined Answer",
        key="btn_refined",
        disabled=st.session_state.is_refining,
    )

    status_rretr = st.empty()
    status_rsynth = st.empty()
    status_ridea = st.empty()
    refined_container = st.empty()

    if refine_btn:
        if not extra.strip():
            st.warning("Please provide your clarification before refining.")
            st.stop()

        st.session_state.is_refining = True
        log_buf2 = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_buf2), contextlib.redirect_stderr(
                log_buf2
            ):
                # ── Refined retrieval ──
                status_rretr.markdown("⏳ Retrieving with refined query…")
                refined_query = f"{query} {extra}"
                r_arts = get_top_relevant_articles(
                    query=refined_query, top_n=7, min_similarity=0.6
                )
                status_rretr.markdown(f"✅ Retrieved {len(r_arts)} articles")

                # merge & deduplicate
                seen: set[str] = set()
                all_arts: list[dict] = []
                for art in st.session_state.initial_context["arts"] + r_arts:
                    if art["id"] not in seen:
                        seen.add(art["id"])
                        all_arts.append(art)

                # ── Refined synthesis ──
                status_rsynth.markdown("⏳ Generating refined report…")
                prompt2 = PromptTemplate.from_template(
                    get_prompt("second_synthesis")
                ).format(
                    first_query=query,
                    extra=extra,
                    first_synthesis_output=st.session_state.initial_context["first_out"],
                    num_articles=len(all_arts),
                    articles_data=format_articles(all_arts),
                )

                second_out = ""
                for chunk in llm.stream([("user", prompt2)]):
                    second_out += chunk.content
                    refined_container.markdown(second_out)
                status_rsynth.markdown("✅ Refined report generation complete")

                # ── Refined ideas ──
                status_ridea.markdown("⏳ Generating refined research ideas…")
                ideas2 = generate_ideas(all_arts, refined_query, get_prompt("ideation"))
                status_ridea.markdown("✅ Refined research ideas generated")
        finally:
            st.session_state.is_refining = False

        # save
        st.session_state.refined_context = {
            "all_arts": all_arts,
            "second_out": second_out,
            "ideas2": ideas2,
        }
        st.session_state.refined_done = True
        st.session_state.log_refined = log_buf2.getvalue()

    # show refined extras
    if st.session_state.refined_done:
        with st.expander("💡 Refined Research Ideas"):
            for idea in st.session_state.refined_context["ideas2"]["ideas"]:
                st.markdown(f"- {idea}")

        # with st.expander("📜 Refined Logs"):
        #     st.code(st.session_state.log_refined, language="log")
