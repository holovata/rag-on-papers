import streamlit as st

# Wide mode + custom CSS to force full width
st.set_page_config(page_title="NOCTUA — Welcome", page_icon="🦉", layout="wide")

# Custom CSS for full-width markdown
st.markdown("""
    <style>
    .block-container {
        padding: 2rem 5rem;
    }
    .stMarkdown {
        max-width: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦉 Welcome to NOCTUA")
st.subheader("Information Retrieval & Text Generation System")

st.markdown("""
**NOCTUA** is an intelligent assistant for processing and analyzing scientific articles and documents.

It leverages Retrieval-Augmented Generation (RAG) techniques to provide structured answers based on the content of uploaded PDFs, metadata from [arXiv.org](https://arxiv.org), and internal user documentation.

---

### 🔍 Main Features

1. **Search arXiv abstracts**  
   Ask general scientific questions — NOCTUA retrieves top-matching abstracts and generates a structured overview with links to full PDFs.
   
2. **Work with a single document**  
   Upload a PDF and ask any question — NOCTUA will search the most relevant parts of the file and generate a multi-step reasoned answer.

3. **Work with multiple documents**  
   Upload several papers — the system finds overlaps and contradictions between them and cites sources explicitly.

5. **Documentation QA**  
   You can ask NOCTUA anything about the platform itself — it will search the user guide and return helpful instructions.

---

### 💡 How it works

Each mode is driven by a **YAML pipeline** with clearly defined steps:
- **Retrieval** — find the most relevant chunks using cosine similarity.
- **Synthesis** — generate a structured answer based on selected fragments.
- **Clarification (optional)** — ask for more info from the user.
- **Reasoning** — perform multi-step analysis with source attribution.
- **Ideation** — propose research directions based on your topic.

---

### 🚀 Get Started

Use the sidebar to:
- 🧭 **Select the pipeline**
- 🛠️ **Enable refinement mode** *(Abstracts mode)*
- 📄 **Upload a document** *(Documents modes)*

And start asking questions 💬
""")
