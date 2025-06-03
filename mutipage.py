import streamlit as st

st.set_page_config(
    page_title="Research Assistant Pro",
    page_icon="🔬",
    layout="wide"
)

st.write("# 🔬 Research Assistant Pro")
st.sidebar.success("Select a mode from the sidebar")

st.markdown("""
    ## 🧠 Intelligent Scientific Document Analysis  

    **👈 Choose a mode from the sidebar** to get started:
    - **Research Assistant** – Full-cycle research support  
    - **PDF Analysis** – Advanced document processing pipeline  

    ## 🔎 Key Features:
    - 📑 **Deep PDF document analysis** powered by LangChain and Streamlit  
    - 🔍 **Intelligent query processing** with logical reasoning (CoT)  
    - 🧠 **Context-aware retrieval** using ChromaDB vector search  
    - 📊 **Multi-stage document analysis** for structured insights  
    - 🤖 **Ollama LLM integration** for AI-driven knowledge extraction  

    ## ⚙️ System Requirements:
    - ✅ **Ollama LLM server** running locally  
    - 📂 **ChromaDB vector database** for semantic search  
    - 🐍 **Python 3.9+ environment** with required dependencies  

    🚀 *Unlock smarter document analysis and research automation today!*
""")
