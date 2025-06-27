import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Smart Learning Assistant",
    page_icon="🧠",
    layout="centered"
)

# Main title
st.title("🧠 Smart Learning Assistant")
st.markdown("### Your AI-powered companion for smarter learning!")

# Project Introduction
st.markdown("""
**Smart Learning Assistant** is an intelligent study support system that includes two main features:

1. 📄 **Document Q&A (RAG)**  
   Ask questions and retrieve relevant information from learning materials. This feature uses Retrieval-Augmented Generation (RAG) to provide accurate answers with source references from your documents.

2. 🗺️ **Multi-Agent Learning Planner**  
   An intelligent agent system that helps you build a personalized learning roadmap. Just input your target topic, and the agents will break it down into smaller steps and suggest an optimized study plan.

""")

# Sidebar navigation
with st.sidebar:
    st.title("📚 Navigation")
    st.page_link("pages/1_RAG_QA.py", label="Document Q&A (RAG)", icon="📄")
    st.page_link("pages/2_Agent_Planner.py", label="Study Planner (Multi-Agent)", icon="🗺️")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='position: fixed; bottom: 50px; right: 15px; font-weight: bold;'>
        Made by Do Duc Duy - 2025
    </div>
    """,
    unsafe_allow_html=True
)
