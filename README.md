# 📚 Smart Learning Assistant

**An AI-powered tool designed to support self-directed learning through document-based Q&A and personalized learning path planning.**

## 🔍 Overview

Smart Learning Assistant is an AI solution designed to empower learners in navigating complex topics independently. This tool combines Retrieval-Augmented Generation (RAG) techniques with an agentic multi-agent architecture to deliver two core functionalities:

1. **RAG Bot**: Enables accurate document-based question answering using hybrid search on uploaded PDFs, DOCX files, and external URLs.
2. **Agentic System**: Uses intelligent agents to discover online resources, plan learning paths, and provide step-by-step study roadmaps based on user's topic.


## ✨ Key Features
- **🔍 Hybrid RAG Retrieval**: 
	- Multi-format document support, such as PDFs, DOCX files, and web links.
	- Combines semantic and keyword-based (keyword + vector)  search to enhance answer accuracy.
- **🤖 Multi-Agent System**: Intelligent agents collaborate to:
  - Search for relevant topics and resources.
  - Build personalized study plans.
- **🖥️ User Interface**: Interactive Streamlit frontend for ease of use and real-time interaction.


## 🚀 Demonstration

- Click the Video below for full demo.
- 
<p align="center">
  <a href="https://youtu.be/9VBbaElARM8" target="_blank">
    <img src="https://img.youtube.com/vi/9VBbaElARM8/hqdefault.jpg" alt="Video Demo" width="60%">
  </a>
</p>
<p align="center"><em>(Video Demo)</em></p>


## ⚙️ Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Install Ollama

Ollama is used to run local LLMs. Install it via:

-   **Windows / macOS / Linux**:  
    Follow the official instructions at https://ollama.com/download
    
After installation, you can run a model (e.g. `llama3`) directly.

### 3. Launch the App

To start the Streamlit interface, run:

```bash
streamlit run Home.py
```
Then open the provided local URL in your browser to start using the app.
