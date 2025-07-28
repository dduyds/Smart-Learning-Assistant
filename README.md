# 📚 Smart Learning Assistant  
An AI-powered tool designed to support self-directed learning through document-based Q&A and guided study planning.

---

## 🔍 Overview

**Smart Learning Assistant** helps users independently explore complex topics by combining two core mechanisms:

- **RAG-based Q&A Bot**: Leverages **Retrieval-Augmented Generation (RAG)** to answer questions based on uploaded PDFs, DOCX files, or external URLs using hybrid semantic + keyword search.

- **Graph-based Learning Planner**: Implements a structured **graph pipeline** that simulates agent-like behavior, guiding users through online resource discovery and personalized learning path generation.


## ✨ Key Features

### 🔍 Hybrid RAG Retrieval
- Supports multiple formats: **PDFs**, **DOCX**, and **web links**.
- Performs **hybrid search** combining **semantic vectors** and **keywords** for accurate and context-aware answers.

### 🧠 Graph-Based Study Planner
- Built using a **LangGraph pipeline**, where each node represents a specialized task:
  - Research relevant resources.
  - Generate structured learning materials.
  - Curate additional reference content.
- Each node integrates **LLMs and tools**, simulating agentic behavior.

### 🖥️ Interactive User Interface
- Developed with **Streamlit** for real-time interaction and document upload.
- Provides a simple, accessible experience for learners and educators.


## 💬 Use Cases
- Ask questions about content inside your documents.
- Get a structured study roadmap on any technical topic.
- Automatically gather online tutorials, documentation, and practice problems.


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
