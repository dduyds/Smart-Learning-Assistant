import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema import BaseRetriever, Document
from langchain.retrievers import EnsembleRetriever
from typing import List, Dict, Optional
import os
import tempfile
import validators
import re

# Page Configuration
st.set_page_config(
    page_title="🤖 DocQA Chatbot",
    page_icon="📄",
    layout="wide"
)

# Main Title
st.title("🤖 Document & Web Q&A Chatbot")
st.markdown("**Chat with your PDF/DOCX documents and web pages using AI!**")

# Configuration Constants
CHUNK_SIZE = 1000  # Size of each text chunk for processing
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context
OLLAMA_MODEL_EMBEDDINGS = "mxbai-embed-large"  # Model for creating embeddings
OLLAMA_MODEL_LLM = "llama3.2"  # Language model for generating responses

# Initialize Components
@st.cache_resource
def init_components():
    """
    Initialize the core components needed for the chatbot:
    - Embeddings: Convert text to numerical vectors for similarity search
    - Vector Store: Store and search document embeddings
    - Model: Generate responses to user questions
    """
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL_EMBEDDINGS)
    vector_store = InMemoryVectorStore(embeddings)
    model = OllamaLLM(model=OLLAMA_MODEL_LLM)
    return embeddings, vector_store, model

embeddings, vector_store, model = init_components()

# Document Context Storage for Keyword Search
if 'document_context' not in st.session_state:
    st.session_state.document_context = {}

# Session state for tracking current content
if 'content_processed' not in st.session_state:
    st.session_state.content_processed = False
if 'current_content' not in st.session_state:
    st.session_state.current_content = None
if 'content_type' not in st.session_state:
    st.session_state.content_type = None

# Custom Keyword Retriever Class
class KeywordRetriever(BaseRetriever):
    """
    Custom keyword-based retriever that searches for exact keyword matches.
    This complements the vector search by finding documents that contain
    specific terms mentioned in the user's query.
    """
    
    def __init__(self, context_dict: Dict[str, str], **kwargs):
        # Initialize BaseRetriever with kwargs
        super().__init__(**kwargs)
        # Store context_dict as an instance attribute
        self._context_dict = context_dict

    def _get_relevant_documents(
        self, 
        query: str,
    ) -> List[Document]:
        """
        Find documents that contain keywords from the user's query.
        Returns documents ranked by the number of matching keywords.
        """
        relevant_docs = []
        query_keywords = set(query.lower().split())
        
        for doc_id, content in self._context_dict.items():
            content_lower = content.lower()
            # Calculate score based on number of matching keywords
            matches = sum(1 for keyword in query_keywords if keyword in content_lower)
            if matches > 0:
                # Find position of first matching keyword
                first_keyword = next((kw for kw in query_keywords if kw in content_lower), None)
                if first_keyword:
                    # Create short snippet around the found keyword
                    keyword_pos = content_lower.find(first_keyword)
                    snippet_start = max(0, keyword_pos - 100)
                    snippet_end = min(len(content), snippet_start + 500)
                    snippet = content[snippet_start:snippet_end]
                    
                    relevant_docs.append(Document(
                        page_content=snippet if len(snippet) < len(content) else content,
                        metadata={
                            "source": doc_id,
                            "keyword_matches": matches,
                            "full_content": content
                        }
                    ))
        
        # Sort by number of matching keywords (descending)
        relevant_docs.sort(key=lambda x: x.metadata.get("keyword_matches", 0), reverse=True)
        return relevant_docs[:10]  # Return top 10 results

    async def _aget_relevant_documents(
        self, 
        query: str, 
    ) -> List[Document]:
        """Async version of the retrieval method"""
        return self._get_relevant_documents(query)

# Response Template
TEMPLATE = """
You are an intelligent assistant specialized in answering questions based on document content or web page content.
Use the provided information to answer questions accurately and concisely.

Requirements:
- Answer clearly and with good structure in English
- If information is not found in the content, say "I cannot find this information in the provided content"
- Keep answers concise but complete (maximum 3-4 sentences)
- When possible, cite specific information from the content

Question: {question}
Content: {context}

Answer:
"""

# Document Processing Functions
def load_document(file_path, file_type):
    """
    Load documents from file path based on file type.
    Supports PDF and DOCX formats.
    """
    try:
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_type}")
        
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None

def validate_url(url):
    """
    Validate if the provided string is a valid URL.
    """
    return validators.url(url)

def clean_single_url(url_text):
    """
    Clean and validate a single URL.
    """
    if not url_text:
        return None

    url = url_text.strip()

    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    return url if validate_url(url) else None

def load_web_content(urls):
    """
    Load content from web URLs using SeleniumURLLoader.
    """
    try:
        if not urls:
            return None
        
        # Initialize the loader with the URLs
        loader = SeleniumURLLoader(urls=urls)
        
        # Load the documents
        documents = loader.load()
        
        return documents
    except Exception as e:
        st.error(f"Error loading web content: {e}")
        return None

def split_text(documents):
    """
    Split documents into smaller chunks for better processing.
    Uses recursive character splitting to maintain context.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_documents(documents):
    """
    Index documents in both vector store and keyword context.
    - Vector store: For semantic similarity search
    - Context dict: For keyword-based search
    """
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    # Create context dictionary for keyword search
    context_dict = {}
    for i, doc in enumerate(documents):
        doc_id = f"chunk_{i}"
        context_dict[doc_id] = doc.page_content
    
    # Save context to session state
    st.session_state.document_context = context_dict
    
    return len(documents)

def create_hybrid_retriever():
    """
    Create hybrid retriever combining keyword and vector search.
    This provides both exact keyword matching and semantic similarity.
    """
    if not st.session_state.document_context:
        return None
    
    try:
        # Create keyword retriever
        keyword_retriever = KeywordRetriever(context_dict=st.session_state.document_context)
        
        # Create vector retriever
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": 15}  # Get 15 results from vector search
        )
        
        # Create ensemble retriever with weighted combination
        hybrid_retriever = EnsembleRetriever(
            retrievers=[keyword_retriever, vector_retriever],
            weights=[0.4, 0.6]  # Slightly favor vector search (60% vs 40%)
        )
        
        return hybrid_retriever
    except Exception as e:
        st.error(f"Error creating hybrid retriever: {e}")
        return None

def retrieve_documents_hybrid(query, k=5):
    """
    Use hybrid retriever to search for relevant documents.
    Combines keyword and semantic search, removes duplicates.
    """
    hybrid_retriever = create_hybrid_retriever()
    if hybrid_retriever:
        try:
            results = hybrid_retriever.get_relevant_documents(query)
            # Remove duplicates based on content
            unique_results = []
            seen_content = set()
            
            for doc in results:
                content_hash = hash(doc.page_content[:200])  # Hash first 200 characters
                if content_hash not in seen_content:
                    unique_results.append(doc)
                    seen_content.add(content_hash)
                
                if len(unique_results) >= k:
                    break
            
            return unique_results
        except Exception as e:
            st.error(f"Error in hybrid search: {e}")
            # Fallback to regular vector search
            return vector_store.similarity_search(query, k=k)
    else:
        # Fallback to regular vector search
        return vector_store.similarity_search(query, k=k)

def answer_question(question, context):
    """
    Generate answer using the language model.
    Takes user question and relevant context to produce response.
    """
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def clear_vector_store():
    """
    Clear the vector store and context dictionary.
    Used when processing new content.
    """
    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    # Clear context dictionary
    st.session_state.document_context = {}


# Sidebar with Instructions
with st.sidebar:
    st.header("📖 How to Use")
    st.write("**Option 1: Upload Document**")
    st.write("• Upload a PDF or DOCX file")
    st.write("• Wait for processing")
    st.write("• Ask questions about the content")
    
    st.write("**Option 2: Load Web Content**")
    st.write("• Enter one URLs")
    st.write("• Wait for web scraping")
    st.write("• Chat with the web content")
    
    st.divider()
    
    st.header("💡 Sample Questions")
    example_questions = [
        "What is this content about?",
        "Summarize the main points discussed",
        "What are the key conclusions?",
    ]
    
    for i, example in enumerate(example_questions):
        # Create a button for each example question
        if st.button(example, key=f"example_{i}"):
            # Store example question in session state
            st.session_state.example_question = example

# Single Tab: Content Input with Toggle Options
st.header("📚 Content Input")

# Toggle between input methods
input_method = st.radio(
    "Choose your input method:",
    ["📁 Upload File", "🌐 Load from URL"],
    horizontal=True,
    help="Select whether you want to upload a document or load content from a web URL"
)

st.divider()

# Initialize variables
uploaded_file = None
url_input = None
content_to_process = None
content_id = None
file_type = None

# File Upload Section
if input_method == "📁 Upload File":
    st.subheader("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file to analyze",
        type=["pdf", "docx"],
        help="Supported formats: PDF (.pdf) and Word (.docx)"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        content_id = f"file_{uploaded_file.name}"
        content_to_process = "file"
        
        # Display file information
        st.subheader("📋 Document Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**📄 File name:** {uploaded_file.name}")
        with col2:
            st.write(f"**📊 Size:** {uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.write(f"**📑 Format:** {file_type.upper()}")

# URL Input Section
elif input_method == "🌐 Load from URL":
    st.subheader("🌐 Load Web Content")
    url_input = st.text_input(
        "Paste a URL:",
        placeholder="https://example.com",
        help="Enter a single URL to load content from"
    )
    
    if url_input:
        cleaned_url = clean_single_url(url_input)
        
        if cleaned_url:
            content_id = f"url_{cleaned_url.replace('://', '_').replace('/', '_')}"
            content_to_process = "url"
            
            st.subheader("🔗 URL to Process")
            st.code(cleaned_url, language=None)
        else:
            st.warning("⚠️ Please enter a valid URL.")

# Unified Processing Section
if content_to_process and (uploaded_file or (url_input and cleaned_url)):
    st.divider()
    
    # Check if content needs processing if existing one of the following conditions:
    # - content_processed is False (no content has been processed yet)
    # - current_content does not match content_id (new content uploaded)
    # - content_type does not match content_to_process (content type has changed)
    # content_processed is False if no content has been processed yet
    # content_id and content_id must match if content has been processes
    # content_type and content_to_process must match if content has been processed
    # If you load a new different file or URL, it will be processed again, if you load the same file or URL, it will not be processed again
    needs_processing = (
        not st.session_state.content_processed or 
        st.session_state.current_content != content_id or 
        st.session_state.content_type != content_to_process
    )
    
    if needs_processing:
        # Process button
        process_label = "🚀 Process Document" if content_to_process == "file" else "🚀 Load Web Content"
        
        if st.button(process_label, type="primary", use_container_width=True):
            
            # Unified processing logic
            with st.spinner("🔄 Processing content... Please wait."):
                progress_bar = st.progress(0)
                
                try:
                    # Clear old vector store
                    clear_vector_store()
                    progress_bar.progress(25)
                    
                    documents = None
                    
                    # Process based on content type
                    if content_to_process == "file":
                        # File processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            temp_file_path = tmp_file.name
                        
                        progress_bar.progress(40)
                        documents = load_document(temp_file_path, file_type)
                        progress_bar.progress(60)
                        
                        # Cleanup temp file
                        os.remove(temp_file_path)
                        
                    elif content_to_process == "url":
                        # URL processing
                        progress_bar.progress(40)
                        documents = load_web_content([cleaned_url])
                        progress_bar.progress(60)
                        
                        # Filter out empty documents for web content
                        if documents:
                            documents = [doc for doc in documents if doc.page_content.strip()]
                    
                    # Common processing for both types
                    if documents and len(documents) > 0:
                        # Split documents into manageable chunks
                        chunked_documents = split_text(documents)
                        progress_bar.progress(80)
                        
                        # Index documents in vector store
                        chunks_count = index_documents(chunked_documents)
                        progress_bar.progress(90)
                        
                        # Update session state
                        st.session_state.content_processed = True
                        st.session_state.current_content = content_id
                        st.session_state.content_type = content_to_process
                        st.session_state.chunks_count = chunks_count
                        st.session_state.original_docs_count = len(documents)
                        
                        if content_to_process == "url":
                            st.session_state.processed_urls = [cleaned_url]
                        
                        progress_bar.progress(100)
                        
                        # Success message
                        content_name = uploaded_file.name if content_to_process == "file" else "web content"
                        st.success(f"✅ {content_name} processed successfully! Split into {chunks_count} chunks for analysis.")
                        
                        # Statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            label = "📄 Total pages" if content_to_process == "file" else "🌐 Pages loaded"
                            st.metric(label, len(documents))
                        with col2:
                            st.metric("🔍 Search chunks", chunks_count)
                        
                        # Auto-refresh to show chat interface
                        st.rerun()
                        
                    else:
                        error_msg = "❌ Cannot process document. Please check the file format." if content_to_process == "file" else "❌ No valid content found in the provided URL."
                        st.error(error_msg)
                        
                        if content_to_process == "url":
                            st.write("This might be due to:")
                            st.write("• Website blocking automated access")
                            st.write("• Network issues")
                            st.write("• Invalid URL")
                            st.write("• Content requiring JavaScript")
                
                except Exception as e:
                    st.error(f"❌ Error processing content: {str(e)}")
                    
    else:
        # Content already processed
        st.info("✅ Content is already processed and ready for questions!")
        
        # Show current content info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔍 Search chunks", st.session_state.chunks_count)
        with col2:
            pages_label = "📄 Total pages" if st.session_state.content_type == "file" else "🌐 Pages loaded"
            st.metric(pages_label, st.session_state.original_docs_count)

# Chat Interface (displayed when content is processed)
if st.session_state.content_processed:
    st.divider()
    st.header("💬 Chat with Content")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Reset messages if content changed
    if not st.session_state.messages or st.session_state.get('last_content') != st.session_state.current_content:
        if st.session_state.content_type == "file":
            welcome_msg = f"👋 Hello! I've analyzed your document. You can ask questions about its content!"
        else:
            welcome_msg = f"👋 Hello! I've analyzed the web content. You can ask questions about it!"
        
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.last_content = st.session_state.current_content
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle example questions
    if 'example_question' in st.session_state:
        question = st.session_state.example_question
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display example question in chat
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching for relevant information..."):
                try:
                    # Retrieve documents using hybrid search
                    retrieved_docs = retrieve_documents_hybrid(question, k=5)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    # Generate answer using the context
                    answer = answer_question(question, context)
                    
                    # Display number of relevant sections found
                    # Format and display the answer
                    formatted_answer = f"{answer}\n\n*📚 Based on {len(retrieved_docs)} relevant sections from content*"
                    st.write(formatted_answer)

                    st.write(formatted_answer)
                    # Append answer to chat history
                    st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
                except Exception as e:
                    error_msg = f"❌ Sorry, an error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Remove example question from session state if it exists
        del st.session_state.example_question
    
    # Chat input
    if question := st.chat_input("💭 Ask a question about the content..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching for relevant information..."):
                try:
                    retrieved_docs = retrieve_documents_hybrid(question)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    answer = answer_question(question, context)
                    
                    formatted_answer = f"{answer}\n\n*📚 Based on {len(retrieved_docs)} relevant sections from content*"
                    st.write(formatted_answer)
                    st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, an error occurred while processing the question: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Control buttons
    # Check if there are messages in the chat history
    if st.session_state.messages:
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear chat history"):
                content_type = "document" if st.session_state.content_type == "file" else "web content"
                st.session_state.messages = [
                    {"role": "assistant", "content": f"👋 Hello! I've analyzed the {content_type}. You can ask questions about its content!"}
                ]
                # Delete messages 
                st.rerun()
        
        with col2:
            if st.button("🔄 Reprocess content"):
                # Reset vector store and context
                st.session_state.content_processed = False
                st.rerun()
        
        with col3:
            st.metric("💬 Messages", len(st.session_state.messages))

else:
    # No content processed - Display welcome screen
    st.header("🚀 Getting Started")
    st.info("👆 Please select an input method and provide content to start chatting!")