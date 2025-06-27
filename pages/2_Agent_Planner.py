import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import json
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# ------------------- State Definition ------------------- #

class AgentState(TypedDict):
    """State for the LangGraph workflow"""
    topic: str
    messages: List[Dict[str, str]]
    content: Dict[str, Any]
    current_step: str
    error: str

# ------------------- Tools Setup ------------------- #

class ToolExecutor:
    def __init__(self, tools: list):
        self.tools = {tool.name: tool for tool in tools}
    
    def execute(self, name, **kwargs):
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found.")
        return tool.run(**kwargs)


def setup_tools():
    """Setup search tools"""
    tools = []
    
    # Try SerpAPI first, fallback to DuckDuckGo
    serpapi_key = os.getenv('SERPAPI_API_KEY', '')
    
    if serpapi_key:
        try:
            serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_key)
            serpapi_tool = Tool(
                name="search",
                description="Search the internet for current information",
                func=serpapi.run
            )
            tools.append(serpapi_tool)
        except Exception:
            # Fallback to DuckDuckGo if SerpAPI fails
            ddg_search = DuckDuckGoSearchRun()
            ddg_tool = Tool(
                name="search",
                description="Search the internet for current information",
                func=ddg_search.run
            )
            tools.append(ddg_tool)
    else:
        # Use DuckDuckGo as default
        ddg_search = DuckDuckGoSearchRun()
        ddg_tool = Tool(
            name="search",
            description="Search the internet for current information",
            func=ddg_search.run
        )
        tools.append(ddg_tool)
    
    return tools

# ------------------- LangGraph Agents ------------------- #

class EduSummaristGraph:
    """LangGraph-based teaching system"""
    
    def __init__(self, model_name="llama3.2"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            streaming=True
        )
        self.tools = setup_tools()
        self.tool_executor = ToolExecutor(self.tools)
        
    def research_node(self, state: AgentState) -> AgentState:
        """Research the topic and gather information"""
        topic = state["topic"]
        
        # Search for information about the topic
        search_queries = [
            f"{topic} tutorial beginner guide",
            f"{topic} best practices examples",
            f"{topic} learning resources documentation"
        ]
        
        research_content = []
        for query in search_queries:
            try:
                results = self.tool_executor.invoke({"tool_name": "search", "tool_input": query})
                research_content.append(f"Search: {query}\nResults: {results}\n")
            except Exception as e:
                research_content.append(f"Search failed for: {query}\nError: {str(e)}\n")
        
        state["content"]["research"] = "\n".join(research_content)
        state["current_step"] = "research_complete"
        
        return state
    
    def master_content_node(self, state: AgentState) -> AgentState:
        """Generate main educational content"""
        topic = state["topic"]
        research = state["content"].get("research", "")
        
        prompt = f"""
        Create comprehensive learning materials for: {topic}

        Research Information:
        {research}

        Create a complete educational package with these 4 sections:

        ## 1. TOPIC OVERVIEW & LEARNING PATH
        - Brief summary of the topic based on research
        - Prerequisites and difficulty level
        - Structured learning roadmap (beginner → intermediate → advanced)
        - Time estimates for each level
        - Key learning objectives

        ## 2. DETAILED CONTENT
        - Core concepts explained clearly
        - Step-by-step tutorials with examples
        - Code snippets and practical demonstrations
        - Common pitfalls and how to avoid them
        - Best practices and industry standards

        ## 3. CURATED RESOURCES
        - List 10-15 high-quality resources based on research
        - Include: official docs, tutorials, courses, books, tools
        - Categorize by type and difficulty level
        - Provide brief descriptions and links

        ## 4. PRACTICE EXERCISES
        - 5 beginner exercises with solutions
        - 3 intermediate projects with detailed requirements
        - 2 advanced challenges
        - Assessment criteria and rubrics

        FORMAT: Use clear markdown headers for each section.
        Be comprehensive but concise. Focus on practical, actionable content.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["content"]["master_content"] = response.content
            state["current_step"] = "master_complete"
        except Exception as e:
            state["error"] = f"Master content generation failed: {str(e)}"
            state["current_step"] = "error"
        
        return state
    
    def resource_node(self, state: AgentState) -> AgentState:
        """Generate additional specialized resources"""
        topic = state["topic"]
        
        # Search for specific resources
        resource_queries = [
            f"{topic} official documentation",
            f"{topic} online courses free",
            f"{topic} GitHub repositories examples",
            f"{topic} YouTube tutorials 2024"
        ]
        
        resource_research = []
        for query in resource_queries:
            try:
                results = self.tool_executor.invoke({"tool_name": "search", "tool_input": query})
                resource_research.append(f"Query: {query}\nResults: {results}\n")
            except Exception as e:
                resource_research.append(f"Failed: {query}\nError: {str(e)}\n")
        
        prompt = f"""
        Find the best learning resources for: {topic}

        Research Results:
        {chr(10).join(resource_research)}

        SEARCH STRATEGY:
        1. Official documentation and tutorials
        2. Popular online courses and MOOCs
        3. Recommended books and ebooks
        4. GitHub repositories and code examples
        5. YouTube channels and video tutorials
        6. Community forums and discussion groups

        OUTPUT FORMAT:
        For each resource, provide:
        - **Title**: Name of the resource
        - **Type**: Course/Book/Tutorial/Tool/etc.
        - **Level**: Beginner/Intermediate/Advanced
        - **Description**: Brief summary (1-2 sentences)
        - **Link**: Direct URL if available
        - **Why It's Good**: Specific benefits

        Focus on FREE and high-quality resources.
        Prioritize recent and updated materials (2023-2024).
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["content"]["additional_resources"] = response.content
            state["current_step"] = "resources_complete"
        except Exception as e:
            state["error"] = f"Resource generation failed: {str(e)}"
            state["current_step"] = "error"
        
        return state
    
    def create_fast_workflow(self) -> StateGraph:
        """Create fast workflow (research + master content only)"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("research", self.research_node)
        workflow.add_node("master_content", self.master_content_node)
        
        # Add edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "master_content")
        workflow.add_edge("master_content", END)
        
        return workflow.compile()
    
    def create_detailed_workflow(self) -> StateGraph:
        """Create detailed workflow (research + master content + resources)"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("research", self.research_node)
        workflow.add_node("master_content", self.master_content_node)
        workflow.add_node("resources", self.resource_node)
        
        # Add edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "master_content")
        workflow.add_edge("master_content", "resources")
        workflow.add_edge("resources", END)
        
        return workflow.compile()

# ------------------- Streamlit App ------------------- #

st.set_page_config(
    page_title="🤖 EduSummarist Agent", 
    layout="wide",
    page_icon="🗺️"
    # initial_sidebar_state="collapsed"
)

# Initialize session state
if 'serpapi_api_key' not in st.session_state:
    st.session_state['serpapi_api_key'] = os.getenv('SERPAPI_API_KEY', '')
if 'results' not in st.session_state:
    st.session_state['results'] = {}
if 'processing' not in st.session_state:
    st.session_state['processing'] = False

# Header
st.title("🤖 EduSummarist Agent")
st.markdown("**Generate comprehensive course materials!**")

# API Key check (optional for DuckDuckGo fallback)
api_key_status = "✅ SerpAPI Connected" if st.session_state['serpapi_api_key'] else "⚠️ Using DuckDuckGo (SerpAPI recommended)"
# st.info(f"Search Status: {api_key_status}")

if not st.session_state['serpapi_api_key']:
    with st.expander("🔧 Optional: Add SerpAPI Key for Better Results"):
        manual_key = st.text_input("Enter SerpAPI Key:", type="password")
        if manual_key:
            st.session_state['serpapi_api_key'] = manual_key
            os.environ['SERPAPI_API_KEY'] = manual_key
            st.success("✅ SerpAPI Key saved!")
            st.rerun()

# Main input
st.markdown("### 🎯 What do you want to learn?")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    # Topic input
    topic = st.text_input(
        "Topic",
        placeholder="e.g., Python FastAPI, Docker Containers, React Hooks",
        help="Be specific for better results"
    )

with col2:
    # Generation mode selection
    generation_mode = st.selectbox(
        "Mode",
        ["🚀 Fast (Single Flow)", "🔍 Detailed (Multi-Step)"],
        help="Fast: Research + Content, Detailed: Research + Content + Resources"
    )

with col3:
    # Generate button
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Generate Course", type="primary", use_container_width=True)

# Processing
# Check if generation button was clicked and topic is provided and not already processing
if generate_btn and topic and not st.session_state['processing']:
    st.session_state['processing'] = True
    
    # Initialize LangGraph system
    edu_graph = EduSummaristGraph()
    
    # Initial state
    initial_state = {
        "topic": topic,
        "messages": [],
        "content": {},
        "current_step": "start",
        "error": ""
    }
    
    # Progress indicator
    progress_container = st.container()
    
    with progress_container:
        if "Fast" in generation_mode:
            # Fast workflow
            st.markdown("### 🚀 Generating Course (Fast Mode)")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                workflow = edu_graph.create_fast_workflow()
                
                status_text.text("🔍 Researching topic...")
                progress_bar.progress(0.3)
                
                # Execute workflow
                result = workflow.invoke(initial_state)
                
                progress_bar.progress(0.7)
                status_text.text("🤖 Generating content...")
                
                progress_bar.progress(1.0)
                status_text.text("✅ Course generated successfully!")
                
                st.session_state['results'] = {
                    'content': result['content']['master_content'],
                    'mode': 'fast',
                    'topic': topic
                }
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state['results'] = {}
        
        else:
            # Detailed workflow
            st.markdown("### 🔍 Generating Course (Detailed Mode)")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                workflow = edu_graph.create_detailed_workflow()
                
                status_text.text("🔍 Researching topic...")
                progress_bar.progress(0.2)
                
                status_text.text("🤖 Creating main content...")
                progress_bar.progress(0.6)
                
                status_text.text("🔗 Finding additional resources...")
                progress_bar.progress(0.8)
                
                # Execute workflow
                result = workflow.invoke(initial_state)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Detailed course generated!")
                
                st.session_state['results'] = {
                    'content': {
                        'master_content': result['content']['master_content'],
                        'additional_resources': result['content']['additional_resources']
                    },
                    'mode': 'detailed',
                    'topic': topic
                }
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state['results'] = {}
    
    # Clear progress
    progress_container.empty()
    st.session_state['processing'] = False

elif generate_btn and not topic:
    st.error("Please enter a topic first!")

elif st.session_state['processing']:
    st.info("⏳ Processing... Please wait.")

# Display results
if st.session_state['results'] and not st.session_state['processing']:
    results = st.session_state['results']
    
    st.markdown("---")
    st.markdown(f"## 📚 Course: **{results['topic']}**")
    
    if results['mode'] == 'fast':
        # Single content display
        st.markdown(results['content'])
        
    else:
        # Tabbed display for detailed mode
        tab1, tab2 = st.tabs(["📖 Main Content", "🔗 Additional Resources"])
        
        with tab1:
            st.markdown(results['content']['master_content'])
            
        with tab2:
            st.markdown(results['content']['additional_resources'])
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Generate New Course", use_container_width=True):
            st.session_state['results'] = {}
            st.rerun()
    
    with col2:
        # Export functionality
        if results['mode'] == 'fast':
            export_content = results['content']
        else:
            export_content = f"# {results['topic']}\n\n{results['content']['master_content']}\n\n## Additional Resources\n\n{results['content']['additional_resources']}"
        
        st.download_button(
            "💾 Download Course",
            data=export_content,
            file_name=f"{results['topic'].replace(' ', '_')}_course.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col3:
        # Share functionality
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.info("Content ready to copy! Use Ctrl+A, Ctrl+C above ↑")

# Sidebar info (collapsed by default)
with st.sidebar:
    st.markdown("## 🤖 Agent Modes")
    st.markdown("**🚀 Fast Mode**: Workflow with research + content generation")
    st.markdown("**🔍 Detailed Mode**: Workflow with specialized resource finding")
    
    st.markdown("---")
    st.markdown("### 🎯 What you get:")
    st.markdown("""
    - 📋 Learning roadmap & objectives
    - 📖 Detailed explanations & examples  
    - 🔗 Curated resources & links
    - 💪 Practice exercises & projects
    - 📊 Assessment criteria
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.info("🧠 Model: llama3.2 (Ollama)")
    st.info(f"Search: {api_key_status}")
    
    st.markdown("---")
    # st.markdown("### 🔧 LangGraph Features")
    # st.markdown("""
    # - **State Management**: Structured workflow state
    # - **Tool Integration**: Search tools with fallback
    # - **Error Handling**: Robust error management
    # - **Flexible Workflows**: Fast vs Detailed modes
    # """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Try specific topics like 'Python FastAPI' or 'Machine Learning' for better results!")
# st.markdown("🔄 **Powered by**: LangGraph + Ollama + Search APIs")