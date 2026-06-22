"""
Invoice RAG - One Page Compact UI
==================================
- Everything fits one page
- No scroll
- Same functionality
- Compact spacing
"""

import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from src.rag_pipeline import RAGPipeline
import time

load_dotenv()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Invoice AI Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarUserContent"] { display: none; }
    section[data-testid="stSidebar"] > div { display: none; }
</style>
""", unsafe_allow_html=True)

# =========================
# CUSTOM UI STYLES - ONE PAGE
# =========================
st.markdown("""
<style>

/* GLOBAL THEME */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1c 0%, #111827 50%, #0b1220 100%);
    color: #e5e7eb;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    overflow: hidden;
}

/* REMOVE STREAMLIT DEFAULTS */
#MainMenu, footer, header {
    visibility: hidden;
}

/* MAIN CONTAINER - COMPACT */
[data-testid="stMainBlockContainer"] {
    padding: 0.5rem 1rem !important;
    max-width: 100%;
}

/* TITLE - SMALLER */
.main-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin: 0.8rem 0 0.2rem 0;
}

.sub-title {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
}

/* UPLOAD SECTION - COMPACT */
.upload-section {
    max-width: 700px;
    margin: 0 auto;
    padding: 0 1rem;
}

.upload-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
    border: 2px dashed rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 30px 20px;
    text-align: center;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.upload-box:hover {
    border-color: rgba(59, 130, 246, 0.6);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
}

.upload-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.upload-text {
    color: #cbd5e1;
    font-size: 0.85rem;
    margin: 0.3rem 0;
}

/* FILE UPLOADER */
.stFileUploader {
    margin: 0.5rem 0;
}

.stFileUploader > div > div {
    padding: 10px !important;
}

/* BUTTONS - COMPACT */
.button-group {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin: 0.5rem 0;
    flex-wrap: wrap;
}

.stButton > button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    transform: translateY(-1px);
    box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
}

/* STATUS MESSAGE */
.status-success {
    background: rgba(34, 197, 94, 0.1);
    border-left: 3px solid #22c55e;
    color: #86efac;
    padding: 10px;
    border-radius: 8px;
    margin: 0.5rem 0;
    text-align: center;
    font-weight: 600;
    font-size: 0.85rem;
}

.status-loading {
    background: rgba(59, 130, 246, 0.1);
    border-left: 3px solid #3b82f6;
    color: #93c5fd;
    padding: 10px;
    border-radius: 8px;
    margin: 0.5rem 0;
    text-align: center;
    font-weight: 600;
    font-size: 0.85rem;
}

/* DIVIDER */
.divider-section {
    max-width: 700px;
    margin: 0.5rem auto;
    padding: 0 1rem;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.2), transparent);
    margin: 0.5rem 0;
}

/* CHAT CONTAINER - COMPACT */
.chat-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0.5rem 1rem 80px 1rem;
    max-height: 350px;
    overflow-y: auto;
}

/* USER MESSAGE */
.user-msg {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 10px 14px;
    border-radius: 14px;
    margin: 6px 0;
    width: fit-content;
    max-width: 75%;
    margin-left: auto;
    color: white;
    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.2);
    word-wrap: break-word;
    font-size: 0.9rem;
}

/* BOT MESSAGE */
.bot-msg {
    background: rgba(31, 41, 55, 0.8);
    padding: 10px 14px;
    border-radius: 14px;
    margin: 6px 0;
    width: fit-content;
    max-width: 75%;
    color: #e5e7eb;
    border: 1px solid rgba(148, 163, 184, 0.2);
    word-wrap: break-word;
    line-height: 1.4;
    font-size: 0.9rem;
}

/* INPUT SECTION - FIXED BOTTOM COMPACT */
.input-section {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, #111827, rgba(17, 24, 39, 0.95));
    padding: 12px;
    max-width: 100%;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
}

.input-wrapper {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* TEXT INPUT */
.stTextInput > div > div > input {
    background: rgba(31, 41, 55, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.3);
    color: #e5e7eb;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.9rem;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    background: rgba(31, 41, 55, 0.95);
}

/* EMPTY STATE */
.empty-state {
    text-align: center;
    padding: 20px;
    color: #94a3b8;
    font-size: 0.85rem;
}

.empty-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
def init_session():
    """Initialize all session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pdf_ready" not in st.session_state:
        st.session_state.pdf_ready = False
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

init_session()

# =========================
# PIPELINE INITIALIZATION
# =========================
def get_pipeline():
    """Get or create RAG pipeline."""
    if st.session_state.pipeline is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.error("❌ GROQ_API_KEY not set in .env file")
            st.stop()
        
        try:
            st.session_state.pipeline = RAGPipeline(
                groq_api_key=api_key,
                chunk_size=1000,
                chunk_overlap=200,
                top_k=3,
                vectorstore_dir="vectorstore"
            )
        except Exception as e:
            st.error(f"❌ Error initializing pipeline: {str(e)}")
            st.stop()
    
    return st.session_state.pipeline

# =========================
# PDF PROCESSING
# =========================
def process_pdf(uploaded_file):
    """Process and index PDF file."""
    
    if uploaded_file is None:
        st.error("❌ No file selected")
        return False
    
    try:
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get pipeline and index
        pipeline = get_pipeline()
        pipeline.index_pdf(str(file_path))
        
        # Update session state
        st.session_state.pdf_ready = True
        st.session_state.file_name = uploaded_file.name
        
        return True
    
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False

# =========================
# HEADER
# =========================
st.markdown("<h1 class='main-title'>📄 Invoice AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Chat with your invoices using AI</p>", unsafe_allow_html=True)

# =========================
# MAIN INTERFACE
# =========================

if not st.session_state.pdf_ready:
    # ===== UPLOAD SECTION (CENTERED) =====
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='upload-box'>
        <div class='upload-icon'>📄</div>
        <div class='upload-text'><strong>Upload Your Invoice</strong></div>
        <div class='upload-text' style='font-size: 0.75rem; color: #64748b;'>
            PDF format • Maximum 50MB
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Select PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )
    
    # Process button
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("✅ Process Invoice", use_container_width=True):
                st.markdown("<div class='status-loading'>⏳ Processing...</div>", unsafe_allow_html=True)
                
                if process_pdf(uploaded_file):
                    st.markdown("<div class='status-success'>✅ Ready! Ask below</div>", unsafe_allow_html=True)
                    time.sleep(1)
                    st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # ===== CHAT INTERFACE (AFTER UPLOAD) =====
    
    # Status bar - compact
    st.markdown(f"""
    <div style='text-align: center; padding: 8px; background: rgba(34, 197, 94, 0.08); 
                border-radius: 10px; margin: 0 1rem 0.5rem 1rem; border: 1px solid rgba(34, 197, 94, 0.2);
                font-size: 0.85rem;'>
        <strong>✅ Loaded:</strong> {st.session_state.file_name}
    </div>
    """, unsafe_allow_html=True)
    
    # Chat display
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    if st.session_state.chat:
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-msg'>You: {msg['text']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='bot-msg'>Assistant: {msg['text']}</div>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown("""
        <div class='empty-state'>
            <div class='empty-icon'>💬</div>
            <p>Ask your first question below</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== INPUT SECTION (FIXED BOTTOM) =====
    st.markdown("<div class='input-section'>", unsafe_allow_html=True)
    st.markdown("<div class='input-wrapper'>", unsafe_allow_html=True)
    
    # Query input
    query = st.text_input(
        "Ask about invoice...",
        placeholder="e.g., What is the invoice number?",
        key="user_input",
        label_visibility="collapsed"
    )
    
    # Action buttons - compact
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    ask_btn = False
    clear_btn = False
    new_upload_btn = False
    
    with col1:
        ask_btn = st.button("🔍 Ask", use_container_width=True, key="ask_btn")
    
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True, key="clear_btn")
    
    with col3:
        new_upload_btn = st.button("📤 New", use_container_width=True, key="new_upload_btn")
    
    # Quick questions - compact
    st.markdown("**⚡** ", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_qs = [
        ("Invoice #", "What is the invoice number?"),
        ("Payment", "What is the payment term?"),
        ("Shipper", "What is the shipper line?"),
        ("Shipment", "What is the shipment term?")
    ]
    
    cols = [col1, col2, col3, col4]
    
    for (label, question), col in zip(quick_qs, cols):
        with col:
            if st.button(label, use_container_width=True, key=f"q_{label}"):
                query = question
                ask_btn = True
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process query
    if ask_btn and query:
        # Add user message
        st.session_state.chat.append({"role": "user", "text": query})
        
        # Get answer
        try:
            pipeline = get_pipeline()
            
            with st.spinner("🤖 Thinking..."):
                result = pipeline.answer_question_with_context(query)
            
            # Add bot response
            st.session_state.chat.append({
                "role": "bot",
                "text": result["answer"]
            })
            
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if st.session_state.chat:
                st.session_state.chat.pop()
    
    # Clear history
    if clear_btn:
        st.session_state.chat = []
        st.rerun()
    
    # New upload
    if new_upload_btn:
        st.session_state.pdf_ready = False
        st.session_state.chat = []
        st.session_state.pipeline = None
        st.rerun()