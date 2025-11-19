#!/usr/bin/env python3
"""
OG-RAG Web Interface

Simple web UI for:
1. Upload OWL file
2. Auto-process (parse + build hypergraph)
3. Chat with ontology
"""

import streamlit as st
import os
import sys
import time
import threading
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ontology_manager import OntologyManager, ProcessingStatus
from scripts.parse_owl import OWLParser
from build_hypergraph import build_hypergraph
from query_engine.generic_query_engine import GenericQueryEngine


# Load API keys from environment or yaml file
def load_api_keys():
    """Load API keys from environment variables (HF Secrets) or api_keys.yaml"""
    # Priority 1: Environment variables (for Hugging Face Spaces)
    megallm_key = os.getenv('MEGALLM_API_KEY')
    if megallm_key:
        return {'MEGALLM_API_KEY': megallm_key}
    
    # Priority 2: Local yaml file (for development)
    api_keys_file = Path(__file__).parent / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r') as f:
            return yaml.safe_load(f)
    
    return {}

API_KEYS = load_api_keys()


# Page config
st.set_page_config(
    page_title="OG-RAG - Ontology Q&A",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state
if 'manager' not in st.session_state:
    st.session_state.manager = OntologyManager()

if 'current_ontology_id' not in st.session_state:
    st.session_state.current_ontology_id = None

if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def process_ontology(ontology_id: str, owl_file: str, manager: OntologyManager):
    """Background processing: parse + build hypergraph"""
    # Don't use st.session_state in thread - pass manager as parameter
    
    try:
        # Update status: parsing
        manager.update_status(ontology_id, ProcessingStatus.PARSING)
        
        # Parse OWL
        onto_dir = manager.get_ontology_dir(ontology_id)
        parsed_dir = onto_dir / "parsed"
        parsed_dir.mkdir(exist_ok=True)
        
        parser = OWLParser(owl_file, str(parsed_dir))
        parser.parse_streaming()
        
        manager.set_parsed_dir(ontology_id, str(parsed_dir))
        
        # Update status: building
        manager.update_status(ontology_id, ProcessingStatus.BUILDING)
        
        # Build hypergraph
        metadata = build_hypergraph(str(parsed_dir))
        
        # Update metadata and status
        manager.update_metadata(ontology_id, metadata)
        manager.update_status(ontology_id, ProcessingStatus.READY)
        
    except Exception as e:
        manager.update_status(ontology_id, ProcessingStatus.ERROR, str(e))
        print(f"Error processing ontology: {e}")


def upload_page():
    """Upload and process ontology"""
    st.title("🧬 OG-RAG - Ontology Question Answering")
    st.markdown("### Upload Your Ontology")
    
    st.info("""
    **How it works:**
    1. Upload any OWL ontology file
    2. System automatically parses and builds knowledge graph
    3. Chat with your ontology!
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an OWL file",
        type=['owl'],
        help="Upload any OBO-formatted OWL ontology"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        
        with col2:
            if st.button("Process Ontology", type="primary"):
                with st.spinner("Uploading..."):
                    # Save uploaded file
                    temp_file = Path("temp_upload.owl")
                    with open(temp_file, 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    # Add to manager
                    ontology_id = st.session_state.manager.add_ontology(
                        str(temp_file),
                        name=uploaded_file.name.replace('.owl', '')
                    )
                    
                    temp_file.unlink()
                    
                    st.session_state.current_ontology_id = ontology_id
                
                # Start background processing
                onto_info = st.session_state.manager.get_ontology(ontology_id)
                owl_file = onto_info['owl_file']
                
                thread = threading.Thread(
                    target=process_ontology,
                    args=(ontology_id, owl_file, st.session_state.manager)
                )
                thread.daemon = True
                thread.start()
                
                st.success(f"Processing started! Ontology ID: {ontology_id[:8]}")
                st.rerun()
    
    # Show existing ontologies
    st.markdown("---")
    st.markdown("### Your Ontologies")
    
    ontologies = st.session_state.manager.list_ontologies()
    
    if ontologies:
        for onto in ontologies:
            with st.expander(f"📚 {onto['name']} - {onto['status'].upper()}"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**ID:** {onto['id'][:16]}")
                    st.write(f"**Created:** {onto['created_at'][:19]}")
                
                with col2:
                    st.write(f"**Status:** {onto['status']}")
                    if onto['metadata']:
                        meta = onto['metadata']
                        st.write(f"**Terms:** {meta.get('active_terms', 'N/A'):,}")
                
                with col3:
                    if onto['status'] == 'ready':
                        if st.button("Chat", key=f"chat_{onto['id']}"):
                            st.session_state.current_ontology_id = onto['id']
                            st.rerun()
                    elif onto['status'] == 'error':
                        st.error("Error!")
                    else:
                        st.info("Processing...")
    else:
        st.info("No ontologies yet. Upload one above!")


def chat_page():
    """Chat interface"""
    onto_id = st.session_state.current_ontology_id
    onto_info = st.session_state.manager.get_ontology(onto_id)
    
    if not onto_info:
        st.error("Ontology not found!")
        if st.button("← Back"):
            st.session_state.current_ontology_id = None
            st.rerun()
        return
    
    # Show processing status if not ready
    if onto_info['status'] != 'ready':
        st.title(f"🔄 Processing: {onto_info['name']}")
        
        status = onto_info['status']
        error_msg = onto_info.get('error_message', '')
        
        if status == 'uploading':
            st.info("📤 Uploading ontology file...")
            st.progress(0.1)
        
        elif status == 'parsing':
            st.info("🔍 Parsing OWL file...")
            st.markdown("""
            **Current step:** Extracting terms from OWL/XML structure
            - Reading ontology structure
            - Extracting term definitions, labels, synonyms
            - Mapping relationships between terms
            """)
            st.progress(0.3)
            with st.expander("📋 What's happening?"):
                st.markdown("""
                The parser is reading your OWL file and extracting:
                - **Term IDs and labels**
                - **Definitions and descriptions**
                - **Synonyms and alternative names**
                - **Relationships** (is_a, part_of, etc.)
                - **Cross-references**
                - **Comments and examples**
                """)
        
        elif status == 'building':
            st.info("🏗️ Building knowledge graph...")
            st.markdown("""
            **Current step:** Creating hypergraph structure
            - Flattening terms to facts (chunks)
            - Building hypernode key-value pairs
            - Generating embeddings (this may take a while)
            """)
            st.progress(0.6)
            with st.expander("📋 What's happening?"):
                st.markdown("""
                Building the knowledge graph involves:
                1. **Smart Chunking**: Splitting each term into chunks (core, synonyms, relationships, details)
                2. **HyperNode Creation**: Creating searchable key-value pairs (~17 nodes per term)
                3. **Embedding Generation**: Converting text to vectors using sentence-transformers
                
                This can take a while depending on your ontology.
                """)
        
        elif status == 'error':
            st.error(f"❌ Error during processing!")
            st.code(error_msg, language="text")
            if st.button("← Back to Upload"):
                st.session_state.current_ontology_id = None
                st.rerun()
            return
        
        else:
            st.warning(f"Unknown status: {status}")
        
        # Auto-refresh every 3 seconds
        st.info("🔄 Auto-refreshing... (Page will update automatically when processing completes)")
        time.sleep(3)
        st.rerun()
        return
    
    # Load query engine if needed
    if st.session_state.query_engine is None:
        with st.spinner("Loading query engine..."):
            parsed_dir = st.session_state.manager.get_parsed_dir(onto_id)
            # Get API key from yaml file or environment
            api_key = API_KEYS.get('MEGALLM_API_KEY') or os.getenv('MEGALLM_API_KEY')
            
            try:
                st.session_state.query_engine = GenericQueryEngine(
                    str(parsed_dir),
                    megallm_api_key=api_key
                )
            except Exception as e:
                st.error(f"Error loading engine: {e}")
                return
    
    engine = st.session_state.query_engine
    metadata = onto_info['metadata']
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"💬 Chat with {onto_info['name']}")
        st.caption(f"{metadata['ontology_name']} • {metadata['active_terms']:,} terms")
    
    with col2:
        if st.button("← Change Ontology"):
            st.session_state.current_ontology_id = None
            st.session_state.query_engine = None
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask about this ontology..."):
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching ontology..."):
                result = engine.query(prompt, top_k=5, use_llm=True)
            
            answer = result['answer']
            st.markdown(answer)
            
            # Show context sent to LLM (for hallucination verification)
            with st.expander(f"Context sent to LLM ({len(result['retrieved_facts'])} terms) - Click to verify"):
                st.markdown("**This is the exact information provided to the LLM:**")
                st.markdown("---")
                st.text(result['full_context'])
                st.markdown("---")
                st.caption("💡 Use this to check if LLM answer is grounded in the retrieved facts")
        
        # Add assistant message
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': answer
        })


def clear_session():
    """Clear current session and delete ALL ontologies"""
    import shutil
    
    # Get data directory
    data_dir = Path(__file__).parent / "data" / "ontologies"
    
    # Delete all ontology data
    if data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            st.error(f"Error deleting ontologies: {e}")
    
    # Reset session state completely
    st.session_state.manager = OntologyManager()
    st.session_state.current_ontology_id = None
    st.session_state.query_engine = None
    st.session_state.chat_history = []
    
    st.success("✓ Session cleared! All ontologies and data have been deleted.")
    time.sleep(1)
    st.rerun()


def main():
    """Main app"""
    
    # Sidebar
    with st.sidebar:
        st.title("OG-RAG")
        st.markdown("**Ontology-Grounded RAG**")
        st.markdown("---")
        
        ready_ontos = st.session_state.manager.get_ready_ontologies()
        st.metric("Ready Ontologies", len(ready_ontos))
        
        st.markdown("---")
        
        # Clear session button with confirmation
        st.markdown("### 🗑️ Clear Session")
        st.caption("⚠️ This will delete ALL ontologies")
        
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        
        if not st.session_state.confirm_clear:
            if st.button("Clear All Data", use_container_width=True, type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Are you sure? This cannot be undone!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Yes, Delete", use_container_width=True, type="primary"):
                    st.session_state.confirm_clear = False
                    clear_session()
            with col2:
                if st.button("✗ Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        OG-RAG enables Q&A with any ontology:
        - Upload OWL file
        - Auto-build knowledge graph
        - Chat with AI
        """)
    
    # Main content
    if st.session_state.current_ontology_id:
        chat_page()
    else:
        upload_page()


if __name__ == "__main__":
    main()
