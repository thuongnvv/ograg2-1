import streamlit as st
import os
import sys
import json
import time
import threading
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ontology_manager import OntologyManager, ProcessingStatus
from scripts.parse_owl import OWLParser
from build_hypergraph import build_hypergraph
from query_engine.generic_query_engine import GenericQueryEngine
from ontology_generator import OntologyGenerator


# Load configuration from yaml file
def load_config():
    """Load configuration from api_keys.yaml"""
    # Default configuration (Ollama local)
    default_config = {
        'USE_OLLAMA': True,
        'OLLAMA_MODEL': 'llama3.3:70b',
        'OLLAMA_EMBED_MODEL': 'nomic-embed-text',
        'OLLAMA_BASE_URL': 'http://localhost:11434'
    }
    
    # Try to load from yaml file
    config_file = Path(__file__).parent / "api_keys.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if config:
                # Merge with defaults
                merged = {**default_config, **config}
                
                # Auto-detect: if API key is provided, disable Ollama for LLM
                # (embeddings still use Ollama for privacy)
                if merged.get('openai_api_key') and merged['openai_api_key'] not in ['YOUR_API_KEY_HERE', 'YOUR_OPENROUTER_API_KEY']:
                    merged['USE_OLLAMA'] = False
                
                return merged
    
    return default_config

CONFIG = load_config()


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

if 'page' not in st.session_state:
    st.session_state.page = 'upload'  # 'upload' or 'generate'

if 'generated_owl' not in st.session_state:
    st.session_state.generated_owl = None


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
        
        # Build hypergraph with Ollama support
        use_ollama = CONFIG.get('USE_OLLAMA', True)
        ollama_embed_model = CONFIG.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
        ollama_base_url = CONFIG.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        metadata = build_hypergraph(
            str(parsed_dir),
            use_ollama=use_ollama,
            ollama_model=ollama_embed_model,
            ollama_base_url=ollama_base_url
        )
        
        # Update metadata and status
        manager.update_metadata(ontology_id, metadata)
        manager.update_status(ontology_id, ProcessingStatus.READY)
        
    except Exception as e:
        manager.update_status(ontology_id, ProcessingStatus.ERROR, str(e))
        print(f"Error processing ontology: {e}")


def generate_ontology_page():
    """Generate ontology from documents/URLs"""
    st.title("🤖 Auto-Generate Ontology")
    st.markdown("### Create OWL ontology from your documents or web pages")
    
    st.info("""
    **How it works:**
    1. Upload a document (PDF/DOCX) or provide a URL
    2. LLM analyzes the content and generates an OWL ontology
    3. Review and edit the generated ontology
    4. Download for validation in Protégé (optional)
    5. Process and chat with the ontology
    """)
    
    # Check for API configuration
    api_key = None
    base_url = None
    model_name = "gpt-4"
    use_ollama = CONFIG.get('USE_OLLAMA', False)
    
    # If using Ollama, no API key needed
    if use_ollama:
        model_name = CONFIG.get('OLLAMA_MODEL', 'tinyllama')
        st.success(f"✅ Using Ollama (Local): {model_name}")
    else:
        # Try api_keys.yaml for cloud API
        api_keys_file = Path("api_keys.yaml")
        if api_keys_file.exists():
            with open(api_keys_file, 'r') as f:
                api_keys = yaml.safe_load(f)
                api_key = api_keys.get('openai_api_key')
                if api_key and api_key not in ["YOUR_OPENROUTER_API_KEY", "YOUR_API_KEY_HERE"]:
                    base_url = api_keys.get('openai_base_url')
                    model_name = api_keys.get('openai_model', 'gpt-4')
                else:
                    api_key = None
        
        # API key required for cloud mode
        if not api_key:
            st.warning("⚠️ No API key found in api_keys.yaml. Please add your API key to continue.")
            st.info("Edit `api_keys.yaml` and add your MegaLLM or other provider API key, OR set `USE_OLLAMA: true` to use local Ollama.")
            st.stop()
        
        # Display mode
        if base_url:
            st.success(f"✅ Using {model_name}")
        else:
            st.success(f"✅ Using OpenAI {model_name}")
    
    # Source selection
    st.markdown("---")
    st.markdown("### Step 1: Choose your source")
    
    source_type = st.radio(
        "Source type:",
        ["📄 Upload Document", "🌐 Web URL"],
        horizontal=True
    )
    
    domain = st.text_input(
        "Domain context (optional)",
        placeholder="e.g., biology, medical, environmental...",
        help="Helps LLM understand the context"
    )
    
    generated_result = None
    
    if source_type == "📄 Upload Document":
        uploaded_file = st.file_uploader(
            "Upload PDF or DOCX",
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Upload a document to analyze"
        )
        
        if uploaded_file and st.button("🤖 Generate Ontology", type="primary"):
            with st.spinner(f"Analyzing document with {model_name}..."):
                try:
                    # Save temp file
                    temp_file = Path(f"temp_{uploaded_file.name}")
                    with open(temp_file, 'wb') as f:
                        f.write(uploaded_file.read())
                    
                    # Initialize generator
                    if use_ollama:
                        generator = OntologyGenerator(
                            use_ollama=True,
                            model=model_name,
                            ollama_base_url=f"{CONFIG.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/v1"
                        )
                    else:
                        generator = OntologyGenerator(
                            api_key=api_key,
                            model=model_name,
                            base_url=base_url
                        )
                    
                    generated_result = generator.process_file(
                        str(temp_file),
                        domain=domain or "general"
                    )
                    
                    # Cleanup
                    temp_file.unlink()
                    
                    # Store in session
                    st.session_state.generated_owl = generated_result
                    
                    st.success("✅ Ontology generated successfully!")
                    
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Generation failed: {error_msg}")
                    
                    # Helpful tips
                    if "PDF extraction failed" in error_msg or "corrupted" in error_msg.lower():
                        st.info("""
                        **💡 Tips for PDF files:**
                        - Try a different PDF if this one is corrupted
                        - Image-based (scanned) PDFs cannot be processed
                        - Password-protected PDFs are not supported
                        """)
                    
                    if temp_file.exists():
                        temp_file.unlink()
    
    else:  # Web URL
        url = st.text_input(
            "Enter web page URL",
            placeholder="https://example.com/article",
            help="Only single web page (not entire website)"
        )
        
        # Ontology Mode Selection
        ontology_mode = st.radio(
            "Ontology Mode",
            ["Structured (Recommended)", "Flat (Legacy FAQ)"],
            help="Structured: Creates rich hierarchy and relationships (2-step process). Flat: Creates simple classes only."
        )
        
        mode_value = "structured" if "Structured" in ontology_mode else "flat"
        
        if url and st.button("🤖 Generate Ontology", type="primary"):
            with st.spinner(f"Analyzing web page with {model_name} ({mode_value} mode)..."):
                try:
                    # Initialize generator
                    if use_ollama:
                        generator = OntologyGenerator(
                            use_ollama=True,
                            model=model_name,
                            ollama_base_url=f"{CONFIG.get('OLLAMA_BASE_URL', 'http://localhost:11434')}/v1",
                            ontology_mode=mode_value
                        )
                    else:
                        generator = OntologyGenerator(
                            api_key=api_key,
                            model=model_name,
                            base_url=base_url,
                            ontology_mode=mode_value
                        )
                    
                    generated_result = generator.process_url(
                        url,
                        domain=domain or "general"
                    )
                    
                    # Store schema if available (for structured mode)
                    if hasattr(generator, 'last_discovered_schema'):
                        generated_result['discovered_schema'] = generator.last_discovered_schema
                    
                    # Store in session
                    st.session_state.generated_owl = generated_result
                    
                    st.success("✅ Ontology generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Generation failed: {e}")
    
    # Display generated ontology if available
    if st.session_state.generated_owl:
        result = st.session_state.generated_owl
        
        st.markdown("---")
        st.markdown("### Step 2: Review Generated Ontology")
        
        # Validation results
        val = result['validation']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Classes", val['class_count'])
        with col2:
            st.metric("Properties", val['property_count'])
        with col3:
            if val['valid']:
                st.success("✅ Valid")
            else:
                st.warning("⚠️ Has Issues")
        
        if val['issues']:
            with st.expander("⚠️ Validation Issues"):
                for issue in val['issues']:
                    st.warning(issue)
        
        # OWL content editor
        st.markdown("#### Edit OWL Content")
        st.caption("You can edit the ontology below before saving")
        
        edited_owl = st.text_area(
            "OWL/XML Content",
            value=result['owl_content'],
            height=400,
            help="Edit the generated OWL if needed"
        )
        
        # Update if edited
        if edited_owl != result['owl_content']:
            st.session_state.generated_owl['owl_content'] = edited_owl
            st.info("💾 Content modified (not saved yet)")
        
        # Action buttons
        st.markdown("---")
        st.markdown("### Step 3: Save or Process")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Download for Protégé
            st.download_button(
                label="📥 Download OWL",
                data=edited_owl,
                file_name=f"generated_ontology_{datetime.now().strftime('%Y%m%d_%H%M%S')}.owl",
                mime="application/rdf+xml",
                help="Download to validate in Protégé"
            )
        
        with col2:
            # Save and process
            if st.button("💾 Save & Process", type="primary"):
                try:
                    # Save to temp file
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    owl_file = Path(f"generated_ontology_{timestamp}.owl")
                    
                    with open(owl_file, 'w', encoding='utf-8') as f:
                        f.write(edited_owl)
                    
                    # Add to manager
                    source_name = result.get('source_file', result.get('source_url', 'generated'))
                    ontology_id = st.session_state.manager.add_ontology(
                        str(owl_file),
                        name=f"Generated from {source_name}"
                    )
                    
                    owl_file.unlink()  # Clean up temp file
                    
                    # Save discovered schema if available
                    if 'discovered_schema' in result:
                        onto_info = st.session_state.manager.get_ontology(ontology_id)
                        schema_file = Path(onto_info['owl_file']).parent / "schema_discovered.json"
                        with open(schema_file, 'w', encoding='utf-8') as f:
                            json.dump(result['discovered_schema'], f, indent=2, ensure_ascii=False)
                        print(f"   ✓ Schema saved to {schema_file}")
                    
                    st.session_state.current_ontology_id = ontology_id
                    st.session_state.generated_owl = None  # Clear generated
                    st.session_state.page = 'upload'  # Switch to upload page
                    
                    # Start processing
                    onto_info = st.session_state.manager.get_ontology(ontology_id)
                    owl_file_path = onto_info['owl_file']
                    
                    thread = threading.Thread(
                        target=process_ontology,
                        args=(ontology_id, owl_file_path, st.session_state.manager)
                    )
                    thread.daemon = True
                    thread.start()
                    
                    st.success(f"✅ Saved! Processing ontology...")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Failed to save: {e}")
        
        with col3:
            if st.button("🔄 Regenerate"):
                st.session_state.generated_owl = None
                st.rerun()
        
        with col4:
            if st.button("❌ Cancel"):
                st.session_state.generated_owl = None
                st.session_state.page = 'upload'
                st.rerun()


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
                3. **Embedding Generation**: Converting text to vectors using Ollama (100% local, no data sent to internet)
                
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
            
            # Get configuration from CONFIG
            use_ollama = CONFIG.get('USE_OLLAMA', True)
            ollama_model = CONFIG.get('OLLAMA_MODEL', 'llama3.3:70b')
            ollama_base_url = CONFIG.get('OLLAMA_BASE_URL', 'http://localhost:11434')
            
            # API configuration (for MegaLLM, OpenAI, etc.)
            api_key = CONFIG.get('openai_api_key')
            api_model = CONFIG.get('openai_model', 'gpt-4')
            api_base_url = CONFIG.get('openai_base_url')
            
            try:
                st.session_state.query_engine = GenericQueryEngine(
                    str(parsed_dir),
                    use_ollama=use_ollama,
                    ollama_model=ollama_model,
                    ollama_base_url=f"{ollama_base_url}/v1",
                    api_key=api_key,
                    api_model=api_model,
                    api_base_url=api_base_url
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
                result = engine.query(prompt, top_k=20, use_llm=True)
            
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
        
        # Page navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Choose action:",
            ["📤 Upload Ontology", "🤖 Generate Ontology"],
            key="page_nav"
        )
        
        if page == "📤 Upload Ontology":
            st.session_state.page = 'upload'
        else:
            st.session_state.page = 'generate'
        
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
        - Generate from documents
        - Auto-build knowledge graph
        - Chat with AI
        """)
    
    # Main content
    if st.session_state.current_ontology_id:
        chat_page()
    elif st.session_state.page == 'generate':
        generate_ontology_page()
    else:
        upload_page()


if __name__ == "__main__":
    main()
