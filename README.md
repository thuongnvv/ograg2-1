# Ontology Q&A System

RAG-based chat system for OWL ontologies with auto-generation capability.

## Quick Start

```bash
# 1. Clone & install
git clone <repository-url>
cd ograg2-1
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Open `http://localhost:8501`

## Features

- **Auto-generate**: Create ontologies from PDF/DOCX/URLs using LLM
- **Upload**: Process existing OWL files
- **Chat**: Query ontologies with natural language
- **Privacy**: Works with cloud API (MegaLLM) or local Ollama

## Configuration

`api_keys.yaml` (optional):

```yaml
# Cloud API (default, embedded MegaLLM key)
openai_api_key: "your-key"
openai_base_url: "https://ai.megallm.io/v1"
openai_model: "llama3.3-70b-instruct"

# OR Local Ollama
USE_OLLAMA: true
OLLAMA_MODEL: "llama3.3:70b"
OLLAMA_BASE_URL: "http://localhost:11434"
```

## Local Setup (Optional)

For 100% offline deployment:

```bash
# Install Ollama
./setup_ollama_production.sh    # 40GB download

# Test
python test_ollama.py
```

**Requirements**: 64GB RAM, 50GB disk for local deployment

## Documentation

- [Ontology Generator](ONTOLOGY_GENERATOR.md) - Auto-generate OWL from documents
