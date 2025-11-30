# Ontology Q&A System

Auto-generate and query OWL ontologies using LLM.

## Setup Options

Choose one of three setups:

### Option 1: Cloud API (Recommended for Quick Start)

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/ontology-generator-api
pip install -r requirements.txt

# Setup API
cp api_keys.yaml.template api_keys.yaml
# Edit api_keys.yaml with your API key

streamlit run app.py
```

**API Configuration** (`api_keys.yaml`):
```yaml
openai_api_key: "sk-mega-YOUR_KEY"
openai_base_url: "https://ai.megallm.io/v1"
openai_model: "llama3.3-70b-instruct"
```

**Pros:** Fast setup, no hardware requirements  
**Cons:** Requires API key, costs $0.12-0.30/M tokens

---

### Option 2: Local Ollama - Development (Lightweight)

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/ontology-generator-api
pip install -r requirements.txt

# Auto-setup lightweight models
./setup_ollama_dev.sh

streamlit run app.py
```

**Requirements:** 8GB RAM, 10GB disk  
**Models:** TinyLlama (600MB) - fast but lower quality  
**Pros:** Free, 100% private, works offline  
**Cons:** Lower quality responses

---

### Option 3: Local Ollama - Production (High Quality)

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/ontology-generator-api
pip install -r requirements.txt

# Auto-setup production models
./setup_ollama_production.sh

streamlit run app.py
```

**Requirements:** 64GB RAM, 50GB disk  
**Models:** Llama 3.3 70B (~40GB) - high quality  
**Pros:** Free, 100% private, GPT-4 comparable quality  
**Cons:** High hardware requirements, slower first run
