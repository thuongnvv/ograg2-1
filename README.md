# Ontology Q&A System

Auto-generate and query OWL ontologies using LLM.

## Setup Options

Choose one of three setups:

### Option 1: Cloud API

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/faq-universal-support

# Run setup script
./setup_api.sh
```

Then edit `api_keys.yaml` with your API key and run:
```bash
streamlit run app.py
```

**Requirements:** Python 3.8+  
**Models:** Cloud-based (MegaLLM/Groq/OpenAI)  
**Cost:** $0.12-0.30/M tokens

---

### Option 2: Local Ollama - Development

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/faq-universal-support

# Run setup script
./setup_ollama_dev.sh
```

Then run:
```bash
streamlit run app.py
```

**Requirements:** 8GB RAM, 10GB disk  
**Models:** TinyLlama (600MB)  
**Cost:** Free

---

### Option 3: Local Ollama - Production

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/faq-universal-support

# Run setup script
./setup_ollama_production.sh
```

Then run:
```bash
streamlit run app.py
```

**Requirements:** 64GB RAM, 50GB disk  
**Models:** Llama 3.3 70B (~40GB)  
**Cost:** Free
