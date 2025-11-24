# Ontology Generator - Quick Start

## Setup API Key (Required)

**MegaLLM/Kindo.ai is blocked by Cloudflare when called from Python.**

Choose one of these **FREE** options:

### Option 1: OpenRouter (Recommended)

**Free $5 credits, access to many models**

1. Sign up: https://openrouter.ai/
2. Get API key from dashboard
3. Edit `api_keys.yaml`:
```yaml
openai_api_key: "sk-or-v1-your-actual-key-here"
openai_base_url: "https://openrouter.ai/api/v1"
openai_model: "meta-llama/llama-3.3-70b-instruct"
```

### Option 2: Groq

**100% Free, very fast inference**

1. Sign up: https://console.groq.com/
2. Get API key from dashboard  
3. Edit `api_keys.yaml`:
```yaml
openai_api_key: "gsk_your-actual-key-here"
openai_base_url: "https://api.groq.com/openai/v1"
openai_model: "llama-3.3-70b-versatile"
```

Both options work perfectly and are completely free!

## Test Generation

```bash
python test_generator.py
```

## Use in Web UI

```bash
streamlit run app.py
```

Then:
1. Click "🤖 Generate Ontology" in sidebar
2. Upload PDF/DOCX or enter URL
3. Add domain context (optional)
4. Click "Generate"
5. Review/edit OWL
6. Download or process directly

## CLI Usage

```bash
# From file
python ontology_generator.py --file document.pdf --domain "biology"

# From URL
python ontology_generator.py --url https://example.com --domain "medicine"

# Use Ollama instead (requires llama3.3:70b)
python ontology_generator.py --file doc.pdf --ollama --model llama3.3:70b
```

## Branch Info

- `main`: 100% local with Ollama (no internet after model download)
- `feature/ontology-generator-api`: Uses OpenAI GPT-4 for ontology generation

The API branch uses GPT-4 because generating valid OWL requires large, capable models. Small local models struggle with OWL/XML structure.

## Switch Back to Main

```bash
git checkout main
```
