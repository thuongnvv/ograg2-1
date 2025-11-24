# Ontology Q&A System

Auto-generate and query OWL ontologies using LLM.

## Setup

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/ontology-generator-api
pip install -r requirements.txt

# Add API key
cp api_keys.yaml.template api_keys.yaml
nano api_keys.yaml  # Add your MegaLLM/Groq/OpenAI key

streamlit run app.py
```

## API Configuration

Edit `api_keys.yaml`:

```yaml
# MegaLLM (recommended)
openai_api_key: "sk-mega-YOUR_KEY"
openai_base_url: "https://ai.megallm.io/v1"
openai_model: "llama3.3-70b-instruct"
```

Options: [MegaLLM](https://megallm.io), [Groq](https://groq.com) (free), OpenAI, or local Ollama (see `ONTOLOGY_GENERATOR.md`).
