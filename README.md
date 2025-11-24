# Ontology Q&A System

Auto-generate and query OWL ontologies using LLM.

## Setup

```bash
git clone <repository-url>
cd ograg2-1
git checkout feature/ontology-generator-api
pip install -r requirements.txt

# Copy and edit API config
cp api_keys.yaml.template api_keys.yaml

streamlit run app.py
```

## API Configuration

Edit `api_keys.yaml` with your API key:

```yaml
openai_api_key: "sk-mega-YOUR_KEY"
openai_base_url: "https://ai.megallm.io/v1"
openai_model: "llama3.3-70b-instruct"
```
