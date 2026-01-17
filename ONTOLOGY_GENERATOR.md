# Ontology Generator

Auto-generate OWL ontologies from documents or URLs using LLM.

## Quick Start

```bash
streamlit run app.py
```

Click **"🤖 Generate Ontology"** in sidebar → Upload file or enter URL → Generate

## Features

- **Input**: PDF, DOCX, or web URLs
- **LLM**: MegaLLM Llama 3.3 70B (embedded, no setup)
- **Output**: Valid OWL/XML ontology
- **Workflow**: Generate → Edit → Download → Process → Chat

## Custom API (Optional)

Edit `api_keys.yaml` to use other providers:

```yaml
openai_api_key: "your-key"
openai_base_url: "https://api.provider.com/v1"
openai_model: "model-name"
```

## CLI Usage

```bash
# From file
python ontology_generator.py --file document.pdf --domain "biology"

# From URL
python ontology_generator.py --url https://example.com --domain "medicine"
```

## Architecture

```
Document → Extract Text → LLM Analysis → OWL Generation → Validation → Integration
```

- **Extraction**: pypdf, python-docx, beautifulsoup4
- **Generation**: MegaLLM API (temperature=0.3, max_tokens=16000)
- **Validation**: XML parsing + OWL structure checks
- **Integration**: Seamless with existing hypergraph pipeline
