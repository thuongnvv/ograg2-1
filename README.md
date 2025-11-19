# 🧬 Universal Ontology Q&A System

**Upload any OWL ontology → Chat with AI**

A production-ready web application that transforms any OWL ontology into an intelligent Q&A system using advanced retrieval-augmented generation (RAG) techniques.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_dependencies.py
```

### 3. Set API Key (Optional)
```bash
cp api_keys.yaml.template api_keys.yaml
# Edit api_keys.yaml and add your OpenAI API key
```

### 4. Launch Application
```bash
streamlit run app.py
```

### 5. Use the System
1. Open browser at `http://localhost:8501`
2. Upload any OWL ontology file
3. Wait for automatic processing (2-15 minutes)
4. Start asking questions!

---

## ✨ Key Features

- 📤 **Universal**: Upload any OWL/OBO ontology file
- 🚀 **Zero Configuration**: Automatic parsing and processing
- 🤖 **AI-Powered**: Smart retrieval + LLM answer generation
- 🎯 **Accurate**: Ontology-grounded responses (no hallucination)
- 📊 **Multi-Ontology**: Manage multiple ontologies simultaneously
- 🔍 **Smart Context**: Hierarchical expansion (5 main + 10 parent terms)

---

## 📖 Example Use Cases

### Biological Research
- Upload **Gene Ontology** → Ask "What is DNA repair?"
- Upload **ChEBI** → Ask "What is glucose?"
- Upload **Human Phenotype Ontology** → Ask "What are symptoms of diabetes?"

### Medical Domain
- Upload **Disease Ontology** → Ask about disease classifications
- Upload **SNOMED CT** → Ask about medical procedures

### Any Domain
- Upload **Environmental Ontology** → Ask about ecological terms
- Upload **Chemical Ontologies** → Ask about molecular structures

---

## 🏗️ System Architecture

```
Input: OWL File
     ↓
Parse: XML → Structured JSON (85-90% information retention)
     ↓
Build: Smart Chunking (4 types) → Hypergraph → Embeddings
     ↓
Query: Dual Ranking → Hierarchical Expansion → LLM Generation
     ↓
Output: Accurate, Grounded Answers
```

### Smart Chunking Strategy
Each ontology term is intelligently split into:
- **Core chunk**: ID, label, definition (highest priority)
- **Synonyms chunk**: Alternative names and terms
- **Relationships chunk**: Parent/child relationships and hierarchies  
- **Details chunk**: Examples, comments, and additional metadata

### Enhanced Retrieval Logic
- **Top-5 main terms**: Most relevant to user query
- **Up to 10 parent terms**: 2 parents per main term for hierarchical context
- **Total context**: Maximum 15 terms (5 + 5×2) for comprehensive understanding

---

## 📊 Performance

| Ontology Size | Processing Time | Query Time | Context Size |
|---------------|----------------|------------|--------------|
| Small (~1K terms) | 2-3 minutes | 2-3 seconds | 5-15 terms |
| Medium (~10K terms) | 5-10 minutes | 3-5 seconds | 5-15 terms |
| Large (~50K terms) | 10-15 minutes | 3-6 seconds | 5-15 terms |

---

## 🔧 Configuration

### API Keys
The system works in two modes:
1. **With LLM API key**: Full Q&A with generated answers
2. **Without API key**: Retrieval-only mode (returns relevant ontology terms)

Supported LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- MegaLLM (Llama models)

### Embedding Models
Default: `sentence-transformers/all-MiniLM-L6-v2` (fast, good quality)

---

## 📁 Project Structure

```
├── app.py                          # Main web interface
├── ontology_manager.py             # Multi-ontology management
├── build_hypergraph.py             # Generic hypergraph builder
├── scripts/parse_owl.py            # Universal OWL parser
├── query_engine/generic_query_engine.py  # Query processing
├── requirements.txt                # Complete dependencies
├── api_keys.yaml.template         # API configuration
└── data/ontologies/               # Ontology workspace
```

---

## 🔍 Troubleshooting

### Common Issues

**Processing stuck at "PARSING"**
- Check OWL file format (must be valid XML)
- Ensure file is OBO-formatted OWL

**Out of memory during build**
- Use smaller embedding model
- Reduce batch size in `build_hypergraph.py`

**No relevant answers**
- Try rephrasing your question
- Check if ontology contains relevant terms
- Verify ontology was processed successfully

**Only getting retrieval results (no LLM answers)**
- Check API key configuration in `api_keys.yaml`
- Verify API key is valid and has credit

---

## 🎯 Technical Specifications

| Metric | Value |
|--------|-------|
| Total Size | 268KB (without dependencies) |
| Core Dependencies | 6 packages |
| Processing Memory | ~1GB during build |
| Information Retention | 85-90% of ontology content |
| Supported Formats | OWL, OBO |
| Query Response Time | 3-6 seconds |

---

## 📄 License

MIT License - See LICENSE file for details

---

**Ready to transform your domain knowledge into an intelligent Q&A system!** 🚀