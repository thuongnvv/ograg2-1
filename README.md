# Ontology Q&A System - Local Deployment

**100% local processing. No data transmitted externally.**

---
## Hardware Requirements

### Minimum (Just for testing, in this case, LLM is very bad)
*Target: TinyLlama or Llama 3.2 3B*
- **RAM**: 8GB+
- **GPU**: Optional (Runs on CPU or integrated graphics)
- **Disk**: 10GB+ free space

### Recommended (For high-quality)
*Target: Llama 3.3 70B *
- **Nvidia GPU**: Dual RTX 3090/4090 (48GB VRAM total) or 1x A6000/A100.
- **Apple Silicon**: Mac Studio/MacBook Pro with **64GB+ Unified Memory** (M1/M2/M3 Max or Ultra).
- **CPU Only (Not Recommended)**
- **Disk**: 100GB+ free space (Model requires ~43GB, plus space for Vector DB and OS).

### Software
- Python 3.8+
- Ollama (Latest version)
- Git

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd ograg2-1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup local LLM (one-time, ~1-40GB download)
./setup_ollama_dev.sh          # For fast testing, in this case, LLM is very bad (1GB)
# OR
./setup_ollama_production.sh   # High quality (40GB)

# 4. Verify installation
python test_ollama.py

# 5. Run application
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501` to upload your OWL file and test with your private data.

---

## Key Points

- **Data Privacy**: All processing runs locally on your infrastructure
- **No Internet**: Only needed for initial model download, then fully offline
- **Your Data**: Test with your actual proprietary ontologies
- **Air-gapped**: Compatible with isolated environments after setup

---

## Configuration

Edit `api_keys.yaml`:
```yaml
USE_OLLAMA: true
OLLAMA_MODEL: "llama3.3:70b"              # High quality
OLLAMA_EMBEDDING_MODEL: "nomic-embed-text"
OLLAMA_BASE_URL: "http://localhost:11434"
```

Models:
- `tinyllama` (600MB) - Testing
- `llama3.3:70b` (40GB) - Production

---

## Requirements

- Python 3.8+
- 8GB RAM (for fast testing) or 64GB RAM (high-quality)
- 50GB disk space

---

## Support

Run verification tests:
```bash
python test_ollama.py        # Test local LLM
```

Check logs if issues occur.
