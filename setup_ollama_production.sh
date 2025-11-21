#!/bin/bash
# Ollama Setup Script - Production Version (High Quality Models)
# For production deployment with best quality results

set -e

echo "==========================================================="
echo "🚀 Ollama Setup - Production (Llama 3.3 70B + Embeddings)"
echo "==========================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check system requirements
echo "⚙️  Checking system requirements..."
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_DISK=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')

echo "RAM: ${TOTAL_RAM}GB"
echo "Available Disk: ${AVAILABLE_DISK}GB"

if [ "$TOTAL_RAM" -lt 48 ]; then
    echo -e "${YELLOW}⚠️  Warning: Llama 3.3 70B requires at least 48GB RAM${NC}"
    echo -e "${YELLOW}   Recommended: 64GB+ RAM for smooth operation${NC}"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ "$AVAILABLE_DISK" -lt 50 ]; then
    echo -e "${RED}✗ Insufficient disk space. Need at least 50GB${NC}"
    echo -e "${RED}  Llama 3.3 70B: ~40GB${NC}"
    exit 1
fi

# Check if Ollama is installed
echo ""
echo "📦 Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    
    # Detect OS and install
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo -e "${RED}Please install Homebrew first or download from https://ollama.com${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Unsupported OS. Please install manually from https://ollama.com${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Ollama installed${NC}"
else
    echo -e "${GREEN}✓ Ollama already installed${NC}"
fi

# Check if Ollama server is running
echo ""
echo "🔍 Checking Ollama server..."
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo -e "${YELLOW}Ollama server not running. Starting...${NC}"
    
    # Start Ollama in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for server to start
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama server started${NC}"
            break
        fi
        echo "Waiting for Ollama server... ($i/10)"
        sleep 1
    done
    
    if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo -e "${RED}Failed to start Ollama server${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ollama server is running${NC}"
fi

# Pull production-grade models
echo ""
echo "📥 Downloading production models..."
echo -e "${BLUE}This will download ~40GB of models. Please be patient...${NC}"
echo ""

# LLM Model - Llama 3.3 70B
echo "1/2 Downloading Llama 3.3 70B (~40GB) - High quality LLM..."
echo -e "${YELLOW}⏱  This may take 30-60 minutes depending on your internet speed${NC}"
ollama pull llama3.3:70b
echo -e "${GREEN}✓ Llama 3.3 70B downloaded${NC}"

# Embedding Model - nomic-embed-text
echo ""
echo "2/2 Downloading nomic-embed-text (274MB) - Embedding model..."
ollama pull nomic-embed-text
echo -e "${GREEN}✓ nomic-embed-text downloaded${NC}"

# Verify models
echo ""
echo "✅ Verifying installed models..."
ollama list

# Test models
echo ""
echo "🧪 Testing models..."

# Test LLM
echo ""
echo "Testing Llama 3.3 70B..."
echo -e "${YELLOW}⏱  First run may take a minute to load the model into memory...${NC}"
RESPONSE=$(ollama run llama3.3:70b "Say hello in one word" 2>&1 | head -n 1)
echo "Response: $RESPONSE"

# Test embeddings
echo ""
echo "Testing nomic-embed-text..."
if curl -s -X POST http://localhost:11434/api/embeddings \
    -d '{"model": "nomic-embed-text", "prompt": "test"}' \
    | grep -q "embedding"; then
    echo -e "${GREEN}✓ Embeddings working${NC}"
else
    echo -e "${RED}✗ Embeddings test failed${NC}"
fi

# Create api_keys.yaml
echo ""
echo "📝 Creating api_keys.yaml for PRODUCTION..."

# Backup existing file if present
if [ -f api_keys.yaml ]; then
    echo "Backing up existing api_keys.yaml to api_keys.yaml.bak"
    cp api_keys.yaml api_keys.yaml.bak
fi

cat > api_keys.yaml << 'EOF'
# Ollama Configuration - Production Setup
# Using high-quality models for best results

# Enable Ollama (local LLM - 100% private)
USE_OLLAMA: true

# LLM Model for chat/Q&A
OLLAMA_MODEL: 'llama3.3:70b'  # High-quality 70B parameter model

# Embedding Model
OLLAMA_EMBED_MODEL: 'nomic-embed-text'

# Ollama Server URL
OLLAMA_BASE_URL: 'http://localhost:11434'

# ============================================
# Benefits of this setup:
# ✅ 100% Data Privacy - Nothing leaves your machine
# ✅ No API costs - Completely free
# ✅ Offline operation - No internet required
# ✅ High quality results - 70B parameter model
# ============================================
EOF

echo -e "${GREEN}✓ api_keys.yaml created${NC}"
echo ""
echo "Configuration:"
cat api_keys.yaml | grep -E "OLLAMA_MODEL|OLLAMA_EMBED_MODEL" | sed 's/^/  /'

# Performance tips
echo ""
echo "==========================================================="
echo "✨ Setup Complete - Production Configuration"
echo "==========================================================="
echo ""
echo "Installed Models:"
echo "  • Llama 3.3 70B (~40GB) - State-of-the-art LLM"
echo "  • nomic-embed-text (274MB) - High-quality embeddings"
echo ""
echo "Total Size: ~40GB"
echo ""
echo "🚀 Performance Tips:"
echo "  • First query will be slow (loading model into RAM)"
echo "  • Subsequent queries will be much faster"
echo "  • GPU acceleration automatic if available"
echo "  • Close other applications for best performance"
echo ""
echo "Next Steps:"
echo "  1. Test setup: python test_ollama.py"
echo "  2. Run application: streamlit run app.py"
echo ""
echo "📊 Expected Performance:"
echo "  • Query time: 10-30 seconds (first), 5-15s (subsequent)"
echo "  • Quality: Comparable to GPT-4"
echo "  • Privacy: 100% local, no data leaves your machine"
echo ""
echo "==========================================================="
