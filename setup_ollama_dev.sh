#!/bin/bash
# Ollama Setup Script - Development Version (Lightweight Models)
# For testing and development on limited hardware

set -e

echo "==============================================="
echo "🔧 Ollama Setup - Development (Lightweight)"
echo "==============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Ollama is installed
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

# Pull lightweight models for development
echo ""
echo "📥 Downloading lightweight models for development..."
echo ""

# LLM Model - TinyLlama (smallest, fastest)
echo "1/2 Downloading TinyLlama (600MB) - Smallest model for testing..."
ollama pull tinyllama
echo -e "${GREEN}✓ TinyLlama downloaded${NC}"

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
echo "Testing TinyLlama..."
RESPONSE=$(ollama run tinyllama "Say hello in one word" 2>&1 | head -n 1)
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
echo "📝 Creating api_keys.yaml for DEVELOPMENT..."

# Backup existing file if present
if [ -f api_keys.yaml ]; then
    echo "Backing up existing api_keys.yaml to api_keys.yaml.bak"
    cp api_keys.yaml api_keys.yaml.bak
fi

cat > api_keys.yaml << 'EOF'
# Ollama Configuration - Development Setup
# Using lightweight models for testing

# Enable Ollama (local LLM)
USE_OLLAMA: true

# LLM Model for chat/Q&A
OLLAMA_MODEL: 'tinyllama'  # Lightweight model for development

# Embedding Model
OLLAMA_EMBED_MODEL: 'nomic-embed-text'

# Ollama Server URL
OLLAMA_BASE_URL: 'http://localhost:11434'

# Note: No API keys needed for Ollama - everything runs locally!
EOF

echo -e "${GREEN}✓ api_keys.yaml created${NC}"
echo ""
echo "Configuration:"
cat api_keys.yaml | grep -E "OLLAMA_MODEL|OLLAMA_EMBED_MODEL" | sed 's/^/  /'

# Summary
echo ""
echo "==============================================="
echo "✨ Setup Complete - Development Configuration"
echo "==============================================="
echo ""
echo "Installed Models:"
echo "  • TinyLlama (600MB) - Fast, lightweight LLM"
echo "  • nomic-embed-text (274MB) - Embeddings"
echo ""
echo "Total Size: ~900MB"
echo ""
echo "Next Steps:"
echo "  1. Test Ollama: python test_ollama.py"
echo "  2. Run application: streamlit run app.py"
echo ""
echo "⚠️  Note: TinyLlama is for testing only."
echo "   For production, use: ./setup_ollama_production.sh"
echo ""
echo "==============================================="
