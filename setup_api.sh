#!/bin/bash
# API Setup Script - Cloud-based LLM (MegaLLM/Groq/OpenAI)
# For quick start without local hardware requirements

set -e

echo "==============================================="
echo "☁️  API Setup - Cloud LLM"
echo "==============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python installation
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION installed${NC}"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install Playwright browsers
echo ""
echo "🌐 Installing Playwright browsers..."
playwright install chromium
echo -e "${GREEN}✓ Playwright chromium installed${NC}"

# Create api_keys.yaml from template
echo ""
echo "📝 Creating api_keys.yaml..."

if [ -f api_keys.yaml ]; then
    echo -e "${YELLOW}⚠️  api_keys.yaml already exists${NC}"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing api_keys.yaml"
    else
        cp api_keys.yaml.template api_keys.yaml
        echo -e "${GREEN}✓ api_keys.yaml created from template${NC}"
    fi
else
    cp api_keys.yaml.template api_keys.yaml
    echo -e "${GREEN}✓ api_keys.yaml created from template${NC}"
fi

# Instructions
echo ""
echo "==============================================="
echo "✨ Setup Complete - API Configuration"
echo "==============================================="
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Get an API key from one of these providers:"
echo "   • MegaLLM: https://megallm.io"
echo "   • Groq: https://console.groq.com/"
echo "   • OpenAI: https://platform.openai.com/"
echo ""
echo "2. Edit api_keys.yaml:"
echo "   nano api_keys.yaml"
echo ""
echo "   Add your API key:"
echo "   openai_api_key: \"YOUR_API_KEY\""
echo "   openai_base_url: \"https://ai.megallm.io/v1\""
echo "   openai_model: \"llama3.3-70b-instruct\""
echo ""
echo "3. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "==============================================="
echo ""
echo "💰 Cost Estimate (MegaLLM Llama 3.3 70B):"
echo "   Input:  \$0.12/M tokens"
echo "   Output: \$0.30/M tokens"
echo ""
echo "✅ Benefits:"
echo "   • No hardware requirements"
echo "   • Fast setup (5 minutes)"
echo "   • High quality results"
echo "   • Works on any computer"
echo ""
echo "==============================================="
