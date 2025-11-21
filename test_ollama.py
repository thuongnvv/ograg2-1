#!/usr/bin/env python3
"""
Test Ollama Setup

Verify that Ollama is properly installed and configured for OG-RAG
"""

import sys
import requests
from openai import OpenAI


def test_ollama_server():
    """Test if Ollama server is running"""
    print("🔍 Testing Ollama server connection...")
    
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        response.raise_for_status()
        version = response.json().get('version', 'unknown')
        print(f"✅ Ollama server is running (version: {version})")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        print("   Please run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False


def test_ollama_models():
    """Test if required models are available"""
    print("\n🔍 Checking Ollama models...")
    
    # Must have at least one LLM model
    llm_models = ['llama3.3', 'tinyllama', 'llama3.2', 'mistral', 'qwen2.5']
    
    # Must have embedding model
    required_embedding = 'nomic-embed-text'
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        installed = [model['name'].split(':')[0] for model in response.json().get('models', [])]
        
        # Check for at least one LLM
        has_llm = False
        llm_found = None
        for model in llm_models:
            if model in installed:
                has_llm = True
                llm_found = model
                quality = {
                    'llama3.3': '⭐⭐⭐⭐⭐ Production quality',
                    'llama3.2': '⭐⭐⭐⭐ Balanced',
                    'mistral': '⭐⭐⭐⭐ Balanced',
                    'tinyllama': '⭐⭐ Development only',
                    'qwen2.5': '⭐⭐⭐⭐ High quality'
                }.get(model, '')
                print(f"✅ {model} - LLM for chat - {quality}")
                break
        
        if not has_llm:
            print(f"❌ No LLM model found - Need one of: {', '.join(llm_models)}")
            print(f"   Quick fix: ollama pull tinyllama  (for dev)")
            print(f"   Or run: ./setup_ollama_dev.sh")
        
        # Check embedding model
        has_embedding = required_embedding in installed
        if has_embedding:
            print(f"✅ {required_embedding} - Embeddings for retrieval")
        else:
            print(f"❌ {required_embedding} - Embeddings for retrieval")
            print(f"   Run: ollama pull {required_embedding}")
        
        # Show other installed LLMs
        other_llms = [m for m in llm_models if m in installed and m != llm_found]
        if other_llms:
            print("\nAdditional LLMs available:")
            for model in other_llms:
                print(f"  • {model}")
        
        return has_llm and has_embedding
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False


def test_ollama_embedding():
    """Test embedding generation"""
    print("\n🔍 Testing embedding generation...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": "test embedding"
            },
            timeout=30
        )
        response.raise_for_status()
        embedding = response.json().get('embedding', [])
        
        if embedding:
            print(f"✅ Embeddings working (dimension: {len(embedding)})")
            return True
        else:
            print("❌ No embedding returned")
            return False
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False


def test_ollama_chat():
    """Test chat completion"""
    print("\n🔍 Testing chat completion...")
    
    try:
        client = OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1"
        )
        
        # Try to detect which model to use
        response_tags = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in response_tags.json().get('models', [])]
        
        # Prefer production models, but accept any LLM
        model_priority = ['llama3.3:70b', 'llama3.2', 'mistral', 'qwen2.5', 'tinyllama']
        test_model = None
        for model in model_priority:
            if model in models:
                test_model = model
                break
        
        # Also check for base model names without version
        if not test_model:
            for model in models:
                base_name = model.split(':')[0]
                if base_name in ['llama3', 'llama2', 'mistral', 'qwen', 'tinyllama']:
                    test_model = model
                    break
        
        if not test_model:
            print("❌ No compatible LLM model found")
            print("   Available models:", models)
            print("   Run: ollama pull tinyllama  (for quick test)")
            return False
        
        print(f"Using model: {test_model}")
        
        response = client.chat.completions.create(
            model=test_model,
            messages=[
                {"role": "user", "content": "Say 'hello' in one word"}
            ],
            max_tokens=10
        )
        
        answer = response.choices[0].message.content
        print(f"✅ Chat working - Response: {answer}")
        return True
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Ollama Setup Verification for OG-RAG")
    print("="*60)
    
    results = []
    
    # Test 1: Server
    results.append(("Server", test_ollama_server()))
    
    # Test 2: Models
    results.append(("Models", test_ollama_models()))
    
    # Test 3: Embeddings
    results.append(("Embeddings", test_ollama_embedding()))
    
    # Test 4: Chat
    results.append(("Chat", test_ollama_chat()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Ollama is ready for OG-RAG")
        print("\nYour configuration:")
        print("1. Ollama server: ✅ Running")
        print("2. LLM model: ✅ Ready")
        print("3. Embeddings: ✅ Ready")
        print("\nNext steps:")
        print("1. Verify api_keys.yaml has USE_OLLAMA: true")
        print("2. Run: streamlit run app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nQuick fixes:")
        print("1. Start Ollama: ollama serve")
        print("2. For production: ./setup_ollama_production.sh")
        print("3. For development: ./setup_ollama_dev.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
