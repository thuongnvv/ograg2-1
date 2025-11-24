#!/usr/bin/env python3
"""
Test MegaLLM API - Simple chat test
"""

from openai import OpenAI

# MegaLLM configuration
API_KEY = "sk-mega-cfeefed3f8e0fc99bb83d0026d631532342a1c6543a782433c262d8248506399"
BASE_URL = "https://ai.megallm.io/v1"  # Correct endpoint from official docs
MODEL = "llama3.3-70b-instruct"

print("="*60)
print("Testing MegaLLM API")
print("="*60)

# Initialize client
print(f"\n1. Connecting to {BASE_URL}")
print(f"   Model: {MODEL}")

# Add browser-like headers to bypass Cloudflare
extra_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers=extra_headers
)

# Test simple chat
print("\n2. Sending test message...")

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Say 'Hello from MegaLLM!' and nothing else."}
        ],
        max_tokens=50,
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    print(f"\n✅ Success! Response:")
    print(f"   {answer}")
    
    print(f"\n✅ MegaLLM API is working!")
    print(f"   Total tokens: {response.usage.total_tokens}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\nFull error details:")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
