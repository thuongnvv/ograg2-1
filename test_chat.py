#!/usr/bin/env python3
"""Quick test chat with Llama via Ollama"""

from openai import OpenAI

# Connect to Ollama
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

print("🤖 Chat with Llama (type 'quit' to exit)\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    if not query:
        continue
    
    # Send to Llama
    response = client.chat.completions.create(
        model="tinyllama",  # or "llama3.3:70b"
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    print(f"\nLlama: {answer}\n")
