#!/usr/bin/env python3
"""
Test Ontology Generator with OpenAI API

Usage:
    export OPENAI_API_KEY='your-key-here'
    python test_generator.py
"""

from ontology_generator import OntologyGenerator
from pathlib import Path
import os
import yaml

def test_generator():
    """Test ontology generation with GPT-4"""
    print("="*60)
    print("Testing Ontology Generator (GPT-4)")
    print("="*60)
    
    # Check API key from multiple sources
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = None
    model = "gpt-4"
    
    if not api_key:
        # Try api_keys.yaml
        api_keys_file = Path("api_keys.yaml")
        if api_keys_file.exists():
            with open(api_keys_file, 'r') as f:
                api_keys = yaml.safe_load(f)
                api_key = api_keys.get('openai_api_key')
                base_url = api_keys.get('openai_base_url')
                model = api_keys.get('openai_model', 'gpt-4')
    
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        print("❌ Error: OpenAI API key not configured")
        print("\nPlease either:")
        print("  1. Set environment: export OPENAI_API_KEY='your-key'")
        print("  2. Edit api_keys.yaml with your key")
        print("\nGet API key from: https://platform.openai.com/api-keys")
        return False
    
    # Initialize generator
    print("\n1. Initializing generator...")
    print(f"   Model: {model}")
    if base_url:
        print(f"   Base URL: {base_url}")
    generator = OntologyGenerator(
        api_key=api_key,
        model=model,
        base_url=base_url
    )
    print("✅ Generator initialized")
    
    # Test with simple text
    print("\n2. Testing with sample text...")
    sample_text = """
    Machine Learning is a subset of Artificial Intelligence. 
    It includes techniques like Neural Networks and Decision Trees.
    Deep Learning is a specialized form of Machine Learning that uses multiple layers.
    Supervised Learning and Unsupervised Learning are two main categories.
    """
    
    try:
        owl_content = generator.generate_ontology(sample_text, domain="computer science")
        print(f"✅ Generated OWL ({len(owl_content)} chars)")
        
        # Validate
        validation = generator.validate_owl(owl_content)
        print(f"\n3. Validation results:")
        print(f"   Valid: {validation['valid']}")
        print(f"   Classes: {validation['class_count']}")
        print(f"   Properties: {validation['property_count']}")
        
        if validation['issues']:
            print(f"   Issues:")
            for issue in validation['issues']:
                print(f"     - {issue}")
        
        # Save sample
        output_file = Path("test_generated.owl")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(owl_content)
        print(f"\n✅ Saved to: {output_file}")
        print("\nYou can now:")
        print(f"  1. Open {output_file} in Protégé to validate")
        print(f"  2. Review the generated ontology structure")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_generator()
    exit(0 if success else 1)
