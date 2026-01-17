#!/usr/bin/env python3
"""
Generate Q&A output for manual review (no LLM-as-Judge evaluation)
"""

import json
import time
import sys
import random
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from query_engine.multi_ontology_engine import MultiOntologyQueryEngine


def load_api_config():
    config_file = Path(__file__).parent.parent / "api_keys.yaml"
    with open(config_file) as f:
        return yaml.safe_load(f)


def run():
    print("=" * 80)
    print("Generate Q&A Output for Manual Review")
    print("=" * 80)
    
    base_dir = Path(__file__).parent.parent
    ontologies_dir = base_dir / "data" / "ontologies"
    ground_truth_file = base_dir / "evaluation" / "faq_ground_truth_complete.json"
    
    # Find ontologies
    ontology_dirs = []
    for item in ontologies_dir.iterdir():
        if item.is_dir() and (item / "parsed" / "ontology_metadata.json").exists():
            ontology_dirs.append(str(item / "parsed"))
    
    print(f"\n📂 Found {len(ontology_dirs)} ontologies")
    
    # Load ground truth
    with open(ground_truth_file) as f:
        dataset = json.load(f)
    print(f"📖 Loaded {len(dataset['questions'])} questions")
    
    # Sample 4 per category
    random.seed(42)
    questions_per_category = 4
    categories = {}
    for qa in dataset['questions']:
        cat = qa['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(qa)
    
    sampled = []
    for cat, qs in categories.items():
        sample = random.sample(qs, min(questions_per_category, len(qs)))
        sampled.extend(sample)
        print(f"   📌 {cat}: {len(sample)} questions")
    
    print(f"\n📊 Total: {len(sampled)} questions")
    
    # Load API config
    config = load_api_config()
    use_ollama = config.get('USE_OLLAMA', False)
    
    if use_ollama:
        engine_kwargs = {
            "use_ollama": True,
            "ollama_model": config.get('OLLAMA_MODEL'),
            "ollama_base_url": f"{config.get('OLLAMA_BASE_URL')}/v1"
        }
    else:
        engine_kwargs = {
            "use_ollama": False,
            "api_key": config.get('openai_api_key'),
            "api_model": config.get('openai_model'),
            "api_base_url": config.get('openai_base_url')
        }
    
    print(f"\n🚀 Initializing Multi-Ontology Engine...")
    engine = MultiOntologyQueryEngine(
        ontology_dirs=ontology_dirs,
        engine_type="hypergraph",
        **engine_kwargs
    )
    
    # Generate Q&A
    print(f"\n{'=' * 80}")
    print("Generating Answers...")
    print(f"{'=' * 80}\n")
    
    results = []
    
    for i, qa in enumerate(sampled, 1):
        q_id = qa['id']
        question = qa['question']
        reference = qa['reference_answer']
        category = qa['category']
        
        print(f"[{i}/{len(sampled)}] {category}: {question[:50]}...")
        
        try:
            result = engine.query(question, top_k=10, use_llm=True)
            answer = result['answer']
            num_facts = len(result['retrieved_facts'])
            
            print(f"  ✓ Retrieved {num_facts} facts")
            
            results.append({
                'id': q_id,
                'category': category,
                'question': question,
                'reference_answer': reference,
                'generated_answer': answer,
                'num_facts': num_facts
            })
            
            # Delay to avoid rate limit
            time.sleep(5)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'id': q_id,
                'category': category,
                'question': question,
                'reference_answer': reference,
                'generated_answer': f"ERROR: {e}",
                'num_facts': 0
            })
    
    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = base_dir / "evaluation" / "results" / f"qa_output_{timestamp}.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save as Markdown for easy reading
    md_file = base_dir / "evaluation" / "results" / f"qa_output_{timestamp}.md"
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Q&A Output for Manual Review\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total Questions: {len(results)}\n\n")
        f.write("---\n\n")
        
        current_cat = None
        for r in results:
            if r['category'] != current_cat:
                current_cat = r['category']
                f.write(f"## {current_cat}\n\n")
            
            f.write(f"### {r['id']}: {r['question']}\n\n")
            f.write(f"**Generated Answer:**\n\n{r['generated_answer']}\n\n")
            f.write(f"**Reference Answer:**\n\n{r['reference_answer']}\n\n")
            f.write("---\n\n")
    
    print(f"\n{'=' * 80}")
    print("Done!")
    print(f"{'=' * 80}")
    print(f"📄 JSON: {json_file}")
    print(f"📝 Markdown: {md_file}")


if __name__ == "__main__":
    run()
