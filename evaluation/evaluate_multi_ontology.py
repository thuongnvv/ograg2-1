#!/usr/bin/env python3
"""
Evaluate Multi-Ontology Engine on Creative Commons FAQ

This script evaluates the Multi-Ontology Query Engine using:
- 120 ground truth Q&A pairs from Creative Commons FAQ
- LLM-as-Judge metrics: Faithfulness, Correctness, Completeness
- Performance metrics: Latency, retrieval stats
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_engine.multi_ontology_engine import MultiOntologyQueryEngine
from openai import OpenAI


def load_api_config():
    """Load API configuration from api_keys.yaml"""
    config_file = Path(__file__).parent.parent / "api_keys.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found: {config_file}")


class FAQEvaluator:
    """Evaluator for RAG system using LLM-as-Judge"""
    
    # Rate limiting: delay between API calls
    API_DELAY_SECONDS = 10  # 10 seconds to avoid rate limits
    MAX_RETRIES = 3
    
    def __init__(self, llm_client, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
        self.last_api_call = 0
    
    def _rate_limit(self):
        """Ensure minimum delay between API calls"""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.API_DELAY_SECONDS:
            time.sleep(self.API_DELAY_SECONDS - elapsed)
        self.last_api_call = time.time()
    
    def _call_llm_with_retry(self, messages: list, max_tokens: int = 300, debug: bool = False) -> str:
        """Call LLM with retry and rate limiting"""
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                
                # DEBUG: Print raw LLM response
                if debug:
                    print(f"    [DEBUG] Raw LLM response ({len(content) if content else 0} chars):")
                    print(f"    >>> {content[:300] if content else 'None'}{'...' if content and len(content) > 300 else ''}")
                
                return content if content else ""
            except Exception as e:
                if "429" in str(e):
                    # Rate limit - wait longer
                    wait_time = (attempt + 1) * 5
                    print(f"    ⏳ Rate limit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(2)
                    else:
                        raise e
        return ""
    
    def _parse_json_response(self, response_text: str) -> dict:
        """Robustly parse JSON from LLM response"""
        import re
        
        if not response_text:
            return None
        
        # Try direct parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try extracting JSON object with regex
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict:
        """Evaluate if answer is faithful to retrieved context"""
        # Truncate context aggressively
        if len(context) > 2000:
            context = context[:2000] + "..."
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        prompt = f"""Rate how well the answer uses the retrieved context.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Rating guide (be generous):
- 5: Answer is well-grounded in context
- 4: Mostly based on context with acceptable additions
- 3: Uses context but adds helpful general knowledge
- 2: Some context used but mostly external knowledge
- 1: Barely uses context
- 0: Contradicts or ignores context

JSON only: {{"faithfulness_score": <0-5>, "reasoning": "brief"}}"""

        try:
            messages = [
                {"role": "system", "content": "Output only valid JSON, no markdown."},
                {"role": "user", "content": prompt}
            ]
            content = self._call_llm_with_retry(messages, max_tokens=300)
            result = self._parse_json_response(content)
            
            if result and 'faithfulness_score' in result:
                return result
            else:
                return {"faithfulness_score": -1, "hallucinated_claims": [], "reasoning": f"Parse failed"}
                
        except Exception as e:
            print(f"  ⚠️ Faithfulness eval error: {e}")
            return {"faithfulness_score": -1, "hallucinated_claims": [], "reasoning": f"Error: {e}"}
    
    def evaluate_correctness(self, question: str, answer: str, reference: str) -> Dict:
        """Evaluate correctness against reference answer"""
        if len(reference) > 2000:
            reference = reference[:2000] + "..."
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        prompt = f"""Compare the answer with the official reference.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Rating guide (focus on key ideas, not exact wording):
- 5: Captures main ideas correctly
- 4: Mostly correct with minor differences
- 3: Correct on key points
- 2: Partially correct
- 1: Mostly incorrect
- 0: Wrong or off-topic

JSON only: {{"correctness_score": <0-5>, "reasoning": "brief"}}"""

        try:
            messages = [
                {"role": "system", "content": "Output only valid JSON, no markdown."},
                {"role": "user", "content": prompt}
            ]
            content = self._call_llm_with_retry(messages, max_tokens=300)
            result = self._parse_json_response(content)
            
            if result and 'correctness_score' in result:
                return result
            else:
                return {"correctness_score": -1, "missing_info": "", "reasoning": "Parse failed"}
                
        except Exception as e:
            print(f"  ⚠️ Correctness eval error: {e}")
            return {"correctness_score": -1, "missing_info": "", "reasoning": f"Error: {e}"}
    
    def evaluate_completeness(self, question: str, answer: str, reference: str) -> Dict:
        """Evaluate completeness of answer"""
        if len(reference) > 1500:
            reference = reference[:1500] + "..."
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        prompt = f"""Rate if the answer covers the main points.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Rating guide (focus on main points, not every detail):
- 5: Covers all main points
- 4: Covers most main points
- 3: Covers key points
- 2: Missing important points
- 1: Very incomplete
- 0: Empty or irrelevant

JSON: {{"completeness_score": <0-5>, "reasoning": "brief"}}"""

        try:
            messages = [
                {"role": "system", "content": "Output only valid JSON, no markdown."},
                {"role": "user", "content": prompt}
            ]
            content = self._call_llm_with_retry(messages, max_tokens=200)
            result = self._parse_json_response(content)
            
            if result and 'completeness_score' in result:
                return result
            else:
                return {"completeness_score": -1, "reasoning": "Parse failed"}
                
        except Exception as e:
            print(f"  ⚠️ Completeness eval error: {e}")
            return {"completeness_score": -1, "reasoning": f"Error: {e}"}


def run_evaluation():
    """Run complete evaluation pipeline"""
    
    print("=" * 80)
    print("Multi-Ontology Engine Evaluation on Creative Commons FAQ")
    print("=" * 80)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    ontologies_dir = base_dir / "data" / "ontologies"
    ground_truth_file = base_dir / "evaluation" / "faq_ground_truth_complete.json"
    results_dir = base_dir / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parsed ontologies
    ontology_dirs = []
    for item in ontologies_dir.iterdir():
        if item.is_dir() and (item / "parsed" / "ontology_metadata.json").exists():
            ontology_dirs.append(str(item / "parsed"))
    
    print(f"\n📂 Found {len(ontology_dirs)} ontologies:")
    for i, onto_dir in enumerate(ontology_dirs, 1):
        meta_file = Path(onto_dir) / "ontology_metadata.json"
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"   {i}. {onto_dir.split('/')[-2]}: {meta['total_terms']} terms, {meta['hypergraph']['total_facts']} facts")
    
    # Load ground truth
    print(f"\n📖 Loading ground truth: {ground_truth_file}")
    with open(ground_truth_file) as f:
        dataset = json.load(f)
    
    total_questions = dataset['dataset_info']['total_questions']
    print(f"   Total questions: {total_questions}")
    
    # Load API config
    print(f"\n📡 Loading API configuration from api_keys.yaml...")
    config = load_api_config()
    use_ollama = config.get('USE_OLLAMA', False)
    
    if use_ollama:
        ollama_model = config.get('OLLAMA_MODEL', 'tinyllama')
        ollama_base_url = config.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        print(f"   Using Ollama: {ollama_model}")
        engine_kwargs = {
            "use_ollama": True,
            "ollama_model": ollama_model,
            "ollama_base_url": f"{ollama_base_url}/v1"
        }
        # Create separate LLM client for evaluator
        llm_client = OpenAI(api_key="ollama", base_url=f"{ollama_base_url}/v1")
        model_name = ollama_model
    else:
        api_key = config.get('openai_api_key')
        api_base_url = config.get('openai_base_url')
        api_model = config.get('openai_model', 'gpt-4')
        print(f"   Using API: {api_model} @ {api_base_url}")
        engine_kwargs = {
            "use_ollama": False,
            "api_key": api_key,
            "api_model": api_model,
            "api_base_url": api_base_url
        }
        # Create separate LLM client for evaluator
        llm_client = OpenAI(api_key=api_key, base_url=api_base_url)
        model_name = api_model
    
    # Initialize Multi-Ontology Engine
    print(f"\n🚀 Initializing Multi-Ontology Engine...")
    engine = MultiOntologyQueryEngine(
        ontology_dirs=ontology_dirs,
        engine_type="hypergraph",
        **engine_kwargs
    )
    
    # Initialize evaluator with same LLM config
    print(f"🤖 Initializing LLM-as-Judge evaluator...")
    evaluator = FAQEvaluator(
        llm_client=llm_client,
        model_name=model_name
    )
    
    # === SAMPLE 4 QUESTIONS PER CATEGORY ===
    import random
    random.seed(42)  # For reproducibility
    
    questions_per_category = 4
    categories_in_data = {}
    
    for qa in dataset['questions']:
        cat = qa['category']
        if cat not in categories_in_data:
            categories_in_data[cat] = []
        categories_in_data[cat].append(qa)
    
    sampled_questions = []
    for cat, questions in categories_in_data.items():
        sample = random.sample(questions, min(questions_per_category, len(questions)))
        sampled_questions.extend(sample)
        print(f"   📌 {cat}: sampled {len(sample)}/{len(questions)} questions")
    
    total_to_evaluate = len(sampled_questions)
    print(f"\n📊 Total sampled: {total_to_evaluate} questions (from {len(categories_in_data)} categories)")
    
    # Run evaluation
    print(f"\n{'=' * 80}")
    print(f"Running Evaluation on {total_to_evaluate} Sampled Questions")
    print(f"{'=' * 80}\n")
    
    results = []
    start_time = time.time()
    
    for i, qa in enumerate(sampled_questions, 1):
        q_id = qa['id']
        question = qa['question']
        reference = qa['reference_answer']
        category = qa['category']
        difficulty = qa['difficulty']
        
        print(f"[{i}/{total_to_evaluate}] {q_id} ({category}) - {question[:50]}...")
        
        # Query engine
        query_start = time.time()
        try:
            result = engine.query(question, top_k=10, use_llm=True)
            query_time = time.time() - query_start
            
            answer = result['answer']
            context = result['context']
            num_facts = len(result['retrieved_facts'])
            
            print(f"  ✓ Query completed in {query_time:.2f}s ({num_facts} facts retrieved)")
            
            # Evaluate with LLM-as-Judge
            print(f"  📊 Evaluating...")
            
            faithfulness = evaluator.evaluate_faithfulness(question, answer, context)
            correctness = evaluator.evaluate_correctness(question, answer, reference)
            completeness = evaluator.evaluate_completeness(question, answer, reference)
            
            # Store result
            results.append({
                'id': q_id,
                'question': question,
                'category': category,
                'difficulty': difficulty,
                'reference_answer': reference,
                'generated_answer': answer,
                'retrieved_context': context,
                'num_facts_retrieved': num_facts,
                'query_time_seconds': query_time,
                'evaluation': {
                    'faithfulness': faithfulness,
                    'correctness': correctness,
                    'completeness': completeness
                }
            })
            
            # Print scores
            f_score = faithfulness.get('faithfulness_score', -1)
            c_score = correctness.get('correctness_score', -1)
            comp_score = completeness.get('completeness_score', -1)
            print(f"  📈 Scores: F={f_score}/5, C={c_score}/5, Comp={comp_score}/5\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            results.append({
                'id': q_id,
                'question': question,
                'category': category,
                'difficulty': difficulty,
                'error': str(e)
            })
    
    total_time = time.time() - start_time
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'total_questions': total_questions,
                'total_time_seconds': total_time,
                'engine_type': 'multi_ontology_hypergraph',
                'num_ontologies': len(ontology_dirs)
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"Evaluation Complete!")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time:.1f}s ({total_time/total_questions:.2f}s per question)")
    print(f"Results saved to: {results_file}")
    
    # Compute aggregate statistics
    compute_statistics(results, results_dir, timestamp)


def compute_statistics(results: List[Dict], results_dir: Path, timestamp: str):
    """Compute and save aggregate statistics"""
    
    print(f"\n{'=' * 80}")
    print("Aggregate Statistics")
    print(f"{'=' * 80}\n")
    
    # Filter successful results
    successful = [r for r in results if 'evaluation' in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    if failed:
        print(f"Failed: {len(failed)} questions")
    
    if not successful:
        print("No successful evaluations to analyze")
        return
    
    # Extract scores
    faithfulness_scores = [r['evaluation']['faithfulness'].get('faithfulness_score', -1) 
                          for r in successful if r['evaluation']['faithfulness'].get('faithfulness_score', -1) >= 0]
    correctness_scores = [r['evaluation']['correctness'].get('correctness_score', -1) 
                         for r in successful if r['evaluation']['correctness'].get('correctness_score', -1) >= 0]
    completeness_scores = [r['evaluation']['completeness'].get('completeness_score', -1) 
                          for r in successful if r['evaluation']['completeness'].get('completeness_score', -1) >= 0]
    
    # Overall statistics
    print(f"\n📊 Overall Scores (out of 5):")
    print(f"  Faithfulness:  {np.mean(faithfulness_scores):.2f} ± {np.std(faithfulness_scores):.2f}")
    print(f"  Correctness:   {np.mean(correctness_scores):.2f} ± {np.std(correctness_scores):.2f}")
    print(f"  Completeness:  {np.mean(completeness_scores):.2f} ± {np.std(completeness_scores):.2f}")
    
    # By category
    categories = {}
    for r in successful:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'faithfulness': [], 'correctness': [], 'completeness': []}
        
        f_sc = r['evaluation']['faithfulness'].get('faithfulness_score', -1)
        c_sc = r['evaluation']['correctness'].get('correctness_score', -1)
        comp_sc = r['evaluation']['completeness'].get('completeness_score', -1)
        
        if f_sc >= 0: categories[cat]['faithfulness'].append(f_sc)
        if c_sc >= 0: categories[cat]['correctness'].append(c_sc)
        if comp_sc >= 0: categories[cat]['completeness'].append(comp_sc)
    
    print(f"\n📂 Scores by Category:")
    for cat, scores in categories.items():
        if scores['correctness']:
            avg_correct = np.mean(scores['correctness'])
            print(f"  {cat}: {avg_correct:.2f}/5")
    
    # By difficulty
    difficulties = {}
    for r in successful:
        diff = r['difficulty']
        if diff not in difficulties:
            difficulties[diff] = {'faithfulness': [], 'correctness': [], 'completeness': []}
        
        f_sc = r['evaluation']['faithfulness'].get('faithfulness_score', -1)
        c_sc = r['evaluation']['correctness'].get('correctness_score', -1)
        comp_sc = r['evaluation']['completeness'].get('completeness_score', -1)
        
        if f_sc >= 0: difficulties[diff]['faithfulness'].append(f_sc)
        if c_sc >= 0: difficulties[diff]['correctness'].append(c_sc)
        if comp_sc >= 0: difficulties[diff]['completeness'].append(comp_sc)
    
    print(f"\n🎯 Scores by Difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in difficulties and difficulties[diff]['correctness']:
            avg_correct = np.mean(difficulties[diff]['correctness'])
            print(f"  {diff}: {avg_correct:.2f}/5")
    
    # Save statistics
    stats = {
        'overall': {
            'faithfulness': {'mean': float(np.mean(faithfulness_scores)), 'std': float(np.std(faithfulness_scores))},
            'correctness': {'mean': float(np.mean(correctness_scores)), 'std': float(np.std(correctness_scores))},
            'completeness': {'mean': float(np.mean(completeness_scores)), 'std': float(np.std(completeness_scores))}
        },
        'by_category': {cat: {
            'correctness_mean': float(np.mean(scores['correctness'])) if scores['correctness'] else 0
        } for cat, scores in categories.items()},
        'by_difficulty': {diff: {
            'correctness_mean': float(np.mean(scores['correctness'])) if scores['correctness'] else 0
        } for diff, scores in difficulties.items()}
    }
    
    stats_file = results_dir / f"evaluation_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistics saved to: {stats_file}")


if __name__ == "__main__":
    run_evaluation()
