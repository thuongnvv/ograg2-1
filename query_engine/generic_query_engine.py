#!/usr/bin/env python3
"""
Generic Query Engine - Works with ANY ontology hypergraph

Provides:
- Dual ranking retrieval (key + value similarity)
- Hierarchical expansion
- LLM-based answer generation
- 100% local with Ollama (no data sent to internet)
"""

import os
import json
import pickle
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI


class GenericQueryEngine:
    """Generic query engine for any ontology - Supports Ollama (local) or API (cloud)"""
    
    def __init__(self, ontology_dir: str, 
                 use_ollama: bool = True, 
                 ollama_model: str = "llama3.3:70b",
                 ollama_base_url: str = "http://localhost:11434/v1",
                 api_key: str = None,
                 api_model: str = "gpt-4",
                 api_base_url: str = None):
        """
        Initialize query engine
        
        Args:
            ontology_dir: Directory containing parsed ontology and hypergraph
            use_ollama: Whether to use Ollama for local LLM (default: True)
            ollama_model: Ollama model name (default: "llama3.3:70b")
            ollama_base_url: Ollama API base URL (default: "http://localhost:11434/v1")
            api_key: OpenAI/MegaLLM API key (required if use_ollama=False)
            api_model: API model name (default: "gpt-4")
            api_base_url: Custom API base URL (for MegaLLM, etc.)
        """
        self.ontology_dir = Path(ontology_dir)
        self.use_ollama = use_ollama
        
        # Load metadata
        metadata_file = self.ontology_dir / 'ontology_metadata.json'
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.ontology_name = self.metadata['ontology_name']
        self.key_prefix = self.metadata['id_prefix']
        
        # Load hypergraph
        print(f"Loading {self.ontology_name} hypergraph...")
        
        with open(self.ontology_dir / 'hypergraph_facts.pkl', 'rb') as f:
            self.facts = pickle.load(f)
        
        with open(self.ontology_dir / 'hypergraph_nodes.pkl', 'rb') as f:
            self.hypernodes = pickle.load(f)
        
        self.key_embeddings = np.load(self.ontology_dir / 'hypernode_key_embeddings.npy')
        self.value_embeddings = np.load(self.ontology_dir / 'hypernode_value_embeddings.npy')
        
        # Load embedding configuration
        model_name = self.metadata['hypergraph']['model']
        use_ollama_embeddings = self.metadata['hypergraph'].get('use_ollama', False)
        
        self.use_ollama_embeddings = use_ollama_embeddings
        self.embedding_model_name = model_name
        self.ollama_base_url_raw = ollama_base_url.replace('/v1', '')  # Remove /v1 for embeddings API
        
        if use_ollama_embeddings:
            print(f"🔐 Using Ollama embeddings: {model_name} (100% local)")
            self.model = None
        else:
            raise ValueError(
                "Non-Ollama embeddings are no longer supported for security.\n"
                "Please rebuild your hypergraph with use_ollama=True"
            )
        
        # Setup LLM client - Support BOTH Ollama and API
        self.llm_client = None
        
        if use_ollama:
            # Local Ollama
            print(f"🔐 Using Ollama LLM: {ollama_model} (100% local)")
            self.model_name = ollama_model
            self.llm_client = OpenAI(
                api_key="ollama",  # Ollama doesn't need real API key
                base_url=ollama_base_url
            )
        else:
            # Cloud API (OpenAI, MegaLLM, etc.)
            if not api_key:
                raise ValueError(
                    "api_key required when use_ollama=False\n"
                    "Provide your OpenAI/MegaLLM API key"
                )
            
            self.model_name = api_model
            kwargs = {
                "api_key": api_key,
                "timeout": 60.0,
                "max_retries": 3
            }
            if api_base_url:
                kwargs["base_url"] = api_base_url
                print(f"☁️  Using API LLM: {api_model} (via {api_base_url})")
            else:
                print(f"☁️  Using OpenAI LLM: {api_model}")
            
            self.llm_client = OpenAI(**kwargs)
        
        print(f"✓ Ready! ({len(self.facts):,} facts, {len(self.hypernodes):,} nodes)\n")

    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama (100% local)"""
        # Use Ollama embeddings API
        try:
            response = requests.post(
                f"{self.ollama_base_url_raw}/api/embeddings",
                json={"model": self.embedding_model_name, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()['embedding']
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting Ollama embedding: {e}")
            # Fallback to zero vector
            return np.zeros(768)
    
    def retrieve(self, query: str, top_k: int = 5, expand_hierarchy: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant facts with Smart Context
        
        Strategy:
        1. Find top-k most relevant chunks
        2. Group by term_id (merge chunks of same term)
        3. Optionally expand with parent/related terms (hierarchical context)
        
        Args:
            query: Search query
            top_k: Number of TERMS to retrieve (not chunks)
            expand_hierarchy: Whether to add parent terms for context
        
        Returns:
            List of retrieved facts (merged by term) with scores
        """
        # Encode query using Ollama (100% local)
        query_embedding = self._get_embedding(query)
        
        # Normalize query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        key_scores = np.dot(self.key_embeddings, query_embedding)
        value_scores = np.dot(self.value_embeddings, query_embedding)
        
        # Combined score (max of key and value)
        combined_scores = np.maximum(key_scores, value_scores)
        
        # Group by fact (chunk)
        fact_scores = {}
        for idx, hypernode in enumerate(self.hypernodes):
            fact_idx = hypernode['fact_idx']
            score = combined_scores[idx]
            
            if fact_idx not in fact_scores or score > fact_scores[fact_idx]:
                fact_scores[fact_idx] = score
        
        # Get top-k*3 chunks first (may have multiple chunks per term)
        sorted_facts = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
        top_chunks = sorted_facts[:top_k * 3]
        
        # Group by term_id and merge chunks
        term_groups = {}
        for fact_idx, score in top_chunks:
            fact = self.facts[fact_idx]
            term_id = fact.get('_term_id', fact.get('id', ''))
            
            if term_id not in term_groups:
                term_groups[term_id] = {
                    'term_id': term_id,
                    'max_score': score,
                    'chunks': [],
                    'raw_term': fact.get('_raw_term', {})
                }
            
            term_groups[term_id]['chunks'].append({
                'fact': fact,
                'score': score,
                'chunk_type': fact.get('_chunk_type', 'unknown')
            })
            term_groups[term_id]['max_score'] = max(term_groups[term_id]['max_score'], score)
        
        # Sort terms by max score and get top-k TERMS
        sorted_terms = sorted(term_groups.values(), key=lambda x: x['max_score'], reverse=True)
        top_terms = sorted_terms[:top_k]
        
        # Merge chunks into single fact per term
        results = []
        for term_data in top_terms:
            # Combine all chunk texts
            merged_fact = {
                '_raw_term': term_data['raw_term'],
                '_term_id': term_data['term_id'],
                'id': term_data['term_id'],
                'label': term_data['raw_term'].get('label', ''),
                '_chunks': term_data['chunks'],  # Keep chunk info
                '_merged': True
            }
            
            results.append({
                'fact': merged_fact,
                'score': float(term_data['max_score']),
                'term_id': term_data['term_id']
            })
        
        # Smart Context: Add parent terms if requested (up to 2 parents per main term)
        if expand_hierarchy and results:
            parent_results = []
            seen_term_ids = {r['term_id'] for r in results}  # Track existing terms
            
            for result in results:
                raw_term = result['fact']['_raw_term']
                rels = raw_term.get('relationships', {})
                
                if 'is_a' in rels:
                    parents = rels['is_a'] if isinstance(rels['is_a'], list) else [rels['is_a']]
                    
                    # Add up to 2 parents for this specific term
                    term_parents_added = 0
                    for parent_id in parents:
                        if term_parents_added >= 2:
                            break
                        
                        # Skip if already retrieved or added
                        if parent_id in seen_term_ids:
                            continue
                            
                        # Find parent chunks
                        for fact in self.facts:
                            fact_term_id = fact.get('_term_id', '')
                            if fact_term_id == parent_id and fact.get('_chunk_type') == 'core':
                                parent_results.append({
                                    'fact': fact,
                                    'score': 0.5,  # Lower score for context
                                    'term_id': fact_term_id,
                                    '_is_parent': True,
                                    '_parent_of': result['term_id']
                                })
                                seen_term_ids.add(fact_term_id)
                                term_parents_added += 1
                                break
            
            # Add unique parent terms
            results.extend(parent_results)
        
        # IMPORTANT: Re-sort by relevance score after adding parent terms
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return results
    
    def query(self, query: str, top_k: int = 5, use_llm: bool = True) -> Dict[str, Any]:
        """
        Full query pipeline: retrieve + generate answer
        
        Args:
            query: User question
            top_k: Number of facts to retrieve
            use_llm: Whether to use LLM for answer generation
        
        Returns:
            Dict with 'answer', 'retrieved_facts', 'context'
        """
        # Retrieve relevant facts
        retrieved = self.retrieve(query, top_k=top_k)
        
        # Format context
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact = result['fact']
            score = result['score']
            
            # Format fact as readable text
            fact_text = self._format_fact(fact)
            context_parts.append(f"{i}. {fact_text} (relevance: {score:.3f})")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        if use_llm and self.llm_client:
            answer = self._generate_llm_answer(query, retrieved)
        else:
            # Fallback: return top result
            if retrieved:
                answer = self._format_fact(retrieved[0]['fact'])
            else:
                answer = f"No relevant information found in {self.ontology_name} ontology."
        
        # Format full context for verification
        full_context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact = result['fact']
            score = result['score']
            fact_text = self._format_fact(fact)
            full_context_parts.append(f"[{i}] (Relevance: {score:.3f})\n{fact_text}")
        
        full_context = "\n\n".join(full_context_parts)
        
        return {
            'answer': answer,
            'retrieved_facts': retrieved,
            'context': context,
            'full_context': full_context,  # Context sent to LLM for verification
            'ontology': self.ontology_name
        }
    
    def _format_fact(self, fact: Dict) -> str:
        """
        Format fact as human-readable text - GENERIC, no hardcoding
        Extracts from raw term data dynamically
        """
        # Get raw term data (complete JSON)
        raw_term = fact.get('_raw_term', {})
        
        if not raw_term:
            # Fallback for old format
            return str(fact)
        
        lines = []
        
        # ID and Label
        term_id = raw_term.get('id', '')
        label = raw_term.get('label', '')
        if term_id and label:
            lines.append(f"**{label}** ({term_id})")
        elif term_id:
            lines.append(f"**Term: {term_id}**")
        elif label:
            lines.append(f"**{label}**")
        
        # Definition (most important)
        if 'definition' in raw_term:
            lines.append(f"Definition: {raw_term['definition']}")
        
        # Synonyms
        if 'synonyms' in raw_term and raw_term['synonyms']:
            lines.append(f"Synonyms: {', '.join(raw_term['synonyms'])}")
        
        # Namespace
        if 'namespace' in raw_term:
            lines.append(f"Namespace: {raw_term['namespace']}")
        
        # Relationships - dynamic, all types
        if 'relationships' in raw_term:
            lines.append("\nRelationships:")
            for rel_type, targets in raw_term['relationships'].items():
                if isinstance(targets, list):
                    for target in targets:
                        lines.append(f"  • {rel_type}: {target}")
                else:
                    lines.append(f"  • {rel_type}: {targets}")
        
        # Comments (if any)
        if 'comments' in raw_term and raw_term['comments']:
            if len(raw_term['comments']) == 1:
                lines.append(f"\nComment: {raw_term['comments'][0]}")
            else:
                lines.append(f"\nComments:")
                for comment in raw_term['comments'][:2]:  # Max 2
                    lines.append(f"  • {comment[:200]}...")
        
        # Examples (if any)
        if 'examples' in raw_term and raw_term['examples']:
            lines.append(f"\nExamples: {len(raw_term['examples'])} available")
            # Show first example (truncated)
            first_example = raw_term['examples'][0]
            if len(first_example) > 200:
                first_example = first_example[:200] + '...'
            lines.append(f"  • {first_example}")
        
        # Cross-references
        if 'cross_references' in raw_term and raw_term['cross_references']:
            xrefs = raw_term['cross_references'][:5]  # Max 5
            lines.append(f"\nCross-references: {', '.join(xrefs)}")
            
        # Custom Properties (NEW)
        if 'properties' in raw_term and raw_term['properties']:
            lines.append("\nProperties:")
            for prop_name, values in raw_term['properties'].items():
                # Ensure values is list
                if not isinstance(values, list):
                    values = [values]
                for val in values:
                    # Truncate long values
                    val_str = str(val)
                    if len(val_str) > 300:
                        val_str = val_str[:300] + '...'
                    lines.append(f"  • {prop_name}: {val_str}")
        
        return "\n".join(lines)
    
    def _generate_llm_answer(self, query: str, retrieved: List[Dict]) -> str:
        """Generate answer using LLM"""
        # Format context for LLM
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact = result['fact']
            fact_text = self._format_fact(fact)
            context_parts.append(f"[{i}] {fact_text}")
        
        context = "\n\n".join(context_parts)
        
        # Create strict prompt
        prompt = f"""You are a precise expert in {self.ontology_name} ontology. Answer ONLY using the provided terms below. Do NOT add information from outside knowledge.

RETRIEVED ONTOLOGY TERMS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer STRICTLY based on the terms above
2. Cite term IDs when mentioning concepts (e.g., "DNA repair (GO:0006281)")
3. If information is not in the provided terms, say "The provided terms don't contain information about..."
4. Do NOT hallucinate or add external knowledge
5. Keep answer concise and factual

Answer:"""
        
        try:
            # Use configured model (Ollama or API)
            model = self.model_name  # e.g., "llama3.3:70b", "gpt-4", "llama3.3-70b-instruct"
            
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"You are a precise {self.ontology_name} ontology expert. Only use provided information. Never hallucinate."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"LLM generation error: {e}")
            # Fallback to formatted context
            return context


def main():
    """Test query engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generic Query Engine")
    parser.add_argument(
        "--ontology-dir",
        required=True,
        help="Directory containing parsed ontology and hypergraph"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query to search"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM generation"
    )
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv('MEGALLM_API_KEY')
    
    # Initialize engine
    engine = GenericQueryEngine(args.ontology_dir, megallm_api_key=api_key)
    
    # Query
    result = engine.query(args.query, top_k=args.top_k, use_llm=not args.no_llm)
    
    print("=" * 80)
    print(f"Query: {args.query}")
    print("=" * 80)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\n\nRetrieved {len(result['retrieved_facts'])} facts from {result['ontology']}")


if __name__ == "__main__":
    main()
