#!/usr/bin/env python3
"""
HyperGraph Query Engine - Core OG-RAG Algorithm

Based on: OntoHyperGraphQueryEngine from ograg2
Paper: https://arxiv.org/html/2412.15235v1

Key improvements over GenericQueryEngine:
1. HyperNode key-value pair retrieval (not just simple vector similarity)
2. HyperEdge aggregation (multiple nodes form facts)
3. Dual ranking (key + value separately)
4. Hierarchical expansion (Smart Context)
"""

import json
import pickle
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class HyperGraphQueryEngine:
    """
    HyperGraph Query Engine - OG-RAG Core Algorithm
    
    Key difference from Generic: Uses dual ranking (key + value separately)
    and greedy aggregation to select optimal facts.
    """
    
    def __init__(self, ontology_dir: str,
                 use_ollama: bool = True,
                 ollama_model: str = "llama3.3:70b",
                 ollama_base_url: str = "http://localhost:11434/v1",
                 api_key: str = None,
                 api_model: str = "gpt-4",
                 api_base_url: str = None):
        self.ontology_dir = Path(ontology_dir)
        self.use_ollama = use_ollama
        
        # Load metadata
        with open(self.ontology_dir / 'ontology_metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.ontology_name = self.metadata['ontology_name']
        
        # Load data - SAME as Generic engine
        print(f"Loading {self.ontology_name} hypergraph...")
        with open(self.ontology_dir / 'hypergraph_facts.pkl', 'rb') as f:
            self.facts = pickle.load(f)  # List of fact dicts, indexed by fact_idx
        with open(self.ontology_dir / 'hypergraph_nodes.pkl', 'rb') as f:
            self.hypernodes = pickle.load(f)  # List of {fact_idx, key, value}
        
        self.key_embeddings = np.load(self.ontology_dir / 'hypernode_key_embeddings.npy')
        self.value_embeddings = np.load(self.ontology_dir / 'hypernode_value_embeddings.npy')
        
        # Normalize embeddings for cosine similarity
        self.key_embeddings = self.key_embeddings / (np.linalg.norm(self.key_embeddings, axis=1, keepdims=True) + 1e-8)
        self.value_embeddings = self.value_embeddings / (np.linalg.norm(self.value_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Setup embeddings model
        self.embedding_model_name = self.metadata['hypergraph']['model']
        self.ollama_base_url_raw = ollama_base_url.replace('/v1', '')
        
        # Setup LLM
        if use_ollama:
            print(f"🔐 Using Ollama: {ollama_model} (100% local)")
            self.model_name = ollama_model
            self.llm_client = OpenAI(api_key="ollama", base_url=ollama_base_url)
        else:
            if not api_key:
                raise ValueError("api_key required when use_ollama=False")
            self.model_name = api_model
            kwargs = {"api_key": api_key}
            if api_base_url:
                kwargs["base_url"] = api_base_url
            self.llm_client = OpenAI(**kwargs)
            print(f"🔐 Using API: model={api_model}, base_url={api_base_url}")
        
        print(f"✓ HyperGraph ready! ({len(self.facts)} facts, {len(self.hypernodes)} nodes)\n")

    def _get_embedding(self, text: str) -> np.ndarray:
        try:
            response = requests.post(
                f"{self.ollama_base_url_raw}/api/embeddings",
                json={"model": self.embedding_model_name, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            emb = np.array(response.json()['embedding'])
            return emb / (np.linalg.norm(emb) + 1e-8)  # Normalize
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros(768)

    def _generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical answer to improve retrieval.
        The hypothetical answer provides better semantic overlap with ontology.
        """
        try:
            # Match format from ontology_generator._call_llm() which works
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful knowledge base assistant."},
                    {"role": "user", "content": f"Briefly answer this question in 2-3 sentences: {query}"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            content = response.choices[0].message.content
            if not content or len(content.strip()) < 10:
                print(f"   ⚠️ HyDE got empty/short response, using keyword expansion")
                return f"{query} license copyright legal terms conditions"
            
            hypothetical = content.strip()
            print(f"   🔮 HyDE: {hypothetical[:80]}...")
            return hypothetical
            
        except Exception as e:
            print(f"   ⚠️ HyDE failed: {e}, using original query")
            return query

    def retrieve(self, query: str, top_k: int = 10, expand_hierarchy: bool = True, use_hyde: bool = False) -> List[Dict]:
        """
        Retrieve relevant facts using DUAL RANKING (key + value separately)
        
        This is the core OG-RAG algorithm difference:
        - Generic: max(key_score, value_score) for each node
        - HyperGraph: Top-K by key + Top-K by value → union → aggregate by fact
        
        With HyDE: First generate hypothetical answer, then search with that
        """
        # HyDE: Generate hypothetical answer for better retrieval
        search_text = query
        if use_hyde:
            search_text = self._generate_hypothetical_answer(query)
        
        query_embedding = self._get_embedding(search_text)
        
        # Compute scores
        key_scores = np.dot(self.key_embeddings, query_embedding)
        value_scores = np.dot(self.value_embeddings, query_embedding)
        
        # DUAL RANKING: Get top nodes by KEY and by VALUE separately
        top_by_key = np.argsort(key_scores)[-top_k*2:][::-1]
        top_by_value = np.argsort(value_scores)[-top_k*2:][::-1]
        
        # Union of top nodes
        top_node_indices = set(top_by_key.tolist()) | set(top_by_value.tolist())
        
        # Aggregate scores by fact_idx (greedy: take max score per fact)
        fact_scores = {}
        for node_idx in top_node_indices:
            hypernode = self.hypernodes[node_idx]
            fact_idx = hypernode['fact_idx']
            # Combined score for this node
            node_score = key_scores[node_idx] + value_scores[node_idx]
            if fact_idx not in fact_scores or node_score > fact_scores[fact_idx]:
                fact_scores[fact_idx] = node_score
        
        # Sort facts by score
        sorted_facts = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Group by term_id (merge multiple chunks of same term)
        term_groups = {}
        for fact_idx, score in sorted_facts[:top_k * 3]:
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
        
        # Get top-k terms
        sorted_terms = sorted(term_groups.values(), key=lambda x: x['max_score'], reverse=True)
        top_terms = sorted_terms[:top_k]
        
        # Format results
        results = []
        for term_data in top_terms:
            merged_fact = {
                '_raw_term': term_data['raw_term'],
                '_term_id': term_data['term_id'],
                'id': term_data['term_id'],
                'label': term_data['raw_term'].get('label', ''),
                '_chunks': term_data['chunks'],
                '_merged': True
            }
            results.append({
                'fact': merged_fact,
                'score': float(term_data['max_score']),
                'term_id': term_data['term_id']
            })
        
        # Smart Context: Expand with related terms
        if expand_hierarchy and results:
            related_results = []
            seen_term_ids = {r['term_id'] for r in results}
            
            for result in results:
                raw_term = result['fact']['_raw_term']
                rels = raw_term.get('relationships', {})
                
                for rel_type, related_ids in rels.items():
                    if not isinstance(related_ids, list):
                        related_ids = [related_ids]
                    
                    type_added = 0
                    for related_id in related_ids:
                        if type_added >= 2: break
                        if related_id in seen_term_ids: continue
                        
                        for fact in self.facts:
                            fact_term_id = fact.get('_term_id', '')
                            if fact_term_id == related_id and fact.get('_chunk_type') == 'core':
                                related_results.append({
                                    'fact': fact,
                                    'score': 0.5,
                                    'term_id': fact_term_id,
                                    '_is_related': True,
                                    '_related_via': rel_type,
                                    '_related_to': result['term_id']
                                })
                                seen_term_ids.add(fact_term_id)
                                type_added += 1
                                break
            
            results.extend(related_results)
        
        # Re-sort by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def query(self, query: str, top_k: int = 10, use_llm: bool = True, **kwargs) -> Dict[str, Any]:
        """Full query pipeline: retrieve + generate"""
        retrieved = self.retrieve(query, top_k=top_k)
        
        # Format context
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact = result['fact']
            fact_text = self._format_fact(fact)
            context_parts.append(f"[{i}] (Relevance: {result['score']:.3f})\n{fact_text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        if use_llm and self.llm_client:
            answer = self._generate_answer(query, retrieved)
        else:
            if retrieved:
                answer = self._format_fact(retrieved[0]['fact'])
            else:
                answer = f"No relevant information found in {self.ontology_name} ontology."
        
        return {
            'answer': answer,
            'retrieved_facts': retrieved,
            'context': context,
            'full_context': context,
            'ontology': self.ontology_name
        }
    
    def _format_fact(self, fact: Dict) -> str:
        """Format fact as human-readable text"""
        raw_term = fact.get('_raw_term', {})
        if not raw_term:
            return str(fact)
        
        lines = []
        
        # ID and Label
        term_id = raw_term.get('id', '')
        label = raw_term.get('label', '')
        if term_id and label:
            lines.append(f"**{label}** ({term_id})")
        elif label:
            lines.append(f"**{label}**")
        
        # Definition
        if 'definition' in raw_term:
            lines.append(f"Definition: {raw_term['definition']}")
        
        # Synonyms
        if 'synonyms' in raw_term and raw_term['synonyms']:
            lines.append(f"Synonyms: {', '.join(raw_term['synonyms'])}")
        
        # Namespace
        if 'namespace' in raw_term:
            lines.append(f"Namespace: {raw_term['namespace']}")
        
        # Relationships
        if 'relationships' in raw_term:
            lines.append("\nRelationships:")
            for rel_type, targets in raw_term['relationships'].items():
                if isinstance(targets, list):
                    for target in targets:
                        lines.append(f"  • {rel_type}: {target}")
                else:
                    lines.append(f"  • {rel_type}: {targets}")
        
        # Comments
        if 'comments' in raw_term and raw_term['comments']:
            lines.append(f"\nComments:")
            for comment in raw_term['comments'][:2]:
                lines.append(f"  • {comment[:200]}...")
        
        # Properties
        if 'properties' in raw_term and raw_term['properties']:
            lines.append("\nProperties:")
            for prop_name, values in raw_term['properties'].items():
                if not isinstance(values, list):
                    values = [values]
                for val in values:
                    val_str = str(val)[:300]
                    lines.append(f"  • {prop_name}: {val_str}")
        
        return "\n".join(lines)
    
    def _generate_answer(self, query: str, retrieved: List[Dict]) -> str:
        """Generate answer using LLM"""
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact_text = self._format_fact(result['fact'])
            context_parts.append(f"[{i}] {fact_text}")
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a precise expert in {self.ontology_name} ontology. Answer ONLY using the provided terms below.

RETRIEVED ONTOLOGY TERMS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer STRICTLY based on the terms above
2. Cite term IDs when mentioning concepts
3. If information is not in the provided terms, say "The provided terms don't contain information about..."
4. Do NOT hallucinate or add external knowledge
5. Keep answer concise and factual

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a precise {self.ontology_name} ontology expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            return context
