#!/usr/bin/env python3
"""
Multi-Ontology Query Engine - Query across multiple ontologies and merge results

For handling chunked ontologies from large pages.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from query_engine.generic_query_engine import GenericQueryEngine
from query_engine.hypergraph_query_engine import HyperGraphQueryEngine


class MultiOntologyQueryEngine:
    """
    Query engine that searches across multiple ontologies and merges results.
    
    Used for:
    - Chunked ontologies from large pages
    - Multiple related ontologies
    - Cross-domain queries
    """
    
    def __init__(self, ontology_dirs: List[str],
                 engine_type: str = "hypergraph",
                 use_ollama: bool = True,
                 ollama_model: str = "llama3.3:70b",
                 ollama_base_url: str = "http://localhost:11434/v1",
                 api_key: str = None,
                 api_model: str = "gpt-4",
                 api_base_url: str = None):
        """
        Initialize multi-ontology query engine.
        
        Args:
            ontology_dirs: List of parsed ontology directories
            engine_type: "generic" or "hypergraph"
            ...other args same as single-ontology engines
        """
        self.ontology_dirs = [Path(d) for d in ontology_dirs]
        self.engine_type = engine_type
        self.engines = []
        
        # Common kwargs for engines
        engine_kwargs = {
            "use_ollama": use_ollama,
            "ollama_model": ollama_model,
            "ollama_base_url": ollama_base_url,
            "api_key": api_key,
            "api_model": api_model,
            "api_base_url": api_base_url
        }
        
        # Initialize an engine for each ontology
        print(f"🔄 Loading {len(ontology_dirs)} ontologies...")
        
        for i, onto_dir in enumerate(self.ontology_dirs):
            try:
                if engine_type == "hypergraph":
                    engine = HyperGraphQueryEngine(str(onto_dir), **engine_kwargs)
                else:
                    engine = GenericQueryEngine(str(onto_dir), **engine_kwargs)
                
                self.engines.append({
                    'engine': engine,
                    'dir': onto_dir,
                    'name': engine.ontology_name,
                    'index': i
                })
            except Exception as e:
                print(f"   ⚠️ Failed to load {onto_dir}: {e}")
        
        print(f"✓ Loaded {len(self.engines)} ontologies for multi-query\n")
    
    def retrieve(self, query: str, top_k_per_ontology: int = 20, 
                final_top_k: int = 20) -> List[Dict]:
        """
        Retrieve from all ontologies and return top-k by score.
        
        Strategy:
        1. Get top_k_per_ontology from EACH ontology (e.g., 20 x 3 = 60)
        2. Merge all results and sort by score
        3. Return top final_top_k (e.g., 20)
        
        This allows the most relevant ontology to dominate the results.
        """
        all_results = []
        
        print(f"\n🔍 Multi-Ontology Query: '{query[:50]}...'")
        print(f"   Querying {len(self.engines)} ontologies, top_k={top_k_per_ontology} each\n")
        
        # HyDE: Generate hypothetical answer ONCE for all ontologies
        search_query = query.strip()
        if len(search_query.split()) < 50 and self.engines:
            try:
                first_engine = self.engines[0]['engine']
                if hasattr(first_engine, '_generate_hypothetical_answer'):
                    search_query = first_engine._generate_hypothetical_answer(query)
            except Exception as e:
                print(f"   ⚠️ HyDE failed: {e}, using original query")
        
        # Query each ontology
        for engine_info in self.engines:
            engine = engine_info['engine']
            
            try:
                results = engine.retrieve(search_query, top_k=top_k_per_ontology, use_hyde=False)
                
                print(f"   📊 Ontology {engine_info['index']+1} ({engine_info['name'][:30]}):") 
                print(f"      Retrieved {len(results)} facts")
                
                # Tag results with source
                for rank, result in enumerate(results):
                    result['_source_ontology'] = engine_info['name']
                    result['_source_index'] = engine_info['index']
                    result['_local_rank'] = rank + 1
                
                all_results.extend(results)
                
            except Exception as e:
                print(f"   ⚠️ Error querying {engine_info['name']}: {e}")
        
        # Sort all results by score (highest first)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Remove duplicates (keep highest score)
        seen_term_ids = set()
        final_results = []
        
        for result in all_results:
            term_id = result.get('term_id', '')
            if term_id and term_id in seen_term_ids:
                continue
            seen_term_ids.add(term_id)
            final_results.append(result)
            
            if len(final_results) >= final_top_k:
                break
        
        print(f"\n   Total collected: {len(all_results)} facts")
        print(f"   After dedup & top-k: {len(final_results)} facts\n")
        
        return final_results
    
    def query(self, query: str, top_k: int = 20, use_llm: bool = True, 
             **kwargs) -> Dict[str, Any]:
        """
        Full query pipeline: retrieve from all ontologies + generate answer.
        """
        # Retrieve merged results - use higher top_k for multi-ontology
        retrieved = self.retrieve(query, top_k_per_ontology=top_k, 
                                  final_top_k=top_k * 2)
        
        # Format context
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact = result.get('fact', {})
            source = result.get('_source_ontology', 'unknown')
            score = result.get('score', 0)
            
            fact_text = self._format_fact(fact)
            context_parts.append(f"[{i}] (Source: {source}, Relevance: {score:.3f})\n{fact_text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using first engine's LLM
        if use_llm and self.engines:
            answer = self._generate_answer(query, retrieved)
        else:
            answer = context if context else "No relevant information found."
        
        return {
            'answer': answer,
            'retrieved_facts': retrieved,
            'context': context,
            'full_context': context,
            'ontology': f"Multi ({len(self.engines)} ontologies)",
            'sources': [e['name'] for e in self.engines]
        }
    
    def _format_fact(self, fact: Dict) -> str:
        """Format fact for display - show full content"""
        raw_term = fact.get('_raw_term', fact)
        lines = []
        
        label = raw_term.get('label', '')
        term_id = raw_term.get('id', '')
        
        # Title line
        if label and term_id:
            lines.append(f"**{label}** ({term_id})")
        elif label:
            lines.append(f"**{label}**")
        elif term_id:
            lines.append(f"**{term_id}**")
        
        # Definition (check both 'definition' and 'comments' - RDF uses comments)
        if 'definition' in raw_term and raw_term['definition']:
            lines.append(raw_term['definition'])
        elif 'comments' in raw_term and raw_term['comments']:
            # Comments is array of rdfs:comment strings - this IS the main content
            for comment in raw_term['comments']:
                if comment and len(comment) > 5:
                    lines.append(comment)
        
        # Searchable text as fallback
        if not any(len(l) > 30 for l in lines) and '_searchable_text' in fact:
            text = fact['_searchable_text'][:300]
            lines.append(text)
        
        # Relationships
        if 'relationships' in raw_term:
            for rel_type, targets in raw_term['relationships'].items():
                if isinstance(targets, list):
                    for t in targets[:2]:
                        lines.append(f"  • {rel_type}: {t}")
                else:
                    lines.append(f"  • {rel_type}: {targets}")
        
        return "\n".join(lines) if lines else f"[{term_id or 'Unknown term'}]"
    
    def _generate_answer(self, query: str, retrieved: List[Dict]) -> str:
        """Generate answer using LLM"""
        # Use first engine's LLM
        if not self.engines:
            return "No engines available"
        
        engine = self.engines[0]['engine']
        
        # Format context
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            fact = result.get('fact', {})
            fact_text = self._format_fact(fact)
            source = result.get('_source_ontology', '')
            context_parts.append(f"[{i}] ({source})\n{fact_text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""### ROLE
You are a Precise Ontology Analyst. Your goal is to synthesize structured knowledge from the provided context with high fidelity and logical consistency.

### CONTEXT: RETRIEVED FACTS
{context}

### USER QUESTION
{query}

### STRATEGIC INSTRUCTIONS
1. **Source Fidelity:** Answer EXCLUSIVELY using the provided facts. If the context does not contain enough information to answer the question fully, state clearly what is missing rather than speculating.
2. **Citations:** Every claim must be followed by its source ID in brackets, e.g., "The concept of X is a subclass of Y [Fact 1]." 
3. **Fairness & Conflict Handling:** If different facts provide conflicting information about a concept, present all perspectives neutrally. Do not choose one over the other unless the ontology hierarchy explicitly dictates a priority.
4. **Logical Synthesis:** Do not just list facts. Connect related entities and relationships to form a coherent, professional explanation. Use bullet points for complex hierarchies.
5. **No External Bias:** Ignore any prior knowledge you have about this topic outside of the provided facts.

### RESPONSE STRUCTURE
- **Direct Answer:** (A brief summary)
- **Detailed Analysis:** (The meat of the answer with citations)
- **Data Limitations:** (Only if information is missing or ambiguous)

Answer:"""
        
        try:
            response = engine.llm_client.chat.completions.create(
                model=engine.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise ontology expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            return context
