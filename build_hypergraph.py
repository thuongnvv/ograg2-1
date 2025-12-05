#!/usr/bin/env python3
"""
Generic HyperGraph Builder - Works with ANY parsed ontology!

Auto-detects ontology structure and builds hypergraph accordingly.
No configuration needed - just point to parsed ontology directory.

Uses Ollama for 100% local embeddings - no data sent to internet!
"""

import os
import sys
import json
import pickle
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any


def load_ontology_metadata(ontology_dir: str) -> Dict:
    """Load ontology metadata from parsed directory"""
    metadata_file = Path(ontology_dir) / 'ontology_metadata.json'
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            "Please run parse_owl.py first to parse the ontology."
        )
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_term_to_facts(term_data: Dict[str, Any], key_prefix: str) -> list:
    """
    Convert ontology term to MULTIPLE chunks with Smart Context
    
    Smart Context Strategy:
    1. Core chunk: ID, label, definition (most important)
    2. Synonyms chunk: Alternative names
    3. Relationships chunk: Parent/child/related terms
    4. Details chunk: Examples, comments, metadata
    
    Args:
        term_data: Complete parsed term data
        key_prefix: Ontology prefix
    
    Returns:
        List of facts (chunks) for this term
    """
    term_id = term_data.get('id', '')
    facts = []
    
    # CHUNK 1: CORE (always present) - Highest priority
    core_parts = []
    if 'id' in term_data:
        core_parts.append(f"ID: {term_data['id']}")
    if 'label' in term_data:
        core_parts.append(f"Label: {term_data['label']}")
    
    # Flexible content field - works for both formal ontologies AND FAQ
    if 'definition' in term_data:
        core_parts.append(f"Definition: {term_data['definition']}")
    elif 'comments' in term_data and term_data['comments']:
        # Fallback for FAQ-style: use comment as main content
        core_parts.append(f"Answer: {term_data['comments'][0]}")
    
    if 'namespace' in term_data:
        core_parts.append(f"Namespace: {term_data['namespace']}")
    
    if core_parts:
        facts.append({
            '_raw_term': term_data,
            '_ontology_prefix': key_prefix,
            '_chunk_type': 'core',
            '_term_id': term_id,
            'id': term_id,
            'label': term_data.get('label', ''),
            '_searchable_text': '\n'.join(core_parts)
        })
    
    # CHUNK 2: SYNONYMS (if exists)
    synonyms = term_data.get('synonyms', [])
    if synonyms:
        syn_parts = [
            f"Term: {term_id}",
            f"Label: {term_data.get('label', '')}",
            f"Synonyms: {', '.join(synonyms)}"
        ]
        facts.append({
            '_raw_term': term_data,
            '_ontology_prefix': key_prefix,
            '_chunk_type': 'synonyms',
            '_term_id': term_id,
            'id': term_id,
            'label': term_data.get('label', ''),
            '_searchable_text': '\n'.join(syn_parts)
        })
    
    # CHUNK 3: RELATIONSHIPS (if exists) - For hierarchical context
    relationships = term_data.get('relationships', {})
    if relationships:
        rel_parts = [
            f"Term: {term_id}",
            f"Label: {term_data.get('label', '')}",
            "Relationships:"
        ]
        for rel_type, targets in relationships.items():
            if isinstance(targets, list):
                for target in targets:
                    rel_parts.append(f"  {rel_type}: {target}")
            else:
                rel_parts.append(f"  {rel_type}: {targets}")
        
        facts.append({
            '_raw_term': term_data,
            '_ontology_prefix': key_prefix,
            '_chunk_type': 'relationships',
            '_term_id': term_id,
            'id': term_id,
            'label': term_data.get('label', ''),
            '_searchable_text': '\n'.join(rel_parts),
            '_parents': relationships.get('is_a', [])  # For hierarchical expansion
        })
    
    # CHUNK 4: DETAILS (examples, comments)
    detail_parts = []
    
    if 'comments' in term_data and term_data['comments']:
        detail_parts.append(f"Term: {term_id}")
        detail_parts.append(f"Label: {term_data.get('label', '')}")
        detail_parts.append("Comments:")
        for comment in term_data['comments'][:2]:  # Max 2 comments
            if len(comment) > 300:
                comment = comment[:300] + '...'
            detail_parts.append(f"  • {comment}")
    
    if 'examples' in term_data and term_data['examples']:
        if not detail_parts:
            detail_parts.append(f"Term: {term_id}")
            detail_parts.append(f"Label: {term_data.get('label', '')}")
        detail_parts.append("Examples:")
        for example in term_data['examples'][:2]:  # Max 2 examples
            if len(example) > 300:
                example = example[:300] + '...'
            detail_parts.append(f"  • {example}")
    
    if detail_parts:
        facts.append({
            '_raw_term': term_data,
            '_ontology_prefix': key_prefix,
            '_chunk_type': 'details',
            '_term_id': term_id,
            'id': term_id,
            'label': term_data.get('label', ''),
            '_searchable_text': '\n'.join(detail_parts)
        })
    
    # If no chunks created (shouldn't happen), create minimal one
    if not facts:
        facts.append({
            '_raw_term': term_data,
            '_ontology_prefix': key_prefix,
            '_chunk_type': 'minimal',
            '_term_id': term_id,
            'id': term_id,
            'label': term_data.get('label', ''),
            '_searchable_text': f"Term: {term_id}"
        })
    
    return facts


def get_ollama_embeddings(texts: List[str], model: str = "nomic-embed-text", 
                         base_url: str = "http://localhost:11434") -> np.ndarray:
    """
    Get embeddings from Ollama
    
    Args:
        texts: List of texts to embed
        model: Ollama embedding model name
        base_url: Ollama server URL
        
    Returns:
        Embeddings as numpy array
    """
    embeddings = []
    
    for text in tqdm(texts, desc=f"Getting embeddings from Ollama ({model})"):
        try:
            response = requests.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()['embeddings'][0]
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Fallback to zero vector
            embeddings.append([0.0] * 768)  # Default dimension
    
    return np.array(embeddings)


def build_hypergraph(ontology_dir: str, model_name: str = "nomic-embed-text",
                    use_ollama: bool = True, ollama_model: str = "nomic-embed-text",
                    ollama_base_url: str = "http://localhost:11434"):
    """
    Build hypergraph from parsed ontology
    
    Args:
        ontology_dir: Directory containing parsed term JSON files
        model_name: Model name (ignored, kept for compatibility)
        use_ollama: Whether to use Ollama for embeddings (default: True, always True now)
        ollama_model: Ollama embedding model (default: "nomic-embed-text")
        ollama_base_url: Ollama server URL (default: "http://localhost:11434")
    """
    print(f"\n{'='*80}")
    print(f"Generic HyperGraph Builder - 100% Local with Ollama")
    print(f"{'='*80}\n")
    
    # Force use_ollama to True for security
    if not use_ollama:
        print("⚠️  Warning: use_ollama=False is deprecated for security.")
        print("   Forcing use_ollama=True to ensure no data leaves your machine.\n")
        use_ollama = True
    
    # Load metadata
    print("Loading ontology metadata...")
    metadata = load_ontology_metadata(ontology_dir)
    
    ontology_name = metadata['ontology_name']
    key_prefix = metadata['id_prefix']
    total_terms = metadata['active_terms']
    
    print(f"  ✓ Ontology: {ontology_name}")
    print(f"  ✓ Prefix: {key_prefix}")
    print(f"  ✓ Terms: {total_terms}")
    print(f"  ✓ Relationships: {', '.join(metadata['relationships'])}")
    
    # Use Ollama embeddings (100% local, no internet after model download)
    print(f"\n🔐 Security: Using Ollama embeddings (100% local)")
    print(f"  Model: {ollama_model}")
    print(f"  Server: {ollama_base_url}")
    
    # Test connection
    try:
        response = requests.get(f"{ollama_base_url}/api/version", timeout=5)
        print(f"  ✓ Ollama server connected")
        print(f"  ✓ No data leaves your machine!\n")
    except Exception as e:
        print(f"  ✗ Cannot connect to Ollama server: {e}")
        print(f"  Please run: ollama serve")
        sys.exit(1)
    
    model = None  # Will use Ollama API instead
    embed_dim = 768  # nomic-embed-text dimension
    
    # Load all term files - Auto-detect location
    # Try parsed/ subdirectory first (new structure), then ontology_dir (old structure)
    parsed_dir = Path(ontology_dir) / "parsed"
    if parsed_dir.exists():
        ontology_files = sorted(parsed_dir.glob("term_*.json"))
        print(f"Found {len(ontology_files)} term files in parsed/ subdirectory\n")
    else:
        ontology_files = sorted(Path(ontology_dir).glob("term_*.json"))
        print(f"Found {len(ontology_files)} term files in ontology directory\n")
    
    # Flatten to facts
    print("Flattening terms to facts...")
    all_facts = []
    
    for term_file in tqdm(ontology_files, desc="Processing terms"):
        with open(term_file, 'r', encoding='utf-8') as f:
            term_data = json.load(f)
        
        facts = flatten_term_to_facts(term_data, key_prefix)
        all_facts.extend(facts)
    
    print(f"  ✓ Created {len(all_facts)} HyperEdges (facts)\n")
    
    # Build HyperNodes (key-value pairs)
    print("Building HyperNodes (key-value pairs)...")
    hypernodes = []
    hypernode_keys = []
    hypernode_values = []
    
    for fact_idx, fact in enumerate(tqdm(all_facts, desc="Extracting nodes")):
        for key, value in fact.items():
            if value and str(value).strip():
                hypernodes.append({
                    'fact_idx': fact_idx,
                    'key': key,
                    'value': str(value)
                })
                hypernode_keys.append(key)
                hypernode_values.append(str(value))
    
    print(f"  ✓ Created {len(hypernodes):,} HyperNodes\n")
    
    # Generate embeddings in batches
    print("Generating embeddings (batched)...")
    
    chunk_size = 100000  # Process 100K nodes at a time
    num_chunks = (len(hypernodes) + chunk_size - 1) // chunk_size
    
    value_embeddings_list = []
    key_embeddings_list = []
    
    # OLLAMA: Encode all at once (Ollama handles batching internally)
    print(f"\n  Encoding VALUES with Ollama...")
    value_embeddings = get_ollama_embeddings(hypernode_values, ollama_model, ollama_base_url)
    print(f"  ✓ Generated {len(value_embeddings):,} value embeddings")
    
    print(f"\n  Encoding KEYS with Ollama...")
    key_embeddings = get_ollama_embeddings(hypernode_keys, ollama_model, ollama_base_url)
    print(f"  ✓ Generated {len(key_embeddings):,} key embeddings\n")
    
    # Save hypergraph
    output_dir = Path(ontology_dir)
    
    print("Saving hypergraph components...")
    
    # Save facts (HyperEdges)
    facts_file = output_dir / "hypergraph_facts.pkl"
    with open(facts_file, 'wb') as f:
        pickle.dump(all_facts, f)
    print(f"  ✓ Saved facts to {facts_file}")
    
    # Save hypernodes
    hypernodes_file = output_dir / "hypergraph_nodes.pkl"
    with open(hypernodes_file, 'wb') as f:
        pickle.dump(hypernodes, f)
    print(f"  ✓ Saved hypernodes to {hypernodes_file}")
    
    # Save embeddings
    key_emb_file = output_dir / "hypernode_key_embeddings.npy"
    np.save(key_emb_file, key_embeddings)
    print(f"  ✓ Saved key embeddings to {key_emb_file}")
    
    val_emb_file = output_dir / "hypernode_value_embeddings.npy"
    np.save(val_emb_file, value_embeddings)
    print(f"  ✓ Saved value embeddings to {val_emb_file}")
    
    # Update metadata with hypergraph info
    metadata['hypergraph'] = {
        'model': ollama_model if use_ollama else model_name,
        'use_ollama': use_ollama,
        'total_facts': len(all_facts),
        'total_hypernodes': len(hypernodes),
        'embedding_dim': key_embeddings.shape[1],
        'files': {
            'facts': str(facts_file.name),
            'hypernodes': str(hypernodes_file.name),
            'key_embeddings': str(key_emb_file.name),
            'value_embeddings': str(val_emb_file.name)
        }
    }
    
    metadata_file = output_dir / 'ontology_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Updated metadata\n")
    
    print(f"{'='*80}")
    print(f"✅ HyperGraph Build Complete!")
    print(f"{'='*80}")
    print(f"  Ontology: {ontology_name}")
    print(f"  HyperEdges (facts): {len(all_facts):,}")
    print(f"  HyperNodes: {len(hypernodes):,}")
    print(f"  Embedding dimension: {key_embeddings.shape[1]}")
    print(f"  Output: {output_dir}\n")
    
    return metadata


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generic HyperGraph Builder - Works with any parsed ontology (100% local with Ollama)"
    )
    parser.add_argument(
        "--ontology-dir",
        required=True,
        help="Directory containing parsed ontology JSON files"
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text)"
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        default=True,
        help="Use Ollama for embeddings (default: True, always enabled for security)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    build_hypergraph(
        args.ontology_dir, 
        args.model,
        use_ollama=True,  # Force True for security
        ollama_model=args.model,
        ollama_base_url=args.ollama_base_url
    )


if __name__ == "__main__":
    main()
