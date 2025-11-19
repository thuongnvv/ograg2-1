#!/usr/bin/env python3
"""
Generic HyperGraph Builder - Works with ANY parsed ontology!

Auto-detects ontology structure and builds hypergraph accordingly.
No configuration needed - just point to parsed ontology directory.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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
    if 'definition' in term_data:
        core_parts.append(f"Definition: {term_data['definition']}")
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


def build_hypergraph(ontology_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Build hypergraph from parsed ontology
    
    Args:
        ontology_dir: Directory containing parsed term JSON files
        model_name: Embedding model (default: MiniLM for speed)
    """
    print(f"\n{'='*80}")
    print(f"Generic HyperGraph Builder")
    print(f"{'='*80}\n")
    
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
    
    # Load embedding model
    print(f"\nLoading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  ✓ Model loaded (dim: {model.get_sentence_embedding_dimension()})\n")
    
    # Load all term files
    ontology_files = sorted(Path(ontology_dir).glob("term_*.json"))
    print(f"Found {len(ontology_files)} term files\n")
    
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
    
    # STEP 1: Encode VALUES
    print(f"\n  Encoding VALUES - {num_chunks} chunks...")
    for chunk_idx in tqdm(range(num_chunks), desc="VALUE chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(hypernodes))
        
        chunk_values = hypernode_values[start_idx:end_idx]
        
        chunk_val_emb = model.encode(
            chunk_values,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        value_embeddings_list.append(chunk_val_emb)
        
        del chunk_values, chunk_val_emb
        import gc
        gc.collect()
    
    value_embeddings = np.vstack(value_embeddings_list)
    print(f"  ✓ Generated {len(value_embeddings):,} value embeddings")
    
    del value_embeddings_list
    import gc
    gc.collect()
    
    # STEP 2: Encode KEYS
    print(f"\n  Encoding KEYS - {num_chunks} chunks...")
    for chunk_idx in tqdm(range(num_chunks), desc="KEY chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(hypernodes))
        
        chunk_keys = hypernode_keys[start_idx:end_idx]
        
        chunk_key_emb = model.encode(
            chunk_keys,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        key_embeddings_list.append(chunk_key_emb)
        
        del chunk_keys, chunk_key_emb
        import gc
        gc.collect()
    
    key_embeddings = np.vstack(key_embeddings_list)
    print(f"  ✓ Generated {len(key_embeddings):,} key embeddings\n")
    
    del key_embeddings_list
    import gc
    gc.collect()
    
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
        'model': model_name,
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
        description="Generic HyperGraph Builder - Works with any parsed ontology"
    )
    parser.add_argument(
        "--ontology-dir",
        required=True,
        help="Directory containing parsed ontology JSON files"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model (default: MiniLM for speed)"
    )
    
    args = parser.parse_args()
    
    build_hypergraph(args.ontology_dir, args.model)


if __name__ == "__main__":
    main()
