"""
Parse Gene Ontology (GO) OWL file into structured JSON format.

This script reads go-basic.owl and extracts GO terms with their:
- ID, label, namespace
- Definitions and synonyms
- Relationships (is_a, part_of, regulates, etc.)
- Cross-references (Reactome, PMID, Wikipedia)

Only active (non-obsolete) terms are extracted.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal
from rdflib.term import URIRef
from tqdm import tqdm


# Define namespaces
OBO = Namespace("http://purl.obolibrary.org/obo/")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
GO_NS = Namespace("http://purl.obolibrary.org/obo/go#")


def extract_go_id(uri: URIRef) -> str:
    """Extract GO ID from URI (e.g., GO:0006281)"""
    uri_str = str(uri)
    if "GO_" in uri_str:
        # Format: http://purl.obolibrary.org/obo/GO_0006281
        go_id = uri_str.split("GO_")[-1]
        return f"GO:{go_id}"
    return uri_str


def get_namespace(graph: Graph, term_uri: URIRef) -> str:
    """Get GO namespace (biological_process, molecular_function, cellular_component)"""
    namespace_pred = OBOINOWL.hasOBONamespace
    for ns in graph.objects(term_uri, namespace_pred):
        return str(ns)
    return "unknown"


def get_synonyms(graph: Graph, term_uri: URIRef) -> Dict[str, List[str]]:
    """Extract all synonym types"""
    synonyms = {
        "exact": [],
        "broad": [],
        "narrow": [],
        "related": []
    }
    
    # Exact synonyms
    for syn in graph.objects(term_uri, OBOINOWL.hasExactSynonym):
        synonyms["exact"].append(str(syn))
    
    # Broad synonyms
    for syn in graph.objects(term_uri, OBOINOWL.hasBroadSynonym):
        synonyms["broad"].append(str(syn))
    
    # Narrow synonyms
    for syn in graph.objects(term_uri, OBOINOWL.hasNarrowSynonym):
        synonyms["narrow"].append(str(syn))
    
    # Related synonyms
    for syn in graph.objects(term_uri, OBOINOWL.hasRelatedSynonym):
        synonyms["related"].append(str(syn))
    
    return synonyms


def get_relationships(graph: Graph, term_uri: URIRef) -> Dict[str, List[str]]:
    """Extract all relationship types (is_a, part_of, regulates, etc.)"""
    relationships = {
        "is_a": [],
        "part_of": [],
        "regulates": [],
        "positively_regulates": [],
        "negatively_regulates": [],
        "occurs_in": [],
        "has_part": []
    }
    
    # is_a relationships (rdfs:subClassOf)
    for parent in graph.objects(term_uri, RDFS.subClassOf):
        if isinstance(parent, URIRef) and "GO_" in str(parent):
            parent_id = extract_go_id(parent)
            # Get parent label
            parent_label = None
            for label in graph.objects(parent, RDFS.label):
                parent_label = str(label)
                break
            
            if parent_label:
                relationships["is_a"].append(f"{parent_id} ({parent_label})")
            else:
                relationships["is_a"].append(parent_id)
    
    # Other relationships (part_of, regulates, etc.) are in restrictions
    # We'll extract them from the OWL restrictions
    for restriction in graph.objects(term_uri, RDFS.subClassOf):
        if isinstance(restriction, URIRef):
            continue
            
        # Check if it's a restriction
        for prop in graph.objects(restriction, OWL.onProperty):
            prop_str = str(prop)
            
            # Get the object of the restriction
            for obj in graph.objects(restriction, OWL.someValuesFrom):
                if "GO_" in str(obj):
                    obj_id = extract_go_id(obj)
                    
                    # Get object label
                    obj_label = None
                    for label in graph.objects(obj, RDFS.label):
                        obj_label = str(label)
                        break
                    
                    value = f"{obj_id} ({obj_label})" if obj_label else obj_id
                    
                    # Determine relationship type
                    if "part_of" in prop_str or "BFO_0000050" in prop_str:
                        relationships["part_of"].append(value)
                    elif "regulates" in prop_str and "positively" not in prop_str and "negatively" not in prop_str:
                        if "RO_0002211" in prop_str:
                            relationships["regulates"].append(value)
                    elif "positively_regulates" in prop_str or "RO_0002213" in prop_str:
                        relationships["positively_regulates"].append(value)
                    elif "negatively_regulates" in prop_str or "RO_0002212" in prop_str:
                        relationships["negatively_regulates"].append(value)
                    elif "occurs_in" in prop_str or "BFO_0000066" in prop_str:
                        relationships["occurs_in"].append(value)
                    elif "has_part" in prop_str or "BFO_0000051" in prop_str:
                        relationships["has_part"].append(value)
    
    # Remove empty lists
    relationships = {k: v for k, v in relationships.items() if v}
    
    return relationships


def get_cross_references(graph: Graph, term_uri: URIRef) -> Dict[str, List[str]]:
    """Extract cross-references (Reactome, PMID, Wikipedia, etc.)"""
    xrefs = {}
    
    for xref in graph.objects(term_uri, OBOINOWL.hasDbXref):
        xref_str = str(xref)
        
        # Parse xref format: "database:id"
        if ":" in xref_str:
            parts = xref_str.split(":", 1)
            database = parts[0]
            ref_id = parts[1] if len(parts) > 1 else xref_str
            
            if database not in xrefs:
                xrefs[database] = []
            xrefs[database].append(ref_id)
    
    return xrefs


def is_obsolete(graph: Graph, term_uri: URIRef) -> bool:
    """Check if term is obsolete"""
    for deprecated in graph.objects(term_uri, OWL.deprecated):
        if str(deprecated).lower() == "true":
            return True
    return False


def parse_go_term(graph: Graph, term_uri: URIRef) -> Dict[str, Any]:
    """Parse a single GO term into structured format"""
    
    # Skip if obsolete
    if is_obsolete(graph, term_uri):
        return None
    
    term_data = {
        "id": extract_go_id(term_uri),
        "uri": str(term_uri)
    }
    
    # Label
    for label in graph.objects(term_uri, RDFS.label):
        term_data["label"] = str(label)
        break
    
    # Namespace
    term_data["namespace"] = get_namespace(graph, term_uri)
    
    # Definition
    for definition in graph.objects(term_uri, OBOINOWL.hasDefinition):
        term_data["definition"] = str(definition)
        break
    
    # Synonyms
    synonyms = get_synonyms(graph, term_uri)
    if any(synonyms.values()):
        term_data["synonyms"] = synonyms
    
    # Relationships
    relationships = get_relationships(graph, term_uri)
    if relationships:
        term_data["relationships"] = relationships
    
    # Cross-references
    xrefs = get_cross_references(graph, term_uri)
    if xrefs:
        term_data["cross_references"] = xrefs
    
    return term_data


def parse_go_owl_streaming(owl_file: str, output_dir: str):
    """
    Parse GO OWL file using streaming approach for better memory efficiency.
    Uses simple regex parsing instead of full RDF graph loading.
    
    Args:
        owl_file: Path to go-basic.owl
        output_dir: Directory to save JSON files
    """
    import re
    
    print(f"Parsing GO ontology from {owl_file} (streaming mode)...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    current_term = None
    active_count = 0
    obsolete_count = 0
    total_terms = 0
    
    with open(owl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Start of a new term
            if '<owl:Class rdf:about="http://purl.obolibrary.org/obo/GO_' in line:
                # Save previous term if exists
                if current_term and not current_term.get('obsolete', False):
                    go_id = current_term["id"].replace(":", "_")
                    output_file = os.path.join(output_dir, f"go_term_{go_id}.json")
                    
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        json.dump(current_term, out_f, indent=2, ensure_ascii=False)
                    
                    active_count += 1
                    if active_count % 1000 == 0:
                        print(f"  Processed {active_count} active terms...")
                
                elif current_term and current_term.get('obsolete', False):
                    obsolete_count += 1
                
                # Extract GO ID
                match = re.search(r'GO_(\d+)', line)
                if match:
                    go_id = f"GO:{match.group(1)}"
                    current_term = {
                        "id": go_id,
                        "uri": f"http://purl.obolibrary.org/obo/GO_{match.group(1)}",
                        "relationships": {},
                        "synonyms": {"exact": [], "broad": [], "narrow": [], "related": []},
                        "cross_references": {}
                    }
                    total_terms += 1
            
            elif current_term:
                # Label
                if '<rdfs:label rdf:datatype=' in line or '<rdfs:label>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["label"] = match.group(1)
                
                # Namespace
                elif '<oboInOwl:hasOBONamespace>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["namespace"] = match.group(1)
                
                # Definition
                elif '<obo:IAO_0000115>' in line or '<oboInOwl:hasDefinition>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["definition"] = match.group(1)
                
                # Obsolete
                elif '<owl:deprecated rdf:datatype=' in line:
                    if '>true<' in line:
                        current_term["obsolete"] = True
                
                # Exact synonym
                elif '<oboInOwl:hasExactSynonym>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["synonyms"]["exact"].append(match.group(1))
                
                # Broad synonym
                elif '<oboInOwl:hasBroadSynonym>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["synonyms"]["broad"].append(match.group(1))
                
                # Narrow synonym
                elif '<oboInOwl:hasNarrowSynonym>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["synonyms"]["narrow"].append(match.group(1))
                
                # Related synonym
                elif '<oboInOwl:hasRelatedSynonym>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        current_term["synonyms"]["related"].append(match.group(1))
                
                # is_a relationship
                elif '<rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/GO_' in line:
                    match = re.search(r'GO_(\d+)', line)
                    if match:
                        parent_id = f"GO:{match.group(1)}"
                        if "is_a" not in current_term["relationships"]:
                            current_term["relationships"]["is_a"] = []
                        current_term["relationships"]["is_a"].append(parent_id)
                
                # Cross-references
                elif '<oboInOwl:hasDbXref>' in line:
                    match = re.search(r'>([^<]+)<', line)
                    if match:
                        xref = match.group(1)
                        if ":" in xref:
                            parts = xref.split(":", 1)
                            database = parts[0]
                            ref_id = parts[1] if len(parts) > 1 else xref
                            
                            if database not in current_term["cross_references"]:
                                current_term["cross_references"][database] = []
                            current_term["cross_references"][database].append(ref_id)
    
    # Save last term
    if current_term and not current_term.get('obsolete', False):
        go_id = current_term["id"].replace(":", "_")
        output_file = os.path.join(output_dir, f"go_term_{go_id}.json")
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Clean up empty fields
            if not current_term["synonyms"]["exact"] and not current_term["synonyms"]["broad"] and \
               not current_term["synonyms"]["narrow"] and not current_term["synonyms"]["related"]:
                del current_term["synonyms"]
            
            if not current_term["relationships"]:
                del current_term["relationships"]
            
            if not current_term["cross_references"]:
                del current_term["cross_references"]
            
            json.dump(current_term, out_f, indent=2, ensure_ascii=False)
        
        active_count += 1
    
    elif current_term and current_term.get('obsolete', False):
        obsolete_count += 1
    
    print(f"\n✅ Parsing complete!")
    print(f"   Total terms found: {total_terms}")
    print(f"   Active terms: {active_count}")
    print(f"   Obsolete terms (skipped): {obsolete_count}")
    print(f"   Output directory: {output_dir}")


def parse_go_owl(owl_file: str, output_dir: str):
    """Wrapper to use streaming parser"""
    parse_go_owl_streaming(owl_file, output_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse GO OWL file")
    parser.add_argument(
        "--owl-file",
        default="data/kg/go/go-basic.owl",
        help="Path to go-basic.owl file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/kg/go/ontology",
        help="Output directory for JSON files"
    )
    
    args = parser.parse_args()
    
    parse_go_owl(args.owl_file, args.output_dir)


if __name__ == "__main__":
    main()
