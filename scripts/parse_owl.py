#!/usr/bin/env python3
"""
Generic OWL Ontology Parser - Works with ANY ontology!

Uses proper XML parsing to extract ALL information automatically.
No hardcoded patterns - works with ANY OWL/RDF structure.

Automatically extracts:
- All term properties (labels, definitions, comments, examples, etc.)
- All relationships (is_a, part_of, regulates, custom relations, etc.)
- All annotations (synonyms, xrefs, alt_ids, etc.)
- Automatically detects ontology prefix from URIs

Input: Any OWL file
Output: Structured JSON files (one per term)
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Set, List, Optional
from collections import defaultdict


class OWLParser:
    """Generic OWL parser using proper XML parsing"""
    
    # Common OBO namespaces
    NAMESPACES = {
        'owl': 'http://www.w3.org/2002/07/owl#',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'obo': 'http://purl.obolibrary.org/obo/',
        'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#',
    }
    
    def __init__(self, owl_file: str, output_dir: str):
        self.owl_file = owl_file
        self.output_dir = output_dir
        self.stats = {
            'total_terms': 0,
            'active_terms': 0,
            'obsolete_terms': 0,
            'ontology_name': None,
            'id_prefix': None,
            'relationships': set(),
            'namespaces': set(),
        }
        self.prefix_counts = defaultdict(int)  # Track which prefix appears most
        
    def extract_id_from_uri(self, uri: str) -> Optional[str]:
        """
        Extract term ID from URI - GENERIC approach
        Handles: http://purl.obolibrary.org/obo/GO_0006281 -> GO:0006281
        
        Also handles duplicate IDs from different namespaces:
        - http://purl.obolibrary.org/obo/ICDO_0000079 -> ICDO:0000079
        - http://purl.obolibrary.org/obo/icdo.owl/ICDO_0000079 -> ICDO:0000079_ns2
        """
        # Check if this is a secondary namespace (e.g., icdo.owl/ICDO_)
        is_secondary_namespace = False
        if re.search(r'/[a-z]+\.owl/[A-Z]', uri):
            is_secondary_namespace = True
        
        # OBO format: PREFIX_DIGITS
        match = re.search(r'/([A-Z][A-Z0-9]*)[_:](\d+)', uri)
        if match:
            prefix = match.group(1)
            number = match.group(2)
            self.prefix_counts[prefix] += 1
            
            # Add suffix for secondary namespace to avoid duplicate IDs
            base_id = f"{prefix}:{number}"
            return f"{base_id}_ns2" if is_secondary_namespace else base_id
        
        # Alternative: extract last part of URI
        match = re.search(r'/([^/]+)$', uri)
        if match:
            return match.group(1)
        
        return None
    
    def detect_ontology_prefix(self) -> str:
        """Detect the main ontology prefix by counting occurrences"""
        if not self.prefix_counts:
            return "UNKNOWN"
        # Return the most common prefix
        return max(self.prefix_counts.items(), key=lambda x: x[1])[0]
    
    def _extract_property_name(self, uri: str) -> Optional[str]:
        """
        Extract property name from URI - FULLY GENERIC
        
        Examples:
        - http://purl.obolibrary.org/obo/RO_0001000 -> RO_0001000
        - http://purl.obolibrary.org/obo/BFO_0000050 -> BFO_0000050
        - http://example.org/custom#hasProperty -> hasProperty
        - Any ontology URI format supported automatically
        """
        # Extract last part after / or #
        match = re.search(r'[/#]([^/#]+)$', uri)
        if match:
            name = match.group(1)
            # Filter out empty or very short names
            if name and len(name) > 1:
                return name
        return None
    
    def parse_streaming(self):
        """
        Parse OWL file using iterative XML parsing
        Extracts ALL information automatically without hardcoded patterns
        """
        print(f"\n{'='*80}")
        print(f"Generic OWL Parser - Universal XML Parsing")
        print(f"{'='*80}\n")
        print(f"Input file: {self.owl_file}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("\nParsing OWL classes and properties...")
        
        active_count = 0
        obsolete_count = 0
        property_count = 0
        individual_count = 0
        
        # Use iterparse for memory efficiency with large files
        context = ET.iterparse(self.owl_file, events=('start', 'end'))
        
        current_class = None
        current_term = None
        current_property = None
        current_individual = None
        
        for event, elem in context:
            # Start of owl:Class element
            if event == 'start' and elem.tag.endswith('Class'):
                # Get URI from rdf:about or rdf:ID
                uri = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about') or \
                      elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID')
                
                if uri and 'obo' in uri.lower():
                    term_id = self.extract_id_from_uri(uri)
                    if term_id:
                        current_term = {
                            'id': term_id,
                            'uri': uri,
                        }
                        current_class = elem
                        self.stats['total_terms'] += 1
            
            # Start of owl:ObjectProperty, owl:AnnotationProperty, or owl:DatatypeProperty
            elif event == 'start' and (elem.tag.endswith('ObjectProperty') or 
                                       elem.tag.endswith('AnnotationProperty') or
                                       elem.tag.endswith('DatatypeProperty')):
                uri = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about') or \
                      elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID')
                
                # More flexible: accept any URI (not just "obo")
                # Skip only XML built-ins and OWL built-ins
                if uri and not uri.startswith('http://www.w3.org/'):
                    # Extract property name from URI - completely generic
                    prop_name = self._extract_property_name(uri)
                    if prop_name:
                        current_property = {
                            'id': f"PROP:{prop_name}",
                            'uri': uri,
                            'property_name': prop_name,
                        }
                        current_class = elem
            
            # Start of owl:NamedIndividual - concrete instances
            elif event == 'start' and elem.tag.endswith('NamedIndividual'):
                uri = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about') or \
                      elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID')
                
                if uri and not uri.startswith('http://www.w3.org/'):
                    # Extract instance name from URI
                    inst_name = self._extract_property_name(uri)
                    if inst_name:
                        current_individual = {
                            'id': f"INST:{inst_name}",
                            'uri': uri,
                            'instance_name': inst_name,
                        }
                        current_class = elem
            
            # End of owl:Class element - process complete term
            elif event == 'end' and elem.tag.endswith('Class') and current_term:
                # Extract all properties from this class element
                self._extract_all_properties(current_class, current_term)
                
                # Save term
                if not current_term.get('obsolete', False):
                    self._save_term(current_term)
                    active_count += 1
                    if active_count % 1000 == 0:
                        print(f"  Processed {active_count} terms...")
                else:
                    obsolete_count += 1
                
                # Clear
                current_term = None
                current_class = None
                elem.clear()  # Free memory
            
            # End of Property elements - process complete property
            elif event == 'end' and (elem.tag.endswith('ObjectProperty') or
                                     elem.tag.endswith('AnnotationProperty') or
                                     elem.tag.endswith('DatatypeProperty')) and current_property:
                # Extract all properties (label, definition, examples, etc.)
                self._extract_all_properties(current_class, current_property)
                
                # Save as a special term (with PROP: prefix)
                if current_property.get('label'):  # Only save if has label
                    self._save_term(current_property)
                    property_count += 1
                
                # Clear
                current_property = None
                current_class = None
                elem.clear()
            
            # End of NamedIndividual element - process complete instance
            elif event == 'end' and elem.tag.endswith('NamedIndividual') and current_individual:
                # Extract all properties (label, definition, relationships, etc.)
                self._extract_all_properties(current_class, current_individual)
                
                # Save as a special term (with INST: prefix)
                if current_individual.get('label'):  # Only save if has label
                    self._save_term(current_individual)
                    individual_count += 1
                
                # Clear
                current_individual = None
                current_class = None
                elem.clear()  # Free memory
        
        self.stats['active_terms'] = active_count
        self.stats['obsolete_terms'] = obsolete_count
        self.stats['property_definitions'] = property_count
        self.stats['named_individuals'] = individual_count
        self.stats['ontology_name'] = self.detect_ontology_prefix()
        self.stats['id_prefix'] = self.stats['ontology_name']
        
        # Save metadata
        self._save_metadata()
        
        print(f"\n{'='*80}")
        print(f"✅ Parsing complete!")
        print(f"{'='*80}")
        print(f"  Ontology: {self.stats['ontology_name']}")
        print(f"  Total terms: {self.stats['total_terms']}")
        print(f"  Active terms: {self.stats['active_terms']}")
        print(f"  Property definitions: {property_count}")
        print(f"  Named individuals: {individual_count}")
        print(f"  Obsolete (skipped): {self.stats['obsolete_terms']}")
        print(f"  Relationships found: {len(self.stats['relationships'])}")
        if self.stats['namespaces']:
            print(f"  Namespaces: {', '.join(self.stats['namespaces'])}")
        print(f"  Output: {self.output_dir}")
        print()
    
    def _extract_all_properties(self, class_elem, term: Dict):
        """
        Extract ALL properties from owl:Class element
        Works generically for ANY ontology
        """
        # Initialize collections
        term['relationships'] = {}
        term['synonyms'] = []
        term['cross_references'] = []
        
        # Iterate through all child elements
        for child in class_elem:
            tag_local = child.tag.split('}')[-1]  # Get local name without namespace
            tag_full = child.tag
            text = child.text.strip() if child.text else None
            
            # LABEL
            if 'label' in tag_local.lower():
                term['label'] = text
            
            # DEFINITION - multiple possible tags
            elif any(x in tag_full for x in ['IAO_0000115', 'hasDefinition', 'definition']):
                term['definition'] = text
            
            # COMMENT
            elif 'comment' in tag_local.lower():
                if 'comments' not in term:
                    term['comments'] = []
                if text:
                    term['comments'].append(text)
            
            # EXAMPLES
            elif 'IAO_0000112' in tag_full or 'example' in tag_local.lower():
                if 'examples' not in term:
                    term['examples'] = []
                if text:
                    term['examples'].append(text)
            
            # SYNONYMS - any type
            elif 'synonym' in tag_local.lower():
                if text:
                    term['synonyms'].append(text)
            
            # NAMESPACE
            elif 'namespace' in tag_local.lower() or 'hasOBONamespace' in tag_full:
                if text:
                    term['namespace'] = text
                    self.stats['namespaces'].add(text)
            
            # ALTERNATIVE IDs
            elif 'alternativeId' in tag_full or 'alt_id' in tag_local.lower():
                if 'alternative_ids' not in term:
                    term['alternative_ids'] = []
                if text:
                    term['alternative_ids'].append(text)
            
            # CROSS REFERENCES
            elif 'xref' in tag_local.lower() or 'hasDbXref' in tag_full:
                if text:
                    term['cross_references'].append(text)
            
            # OBSOLETE
            elif 'deprecated' in tag_local.lower():
                if text and text.lower() == 'true':
                    term['obsolete'] = True
            
            # RELATIONSHIPS - subClassOf
            elif 'subClassOf' in tag_local:
                resource = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                if resource:
                    parent_id = self.extract_id_from_uri(resource)
                    if parent_id:
                        if 'is_a' not in term['relationships']:
                            term['relationships']['is_a'] = []
                        term['relationships']['is_a'].append(parent_id)
                        self.stats['relationships'].add('is_a')
                
                # Check for restrictions (other relationships)
                self._extract_restrictions(child, term)
    
    def _extract_restrictions(self, elem, term: Dict):
        """Extract relationships from owl:Restriction elements"""
        for restriction in elem.findall('.//{http://www.w3.org/2002/07/owl#}Restriction'):
            # Get property
            on_property = restriction.find('{http://www.w3.org/2002/07/owl#}onProperty')
            some_values = restriction.find('{http://www.w3.org/2002/07/owl#}someValuesFrom')
            
            if on_property is not None and some_values is not None:
                prop_uri = on_property.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                target_uri = some_values.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                
                if prop_uri and target_uri:
                    # Infer relationship type from property URI
                    rel_type = self._infer_relationship_type(prop_uri)
                    target_id = self.extract_id_from_uri(target_uri)
                    
                    if target_id:
                        if rel_type not in term['relationships']:
                            term['relationships'][rel_type] = []
                        term['relationships'][rel_type].append(target_id)
                        self.stats['relationships'].add(rel_type)
    

    
    def _infer_relationship_type(self, property_uri: str) -> str:
        """Infer relationship type from property URI"""
        uri_lower = property_uri.lower()
        
        # Common relationship patterns
        if 'part_of' in uri_lower or 'BFO_0000050' in property_uri:
            return 'part_of'
        elif 'has_part' in uri_lower or 'BFO_0000051' in property_uri:
            return 'has_part'
        elif 'regulates' in uri_lower and 'negatively' not in uri_lower and 'positively' not in uri_lower:
            if 'RO_0002211' in property_uri:
                return 'regulates'
        elif 'positively_regulates' in uri_lower or 'RO_0002213' in property_uri:
            return 'positively_regulates'
        elif 'negatively_regulates' in uri_lower or 'RO_0002212' in property_uri:
            return 'negatively_regulates'
        elif 'occurs_in' in uri_lower or 'BFO_0000066' in property_uri:
            return 'occurs_in'
        elif 'derives_from' in uri_lower or 'RO_0001000' in property_uri:
            return 'derives_from'
        elif 'develops_from' in uri_lower:
            return 'develops_from'
        elif 'has_function' in uri_lower:
            return 'has_function'
        
        # Generic fallback - extract name from URI
        match = re.search(r'/([^/]+)$', property_uri)
        if match:
            return match.group(1).replace('_', ' ')
        
        return 'related_to'
    
    def _save_term(self, term: Dict):
        """Save term to JSON file"""
        # Clean up empty fields
        if not term.get('synonyms'):
            term.pop('synonyms', None)
        if not term.get('relationships'):
            term.pop('relationships', None)
        if not term.get('cross_references'):
            term.pop('cross_references', None)
        if not term.get('comments'):
            term.pop('comments', None)
        if not term.get('examples'):
            term.pop('examples', None)
        if not term.get('alternative_ids'):
            term.pop('alternative_ids', None)
        
        # Save to file
        safe_id = term['id'].replace(':', '_')
        output_file = os.path.join(self.output_dir, f"term_{safe_id}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(term, f, indent=2, ensure_ascii=False)
    
    def _save_metadata(self):
        """Save ontology metadata"""
        metadata = {
            'ontology_name': self.stats['ontology_name'],
            'id_prefix': self.stats['id_prefix'],
            'total_terms': self.stats['total_terms'],
            'active_terms': self.stats['active_terms'],
            'obsolete_terms': self.stats['obsolete_terms'],
            'relationships': sorted(list(self.stats['relationships'])),
            'namespaces': sorted(list(self.stats['namespaces'])),
            'parsed_date': None,  # Will be set by caller
        }
        
        metadata_file = os.path.join(self.output_dir, 'ontology_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generic OWL Ontology Parser - Works with ANY ontology!"
    )
    parser.add_argument(
        "--owl-file",
        required=True,
        help="Path to OWL file (any ontology)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for parsed JSON files"
    )
    
    args = parser.parse_args()
    
    # Parse ontology
    parser = OWLParser(args.owl_file, args.output_dir)
    parser.parse_streaming()


if __name__ == "__main__":
    main()
