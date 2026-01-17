#!/usr/bin/env python3
"""
Ontology Generator - Auto-generate OWL from documents/URLs using LLM

Features:
- Extract text from PDF, DOCX, web URLs
- Generate OWL ontology using LLM
- Basic validation
"""

import re
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Document parsers
from pypdf import PdfReader
from docx import Document
import requests
from bs4 import BeautifulSoup

# LLM
from openai import OpenAI


def chunk_text(text: str, chunk_size: int = 30000, overlap: int = 1000) -> list:
    """
    Split large text into overlapping chunks for processing.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk (default: 30000)
        overlap: Characters to overlap between chunks (default: 1000)
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a natural boundary (sentence end, paragraph)
        if end < len(text):
            # Look for sentence end (. ? !) within last 500 chars
            search_start = max(end - 500, start)
            best_break = end
            
            for i in range(end, search_start, -1):
                if text[i-1] in '.?!\n' and (i >= len(text) or text[i] in ' \n'):
                    best_break = i
                    break
            
            end = best_break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


class OntologyGenerator:
    """Generate OWL ontology from documents using LLM"""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4",
                 use_ollama: bool = False,
                 ollama_base_url: str = "http://localhost:11434/v1",
                 base_url: str = None,
                 ontology_mode: str = "structured"):
        """
        Initialize generator
        
        Args:
            api_key: OpenAI API key (required if use_ollama=False)
            model: Model name (gpt-4, gpt-3.5-turbo, or Ollama model)
            use_ollama: Use Ollama instead of OpenAI API
            ollama_base_url: Ollama server URL (only if use_ollama=True)
            base_url: Custom API base URL (for OpenAI-compatible APIs)
            ontology_mode: "flat" for old FAQ style, "structured" for DSL-compatible (default)
        """
        if use_ollama:
            self.llm_client = OpenAI(
                api_key="ollama",
                base_url=ollama_base_url
            )
        else:
            if not api_key:
                raise ValueError("api_key required when use_ollama=False")
            # Support custom base URL for OpenAI-compatible APIs
            kwargs = {
                "api_key": api_key,
                "timeout": 6000.0,  
                "max_retries": 5
            }
            if base_url:
                kwargs["base_url"] = base_url
            self.llm_client = OpenAI(**kwargs)
        
        self.model = model
        self.use_ollama = use_ollama
        self.ontology_mode = ontology_mode  # "flat" or "structured"
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with fallback options"""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    # Skip problematic pages
                    continue
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {e}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF. It may be scanned/image-based or corrupted.")
        
        return text.strip()
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    
    def extract_text_from_url(self, url: str) -> str:
        """
        Extract text from web page with adaptive strategy selection
        
        NEW APPROACH (v2):
        - Auto-detect site type (static HTML, SPA, dynamic, PDF, API)
        - Select optimal scraping strategy
        - Validate content quality
        - Intelligent retry with fallback
        """
        from utils.url_analyzer import URLAnalyzer
        from utils.scraping_strategies import create_strategy, ScrapingConfig
        from utils.content_validator import ContentValidator
        import time
        
        print(f"🔍 Analyzing URL: {url}")
        
        # Step 1: Analyze URL to determine optimal strategy
        analyzer = URLAnalyzer(timeout=10)
        analysis = analyzer.analyze(url)
        
        print(f"   Site Type: {analysis.site_type.value}")
        print(f"   Protection: {analysis.protection_type.value}")
        print(f"   Recommended Strategy: {analysis.recommended_strategy}")
        print(f"   Estimated Timeout: {analysis.estimated_timeout}s")
        
        # Check robots.txt
        if not analysis.is_robots_allowed:
            print(f"   ⚠️  Warning: robots.txt may disallow scraping")
        
        # Step 2: Create scraping configuration
        config = ScrapingConfig(
            timeout=analysis.estimated_timeout,
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            wait_time=2,
            max_retries=3,
            verify_ssl=True
        )
        
        # Step 3: Try primary strategy with retry
        max_attempts = config.max_retries
        last_error = None
        text = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"   📥 Attempt {attempt}/{max_attempts} with {analysis.recommended_strategy} strategy...")
                
                # Create and execute strategy
                strategy = create_strategy(analysis.recommended_strategy, config)
                text = strategy.extract(url)
                
                print(f"   ✓ Extracted {len(text)} characters")
                
                # Step 4: Validate content quality
                validator = ContentValidator(
                    min_length=200,
                    min_quality_score=0.3,
                    max_boilerplate_ratio=0.7
                )
                
                validation = validator.validate(text, url)
                
                if validation.is_valid:
                    print(f"   ✓ Content quality: {validation.quality_score:.2f}")
                    # Use cleaned text if available
                    return validation.cleaned_text or text
                else:
                    print(f"   ⚠️  Content validation issues:")
                    for issue in validation.issues:
                        print(f"       - {issue}")
                    
                    # If this is NOT the last attempt, try fallback strategy
                    if attempt < max_attempts:
                        print(f"   🔄 Trying fallback strategy...")
                        # Try hybrid strategy as fallback
                        if analysis.recommended_strategy != 'hybrid':
                            analysis.recommended_strategy = 'hybrid'
                            config.timeout = 25
                            continue
                    
                    # Last attempt: accept if we have some content
                    if validation.quality_score >= 0.2 and len(text) >= 100:
                        print(f"   ⚠️  Accepting marginal content (score: {validation.quality_score:.2f})")
                        return validation.cleaned_text or text
                    
                    last_error = f"Content quality too low: {validation.quality_score:.2f}"
                
            except Exception as e:
                last_error = str(e)
                print(f"   ⚠️  Attempt {attempt} failed: {e}")
                
                # Exponential backoff
                if attempt < max_attempts:
                    wait_time = 2 ** attempt
                    print(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        # All attempts failed
        if text and len(text) >= 50:
            # We have SOME text, return it despite validation failure
            print(f"   ⚠️  Returning partial content ({len(text)} chars) after all retries failed")
            return text
        
        # Complete failure
        raise ValueError(
            f"Failed to extract sufficient content from URL after {max_attempts} attempts. "
            f"Last error: {last_error}. "
            f"The page may require JavaScript, have anti-scraping protection, or be unavailable."
        )

    
    def generate_ontology(self, text: str, domain: str = "general") -> str:
        """
        Generate OWL ontology from text using LLM (2-Step Process)
        
        Modes:
        - "flat": Old FAQ style (flat classes only)
        - "structured": 2-Step Process (Schema Discovery -> Ontology Generation)
        
        Args:
            text: Input text to analyze
            domain: Domain of the ontology
            
        Returns:
            OWL/XML string
        """
        # Mode 1: FLAT (Baseline)
        if self.ontology_mode == "flat":
            print(f"   Mode: FLAT (baseline)")
            prompt = self._get_flat_prompt(text, domain)
            return self._call_llm(prompt)

        # Mode 2: STRUCTURED (2-Step Adaptive)
        print(f"   Mode: STRUCTURED (2-Step Adaptive)")
        
        # Step 1: Schema Discovery
        print("   📝 Step 1: Discovering Ontology Schema...")
        schema_prompt = self._get_schema_discovery_prompt(text, domain)
        schema_response = self._call_llm(schema_prompt)
        
        schema = None
        schema_raw = schema_response
        try:
            # Extract JSON from response
            if "```json" in schema_response:
                schema_json = re.search(r'```json\s*(.*?)\s*```', schema_response, re.DOTALL).group(1)
            elif "```" in schema_response:
                schema_json = re.search(r'```\s*(.*?)\s*```', schema_response, re.DOTALL).group(1)
            else:
                schema_json = schema_response
                
            schema = json.loads(schema_json)
            print(f"      ✓ Schema discovered: {len(schema.get('classes', []))} classes, {len(schema.get('relationships', []))} props")
            
        except Exception as e:
            print(f"      ⚠️ Schema parsing failed: {e}. Falling back to raw text.")
            schema = schema_response # Use raw text as schema if JSON fails

        # Step 2: Ontology Generation
        print("   🏗️  Step 2: Building Ontology based on Schema...")
        generation_prompt = self._get_ontology_generation_prompt(text, domain, schema)
        owl_content = self._call_llm(generation_prompt, is_xml=True)
        
        # Store schema for later access
        self.last_discovered_schema = schema if isinstance(schema, dict) else {"raw": schema_raw}
        
        return owl_content

    def _call_llm(self, prompt: str, is_xml: bool = False) -> str:
        """Helper to call LLM and handle basic cleanup"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert ontology engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50000
            )
            
            content = response.choices[0].message.content.strip()
            
            if is_xml:
                # Extract XML
                if "```xml" in content:
                    content = re.search(r'```xml\s*(.*?)\s*```', content, re.DOTALL).group(1)
                elif "```" in content:
                    content = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL).group(1)
                
                # Ensure XML declaration
                if not content.startswith("<?xml"):
                    content = '<?xml version="1.0"?>\n' + content
                
                # Auto-fix missing Ontology tag
                if "<owl:Ontology" not in content and "<rdf:RDF" in content:
                    insert_pos = content.find('>') + 1
                    content = content[:insert_pos] + '\n  <owl:Ontology rdf:about=""/>' + content[insert_pos:]
                
                # Sanitize XML: Escape unescaped ampersands
                # Finds '&' not followed by a valid entity reference
                content = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', content)
            
            return content
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            raise

    def _get_flat_prompt(self, text: str, domain: str) -> str:
        """Get FLAT FAQ prompt (no hierarchy/relationships)"""
        return f"""Convert text to OWL ontology. Create ONE owl:Class per distinct concept/question.

DOMAIN: {domain}

TEXT:
{text}

RULES:
1. For Q&A content: One Class per question, label=question, comment=answer
2. For general content: One Class per concept/entity
3. Every Class MUST have:
   - rdfs:label (the term/question)
   - rdfs:comment (full definition/answer - include ALL details)
4. ID format: #FAQ_TopicName or #Concept_Name
5. NO owl:ObjectProperty, NO relationships, NO hierarchy
6. Capture KEY PHRASES as separate classes with exact text

TEMPLATE:
<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/ontology#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
  <owl:Ontology rdf:about=""/>
  
  <owl:Class rdf:about="#FAQ_ExampleQuestion">
    <rdfs:label>What is X?</rdfs:label>
    <rdfs:comment>X is... [complete answer]</rdfs:comment>
  </owl:Class>
  
</rdf:RDF>

Output ONLY raw XML. Begin with <?xml"""
    
    def _get_schema_discovery_prompt(self, text: str, domain: str) -> str:
        """Step 1: Analyze text and design Ontology Schema"""
        return f"""<role>You are a senior knowledge engineer creating an ontology for Q&A. Extract ALL information users might ask about.</role>

<context>
DOMAIN: {domain}
</context>

<source_text>
{text}
</source_text>

<task>
Analyze text and extract EVERYTHING. Think step-by-step:

STEP 1 - ENTITIES: Organizations, documents, tools, roles, processes, concepts
STEP 2 - RELATIONSHIPS: How entities connect (creates, uses, requires, part-of)
STEP 3 - ATTRIBUTES: Properties with values (names, versions, durations, flags)
STEP 4 - KEY PHRASES: Exact terms users will search for (definitions, slogans)
STEP 5 - DENIALS: What something CANNOT, DOES NOT, or IS NOT
STEP 6 - GOALS/PURPOSES: Why something exists, what it aims to achieve
STEP 7 - OPERATIONS: Activities, processes, how things work
STEP 8 - RECOMMENDATIONS: Advice, guidance, best practices
STEP 9 - CONDITIONS: When/if something applies (conditions, prerequisites)
STEP 10 - COMPARISONS: Differences between things (X vs Y, unlike Z)
STEP 11 - QUANTITIES: Numbers, amounts, percentages mentioned
STEP 12 - FAQ_PAIRS: Question-answer patterns found in text
</task>

<critical_rules>
- Extract ALL factual statements, not just main concepts
- Capture operational details (how, why, what for)
- Include purposes, goals, activities
- Every denial/limitation must be captured
- Key phrases = exact text users will search
</critical_rules>

<output_format>
{{
  "classes": [
    {{"name": "EntityName", "definition": "Full description", "is_not": "What it is NOT (if stated)"}}
  ],
  "relationships": [
    {{"name": "relationName", "domain": "Subject", "range": "Object", "description": "Meaning"}}
  ],
  "attributes": [
    {{"name": "attrName", "type": "string|number|boolean", "applies_to": "ClassName"}}
  ],
  "key_phrases": ["phrase 1", "phrase 2"],
  "denials": ["X does not Y", "X is not Z"],
  "goals": ["purpose 1", "objective 2"],
  "operations": ["activity 1", "process 2"],
  "recommendations": ["should do X", "best practice Y"],
  "conditions": ["if X then Y", "when Z applies"],
  "comparisons": ["X differs from Y in...", "unlike Z"],
  "quantities": ["N percent", "amount of X"],
  "faq_pairs": [{{"q": "question", "a": "answer"}}]
}}
</output_format>

Output ONLY valid JSON."""

    def _get_ontology_generation_prompt(self, text: str, domain: str, schema: Any) -> str:
        """Step 2: Generate OWL based on Schema"""
        schema_json = json.dumps(schema, indent=2) if isinstance(schema, dict) else schema
        return f"""<role>You are an OWL ontology expert. Convert the schema into a complete, searchable ontology.</role>

<schema>
{schema_json}
</schema>

<source_text>
{text}
</source_text>

<instructions>
Create OWL elements:

1. **owl:Class** - Each concept with FULL definition in rdfs:comment
2. **owl:ObjectProperty** - Relationships with domain/range  
3. **owl:DatatypeProperty** - Attributes with xsd types
4. **owl:NamedIndividual** - Include:
   - DENIALS: Things entity does NOT do (prefix: DENIAL_)
   - KEY_PHRASES: Exact searchable terms (prefix: KP_)
   - EXAMPLES: Specific instances from text (prefix: EX_)
</instructions>

<critical_rules>
RULE 1: rdfs:label = SHORT name (2-5 words)
RULE 2: rdfs:comment = FULL DESCRIPTION from source text (1-3 sentences minimum!)
RULE 3: rdfs:comment MUST BE DIFFERENT AND LONGER than rdfs:label
RULE 4: COPY explanations from source text into rdfs:comment verbatim
RULE 5: Never leave rdfs:comment empty or identical to label
</critical_rules>

<wrong_example>
<!-- WRONG: comment same as label -->
<owl:NamedIndividual rdf:about="#KP_Something">
  <rdfs:label>Something</rdfs:label>
  <rdfs:comment>Something</rdfs:comment>  <!-- BAD! -->
</owl:NamedIndividual>
</wrong_example>

<correct_example>
<!-- CORRECT: comment has FULL description from source -->
<owl:NamedIndividual rdf:about="#KP_Something">
  <rdfs:label>Something</rdfs:label>
  <rdfs:comment>Something is a concept that represents X. It is used for Y and provides Z functionality according to the source text.</rdfs:comment>
</owl:NamedIndividual>
</correct_example>

<template>
<?xml version="1.0"?>
<rdf:RDF xmlns="http://example.org/ontology#"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
xmlns:owl="http://www.w3.org/2002/07/owl#"
xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
  <owl:Ontology rdf:about=""/>
  <!-- Your OWL content here -->
</rdf:RDF>
</template>

Output ONLY raw XML starting with <?xml"""
    
    def fix_xml_issues(self, owl_content: str) -> str:
        """
        Auto-fix common XML issues in LLM-generated OWL
        
        Common issues:
        - Unescaped & (should be &amp;)
        - Unescaped < > in text content
        - Missing closing tags
        """
        import re
        
        fixed = owl_content
        
        # Fix unescaped & (but not already escaped like &amp; &lt; &gt; &quot; &apos;)
        # Match & not followed by amp; lt; gt; quot; apos; #
        fixed = re.sub(r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', fixed)
        
        # Fix unescaped < in text content (not tag start)
        # This is tricky - look for < not followed by valid tag chars or /
        # For now, just fix common patterns like "A < B" or "A<B" that aren't tags
        
        # Fix common Unicode issues that cause XML parsing errors
        # Remove control characters (except tab, newline, carriage return)
        fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)
        
        # Ensure proper closing
        if not fixed.rstrip().endswith('</rdf:RDF>'):
            # Try to find last complete element and close
            if '<rdf:RDF' in fixed and '</rdf:RDF>' not in fixed:
                fixed = fixed.rstrip() + '\n</rdf:RDF>'
        
        return fixed
    
    def validate_owl(self, owl_content: str) -> Dict[str, Any]:
        """
        Basic validation of OWL content with auto-fix attempt
        
        Returns:
            Dict with validation results
        """
        issues = []
        
        # Check XML declaration
        if not owl_content.strip().startswith("<?xml"):
            issues.append("Missing XML declaration")
        
        # Check for essential OWL elements
        if "http://www.w3.org/2002/07/owl#" not in owl_content:
            issues.append("Missing OWL namespace")
        
        if "<owl:Ontology" not in owl_content and "<Ontology" not in owl_content:
            issues.append("Missing Ontology element")
        
        # Check for basic structure
        if "<owl:Class" not in owl_content and "<Class" not in owl_content:
            issues.append("No classes defined")
        
        # Try to parse as XML (basic check)
        fixed_content = owl_content
        try:
            from xml.etree import ElementTree as ET
            ET.fromstring(owl_content)
        except Exception as e:
            # Try auto-fix
            fixed_content = self.fix_xml_issues(owl_content)
            try:
                ET.fromstring(fixed_content)
                issues.append(f"XML auto-fixed: original had {str(e)[:50]}")
            except Exception as e2:
                issues.append(f"XML parsing error: {str(e)[:100]}")
        
        return {
            "valid": len([i for i in issues if "XML parsing error" in i]) == 0,
            "issues": issues,
            "class_count": owl_content.count("<owl:Class") + owl_content.count("<Class"),
            "property_count": owl_content.count("Property"),
            "fixed_content": fixed_content if fixed_content != owl_content else None
        }
    
    def process_file(self, file_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Process uploaded file and generate ontology
        
        Args:
            file_path: Path to PDF, DOCX, or text file
            domain: Domain context
            
        Returns:
            Dict with OWL content and metadata
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        # Extract text based on file type
        if suffix == '.pdf':
            print(f"📄 Extracting text from PDF: {file_path.name}")
            text = self.extract_text_from_pdf(str(file_path))
        elif suffix in ['.docx', '.doc']:
            print(f"📄 Extracting text from DOCX: {file_path.name}")
            text = self.extract_text_from_docx(str(file_path))
        elif suffix == '.txt':
            print(f"📄 Reading text file: {file_path.name}")
            text = file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        if not text or len(text) < 50:
            raise ValueError("Extracted text is too short or empty")
        
        # Generate ontology
        owl_content = self.generate_ontology(text, domain)
        
        # Validate
        validation = self.validate_owl(owl_content)
        
        return {
            "owl_content": owl_content,
            "source_file": file_path.name,
            "source_type": suffix[1:],
            "text_length": len(text),
            "validation": validation,
            "generated_at": datetime.now().isoformat()
        }
    
    def process_url(self, url: str, domain: str = "general") -> Dict[str, Any]:
        """
        Process web URL and generate ontology
        
        Args:
            url: Web page URL
            domain: Domain context
            
        Returns:
            Dict with OWL content and metadata
        """
        print(f"🌐 Extracting text from URL: {url}")
        text = self.extract_text_from_url(url)
        
        if not text or len(text) < 50:
            raise ValueError("Extracted text is too short or empty")
        
        # Generate ontology
        owl_content = self.generate_ontology(text, domain)
        
        # Validate
        validation = self.validate_owl(owl_content)
        
        return {
            "owl_content": owl_content,
            "source_url": url,
            "source_type": "url",
            "text_length": len(text),
            "validation": validation,
            "generated_at": datetime.now().isoformat()
        }
    
    def process_url_chunked(self, url: str, domain: str = "general", 
                           chunk_size: int = 30000, overlap: int = 1000) -> Dict[str, Any]:
        """
        Process large web URL by chunking and generating multiple ontologies.
        
        Args:
            url: Web page URL
            domain: Domain context
            chunk_size: Max chars per chunk (default: 30000)
            overlap: Overlap between chunks (default: 1000)
            
        Returns:
            Dict with list of OWL contents, one per chunk
        """
        import hashlib
        
        print(f"🌐 Extracting text from URL: {url}")
        text = self.extract_text_from_url(url)
        
        if not text or len(text) < 50:
            raise ValueError("Extracted text is too short or empty")
        
        print(f"   Total text length: {len(text)} characters")
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size, overlap)
        print(f"   Split into {len(chunks)} chunks")
        
        # Generate group_id from URL hash
        group_id = hashlib.md5(url.encode()).hexdigest()[:12]
        
        results = []
        for i, chunk in enumerate(chunks):
            print(f"\n📝 Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            try:
                # Add chunk context to domain
                chunk_domain = f"{domain} (Part {i+1}/{len(chunks)})"
                
                # Generate ontology for this chunk
                owl_content = self.generate_ontology(chunk, chunk_domain)
                validation = self.validate_owl(owl_content)
                
                results.append({
                    "owl_content": owl_content,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "validation": validation,
                    "success": True
                })
                
                print(f"   ✓ Chunk {i+1}: {validation['class_count']} classes")
                
            except Exception as e:
                print(f"   ❌ Chunk {i+1} failed: {e}")
                results.append({
                    "owl_content": None,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "error": str(e),
                    "success": False
                })
        
        # Summary
        successful = [r for r in results if r['success']]
        total_classes = sum(r['validation']['class_count'] for r in successful)
        
        print(f"\n✅ Completed: {len(successful)}/{len(chunks)} chunks successful")
        print(f"   Total classes: {total_classes}")
        
        return {
            "source_url": url,
            "source_type": "url_chunked",
            "group_id": group_id,
            "total_text_length": len(text),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": results,
            "successful_chunks": len(successful),
            "total_classes": total_classes,
            "generated_at": datetime.now().isoformat()
        }



def main():
    """CLI for testing"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Generate OWL ontology from documents/URLs")
    parser.add_argument("--file", help="Path to PDF/DOCX file")
    parser.add_argument("--url", help="Web page URL")
    parser.add_argument("--domain", default="general", help="Domain context")
    parser.add_argument("--output", default="generated_ontology.owl", help="Output OWL file")
    parser.add_argument("--model", default="gpt-4", help="Model name")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama instead of OpenAI")
    
    args = parser.parse_args()
    
    if not args.file and not args.url:
        parser.error("Provide either --file or --url")
    
    # Initialize generator
    if args.ollama:
        generator = OntologyGenerator(use_ollama=True, model=args.model)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            parser.error("OPENAI_API_KEY environment variable required (or use --ollama)")
        generator = OntologyGenerator(api_key=api_key, model=args.model)
    
    try:
        if args.file:
            result = generator.process_file(args.file, args.domain)
        else:
            result = generator.process_url(args.url, args.domain)
        
        # Save OWL
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result['owl_content'])
        
        print(f"\n✅ Ontology generated successfully!")
        print(f"   Output: {args.output}")
        print(f"   Classes: {result['validation']['class_count']}")
        print(f"   Properties: {result['validation']['property_count']}")
        print(f"   Valid: {result['validation']['valid']}")
        
        if result['validation']['issues']:
            print(f"\n⚠️  Validation issues:")
            for issue in result['validation']['issues']:
                print(f"   - {issue}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
