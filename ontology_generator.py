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
        return self._call_llm(generation_prompt, is_xml=True)

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
        """Get FLAT FAQ prompt (baseline - no hierarchy/relationships)"""
        return f"""You are an expert Ontology Engineer. Your task is to convert the provided text into a valid OWL ontology in RDF/XML format.

═══════════════════════════════════════════════════════════════════
DOMAIN CONTEXT: {domain}
═══════════════════════════════════════════════════════════════════

INPUT TEXT:
{text}

═══════════════════════════════════════════════════════════════════
YOUR TASK: Analyze the text and extract ALL knowledge into OWL format
═══════════════════════════════════════════════════════════════════

STEP 1: DETECT CONTENT TYPE (Follow this order - FIRST MATCH WINS)

═══════════════════════════════════════════════════════════════════
PRIORITY 1: FAQ / Q&A CONTENT DETECTION
═══════════════════════════════════════════════════════════════════
DETECTION RULES (if ANY of these are true, treat as FAQ):
✓ Text contains 3+ questions ending with "?"
✓ Questions followed by explanatory text (answers)
✓ Keywords present: "FAQ", "frequently asked", "Q:", "A:"
✓ Pattern: Question → Answer → Question → Answer

IF FAQ DETECTED:
CRITICAL: Create ONLY owl:Class elements (NO Properties, NO Individuals)
- ONE Class per Q&A pair
- rdfs:label = COMPLETE question text (keep the "?")
- rdfs:comment = COMPLETE answer text (all details, multiple sentences OK)
- Use IDs like: #FAQ_WhatIsStripe, #FAQ_HowToRefund
- DO NOT create owl:ObjectProperty
- DO NOT create relationships between questions

EXAMPLE (CORRECT FAQ Structure):
```xml
<owl:Class rdf:about="#FAQ_WhatIsStripe">
  <rdfs:label>What is Stripe?</rdfs:label>
  <rdfs:comment>Stripe is a payment processing platform that allows businesses to accept payments online.</rdfs:comment>
</owl:Class>

<owl:Class rdf:about="#FAQ_UnrecognizedCharge">
  <rdfs:label>What should I do if I don't recognize a charge from Stripe?</rdfs:label>
  <rdfs:comment>Use the charge lookup tool at stripe.com/chargeid to identify which business processed the charge.</rdfs:comment>
</owl:Class>
```

WRONG FAQ Examples (DO NOT DO THIS):
❌ <owl:ObjectProperty rdf:about="#usesChargeLookupTool"> <!-- WRONG! -->
❌ <owl:NamedIndividual rdf:about="#Question1"> <!-- WRONG! -->
❌ Creating relationships between questions <!-- WRONG! -->
❌ Using rdfs:subClassOf between FAQ classes <!-- WRONG! -->

═══════════════════════════════════════════════════════════════════
CRITICAL XML/OWL REQUIREMENTS:
═══════════════════════════════════════════════════════════════════

1. STRUCTURE:
   - Start with: <?xml version="1.0"?>
   - Root element: <rdf:RDF> with ALL required namespaces
   - Include <owl:Ontology rdf:about=""/> element
   - Close all tags properly

2. REQUIRED NAMESPACES (copy exactly):
   xmlns="http://example.org/ontology#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
   xmlns:owl="http://www.w3.org/2002/07/owl#"
   xmlns:xsd="http://www.w3.org/2001/XMLSchema#"

═══════════════════════════════════════════════════════════════════
OUTPUT TEMPLATE:
═══════════════════════════════════════════════════════════════════

<?xml version="1.0"?>
<rdf:RDF
    xmlns="http://example.org/ontology#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
    
  <owl:Ontology rdf:about=""/>
  
  <!-- Flat FAQ Classes - NO hierarchy, NO relationships -->
  
</rdf:RDF>

═══════════════════════════════════════════════════════════════════
FINAL INSTRUCTION:
═══════════════════════════════════════════════════════════════════

Output ONLY the raw XML. Do NOT use markdown code blocks. Do NOT add explanations.
Begin your response with: <?xml version="1.0"?>
"""
    
    def _get_schema_discovery_prompt(self, text: str, domain: str) -> str:
        """Step 1: Analyze text and design Ontology Schema (Goal-Oriented)"""
        return f"""Analyze the provided text and design an Ontology Schema that captures ALL knowledge needed to answer user questions.

═══════════════════════════════════════════════════════════════════
DOMAIN CONTEXT: {domain}
═══════════════════════════════════════════════════════════════════

INPUT TEXT:
{text}

═══════════════════════════════════════════════════════════════════
YOUR TASK: Design a Schema for Actionable Knowledge
═══════════════════════════════════════════════════════════════════

Imagine a user asking "How do I...?", "When...?", "Why...?", or "What is...?".
Design a schema that can capture the answers to these questions.

Your schema MUST cover:
1. **CONCEPTS (Nouns)**: The things/entities involved.
2. **ACTIONS/TASKS (Verbs)**: What can be done? (Crucial for DSL mapping later).
   - e.g., "Cancel Subscription", "Refund Charge".
3. **PROCEDURES (Workflows)**: Steps to complete a task.
4. **RULES/CONDITIONS**: When is an action allowed? (e.g., "within 60 days").
5. **QUANTITATIVE DATA**: Any numbers, prices, durations, deadlines.
   - Define specific attributes for these (e.g., hasDuration, hasCost).
6. **KEY FACTS / ASSERTIONS**: Important statements/rules that don't fit into simple structures.
   - e.g., "Issuer decides refund timing", "Stripe cannot refund directly".
   - Model these as instances of a 'Fact' or 'Assertion' class.

Output a JSON object with the schema design.

EXAMPLE OUTPUT FORMAT:
{{
  "classes": [
    {{ "name": "PaymentPlatform", "description": "System processing payments" }},
    {{ "name": "RefundAction", "description": "Task of returning funds" }}
  ],
  "relationships": [
    {{ "name": "performs", "domain": "User", "range": "Action" }},
    {{ "name": "requiresCondition", "domain": "Action", "range": "Condition" }}
  ],
  "attributes": [
    {{ "name": "duration", "type": "string" }},
    {{ "name": "amount", "type": "decimal" }}
  ],
  "special_structures": "Model 'Refund Process' as a sequence of Actions. Capture 'TimeLimit' as a condition."
}}

Output ONLY the JSON.
"""

    def _get_ontology_generation_prompt(self, text: str, domain: str, schema: Any) -> str:
        """Step 2: Generate OWL based on Schema"""
        return f"""You are an expert Ontology Engineer. Convert the provided text into a valid OWL ontology based on the DESIGN SCHEMA.

═══════════════════════════════════════════════════════════════════
DESIGN SCHEMA (Follow this structure):
═══════════════════════════════════════════════════════════════════
{json.dumps(schema, indent=2) if isinstance(schema, dict) else schema}

═══════════════════════════════════════════════════════════════════
INPUT TEXT:
═══════════════════════════════════════════════════════════════════
{text}

═══════════════════════════════════════════════════════════════════
INSTRUCTIONS:
═══════════════════════════════════════════════════════════════════

1. Create owl:Class for each concept in the schema
2. Create owl:ObjectProperty for each relationship in the schema
3. Create owl:DatatypeProperty for each attribute in the schema
4. Implement any special structures defined in the schema
5. POPULATE the ontology with specific INSTANCES (owl:NamedIndividual) extracted from the text
   - Extract ALL specific values, tools, entities mentioned
   - Link them using the defined properties

CRITICAL REQUIREMENT:
Every <owl:Class> and <owl:NamedIndividual> MUST have an <rdfs:comment> 
containing a natural language description. This is essential for search.

Example:
<owl:NamedIndividual rdf:about="#Stripe">
  <rdfs:comment>A technology company that builds economic infrastructure for the internet.</rdfs:comment>
  ...
</owl:NamedIndividual>

═══════════════════════════════════════════════════════════════════
REQUIRED NAMESPACES:
═══════════════════════════════════════════════════════════════════
xmlns="http://example.org/ontology#"
xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
xmlns:owl="http://www.w3.org/2002/07/owl#"
xmlns:xsd="http://www.w3.org/2001/XMLSchema#"

═══════════════════════════════════════════════════════════════════
OUTPUT TEMPLATE:
═══════════════════════════════════════════════════════════════════
<?xml version="1.0"?>
<rdf:RDF ...>
  <owl:Ontology rdf:about=""/>
  
  <!-- Classes -->
  
  <!-- Properties -->
  
  <!-- Instances (The most important part!) -->
  
</rdf:RDF>

Output ONLY the raw XML. Begin with <?xml version="1.0"?>
"""
    
    def validate_owl(self, owl_content: str) -> Dict[str, Any]:
        """
        Basic validation of OWL content
        
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
        try:
            from xml.etree import ElementTree as ET
            ET.fromstring(owl_content)
        except Exception as e:
            issues.append(f"XML parsing error: {str(e)[:100]}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "class_count": owl_content.count("<owl:Class") + owl_content.count("<Class"),
            "property_count": owl_content.count("Property")
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
