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
                 base_url: str = None):
        """
        Initialize generator
        
        Args:
            api_key: OpenAI API key (required if use_ollama=False)
            model: Model name (gpt-4, gpt-3.5-turbo, or Ollama model)
            use_ollama: Use Ollama instead of OpenAI API
            ollama_base_url: Ollama server URL (only if use_ollama=True)
            base_url: Custom API base URL (for OpenAI-compatible APIs)
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
        Generate OWL ontology from text using LLM
        
        UNIVERSAL APPROACH (v2):
        - Single prompt works for ALL content types (FAQ, Domain Knowledge, Mixed)
        - LLM automatically chooses appropriate OWL constructs
        - Strict format requirements to avoid errors
        
        Args:
            text: Input text to analyze
            domain: Domain of the ontology (for context)
            
        Returns:
            OWL/XML string
        """
        # UNIVERSAL PROMPT - Works for ANY content type
        prompt = f"""You are an expert Ontology Engineer. Your task is to convert the provided text into a valid OWL ontology in RDF/XML format.

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
'''xml
<owl:Class rdf:about="#FAQ_WhatIsStripe">
  <rdfs:label>What is Stripe?</rdfs:label>
  <rdfs:comment>Stripe is a payment processing platform that allows businesses to accept payments online. When you see a charge from Stripe on your statement, it means a business using Stripe processed your payment.</rdfs:comment>
</owl:Class>

<owl:Class rdf:about="#FAQ_UnrecognizedCharge">
  <rdfs:label>What should I do if I don't recognize a charge from Stripe?</rdfs:label>
  <rdfs:comment>Use the charge lookup tool at stripe.com/chargeid to identify which business processed the charge. Enter the charge ID from your bank statement to see the business name and description.</rdfs:comment>
</owl:Class>
'''

WRONG FAQ Examples (DO NOT DO THIS):
❌ <owl:ObjectProperty rdf:about="#usesChargeLookupTool"> <!-- WRONG! -->
❌ <owl:NamedIndividual rdf:about="#Question1"> <!-- WRONG! -->
❌ Creating relationships between questions <!-- WRONG! -->

═══════════════════════════════════════════════════════════════════
PRIORITY 2: DOMAIN KNOWLEDGE / CONCEPTS
═══════════════════════════════════════════════════════════════════
IF NOT FAQ, check for:
- Concept definitions and classifications
- "is a" / "type of" hierarchical relationships
- Formal taxonomy structure

THEN create:
- owl:Class for each concept
- rdfs:subClassOf for hierarchies
- rdfs:label and rdfs:comment for definitions

EXAMPLE:
'''xml
<owl:Class rdf:about="#Mammal">
  <rdfs:label>Mammal</rdfs:label>
  <rdfs:comment>Warm-blooded vertebrate animal.</rdfs:comment>
  <rdfs:subClassOf rdf:resource="#Animal"/>
</owl:Class>
'''

═══════════════════════════════════════════════════════════════════
PRIORITY 3: RELATIONSHIPS / PROCESSES
═══════════════════════════════════════════════════════════════════
IF content describes actions, connections, or processes between entities:

'''xml
<owl:ObjectProperty rdf:about="#processes">
  <rdfs:label>processes</rdfs:label>
  <rdfs:comment>Indicates that one entity processes another.</rdfs:comment>
  <rdfs:domain rdf:resource="#PaymentProcessor"/>
  <rdfs:range rdf:resource="#Payment"/>
</owl:ObjectProperty>
'''

═══════════════════════════════════════════════════════════════════
PRIORITY 4: CONCRETE EXAMPLES / INSTANCES
═══════════════════════════════════════════════════════════════════
IF content mentions specific real-world examples:

'''xml
<owl:NamedIndividual rdf:about="#Pacific_Ocean">
  <rdf:type rdf:resource="#Ocean"/>
  <rdfs:label>Pacific Ocean</rdfs:label>
  <rdfs:comment>Largest ocean on Earth.</rdfs:comment>
</owl:NamedIndividual>
'''


═══════════════════════════════════════════════════════════════════
CRITICAL XML/OWL REQUIREMENTS (MUST FOLLOW EXACTLY):
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

3. URI FORMAT:
   - Use descriptive names: #WhatIsPython NOT #Class001
   - Use CamelCase or Underscores: #What_Is_Python or #WhatIsPython
   - No spaces, no special chars except underscore
   - Must start with # for local URIs

4. REQUIRED ANNOTATIONS:
   - Every owl:Class MUST have rdfs:label
   - Every owl:Class SHOULD have rdfs:comment (if information available)
   - Use rdfs:comment for definitions, descriptions, answers

5. XML SYNTAX:
   - Self-closing tags: <owl:Ontology rdf:about=""/>
   - Proper nesting: close inner tags before outer tags
   - Escape special characters: &lt; &gt; &amp; &quot; &apos;
   - No unclosed tags

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT TEMPLATE:
═══════════════════════════════════════════════════════════════════

<?xml version="1.0"?>
<rdf:RDF
    xmlns="http://example.org/ontology#"
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
    
  <owl:Ontology rdf:about=""/>
  
  <!-- Your OWL entities here -->
  
</rdf:RDF>

═══════════════════════════════════════════════════════════════════
QUALITY CHECKLIST (verify before outputting):
═══════════════════════════════════════════════════════════════════

✓ XML declaration present
✓ All 5 namespaces declared (xmlns, rdf, rdfs, owl, xsd)
✓ <owl:Ontology/> element present
✓ All tags properly closed
✓ All URIs start with #
✓ All Classes have rdfs:label
✓ No placeholder content (replace "..." with actual content)
✓ Special characters properly escaped

═══════════════════════════════════════════════════════════════════
FINAL INSTRUCTION:
═══════════════════════════════════════════════════════════════════

Output ONLY the raw XML. Do NOT use markdown code blocks. Do NOT add explanations before or after.
Begin your response with: <?xml version="1.0"?>
"""


        print("🤖 Generating ontology with LLM...")
        print(f"   Model: {self.model}")
        print(f"   Text length: {len(text)} chars")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert ontology engineer who generates valid OWL/XML ontologies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more structured output
                max_tokens=50000  # Maximum allowed by API (56000 limit, use 50000 for safety)
            )
            
            owl_content = response.choices[0].message.content.strip()
            
            # Log token usage
            if hasattr(response, 'usage'):
                print(f"   Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                if response.choices[0].finish_reason:
                    print(f"   Finish reason: {response.choices[0].finish_reason}")
            
            # Extract XML if wrapped in markdown code blocks
            if "```xml" in owl_content:
                owl_content = re.search(r'```xml\s*(.*?)\s*```', owl_content, re.DOTALL)
                if owl_content:
                    owl_content = owl_content.group(1)
            elif "```" in owl_content:
                owl_content = re.search(r'```\s*(.*?)\s*```', owl_content, re.DOTALL)
                if owl_content:
                    owl_content = owl_content.group(1)
            
            # Ensure it starts with XML declaration
            if not owl_content.startswith("<?xml"):
                owl_content = '<?xml version="1.0"?>\n' + owl_content
            
            # Auto-fix: Add <owl:Ontology> if missing
            if "<owl:Ontology" not in owl_content and "<Ontology" not in owl_content:
                # Insert after <rdf:RDF ...> opening tag
                import re
                match = re.search(r'(<rdf:RDF[^>]*>)', owl_content, re.DOTALL)
                if match:
                    insertion_point = match.end()
                    owl_ontology_tag = '\n  <owl:Ontology rdf:about=""/>\n'
                    owl_content = owl_content[:insertion_point] + owl_ontology_tag + owl_content[insertion_point:]
                    print("   ℹ️  Auto-added missing <owl:Ontology> element")
            
            return owl_content
            
        except Exception as e:
            raise ValueError(f"LLM generation failed: {e}")
    
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
