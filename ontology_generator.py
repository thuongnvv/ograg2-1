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
                "timeout": 6000.0,  # 10 minutes for large ontology generation
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
        """Extract text from web page or PDF URL"""
        try:
            # Download with headers to avoid blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Try with SSL verification first
            try:
                response = requests.get(url, timeout=30, headers=headers)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                # Retry without SSL verification for sites with cert issues
                import warnings
                warnings.filterwarnings('ignore', message='Unverified HTTPS request')
                response = requests.get(url, timeout=30, headers=headers, verify=False)
                response.raise_for_status()
            
            # Check if it's a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')
            
            if is_pdf:
                # Save temporary PDF and extract text
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                
                try:
                    text = self.extract_text_from_pdf(tmp_path)
                except Exception as pdf_error:
                    # Clean up and re-raise with helpful message
                    Path(tmp_path).unlink(missing_ok=True)
                    raise ValueError(
                        f"PDF extraction failed: {pdf_error}. "
                        "The PDF may be corrupted, password-protected, or image-based. "
                        "Try downloading and uploading the file instead."
                    )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                
                return text
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from URL: {e}")
    
    def generate_ontology(self, text: str, domain: str = "general") -> str:
        """
        Generate OWL ontology from text using LLM
        
        Args:
            text: Input text to analyze
            domain: Domain of the ontology (for context)
            
        Returns:
            OWL/XML string
        """
        # Estimate expected output size based on text length
        estimated_classes = min(max(len(text) // 2000, 20), 100)  # 1 class per 2000 chars, min 20, max 100
       
        
        prompt = f"""You are an expert ontology engineer. Analyze the following {len(text)}-character text COMPREHENSIVELY and create a detailed, well-structured OWL ontology.

Domain: {domain}

Text to analyze:
{text}

Instructions:
1. THOROUGHLY identify ALL key concepts (classes) from the text
   - Target: Extract at least {estimated_classes} classes (concepts/entities)
   - Don't stop early - continue until you've captured all important concepts
2. Identify ALL important properties and relationships between concepts
   - Target: At least {estimated_classes // 2} properties
3. Create deep hierarchical structure with multiple levels of subclass relationships
4. Add detailed annotations (labels, definitions, comments) for every class and property
5. Generate valid OWL/XML format with complete coverage

Requirements:
- Extract MAXIMUM information from the text - this is a {len(text)}-char document requiring comprehensive extraction
- Use meaningful IDs (e.g., CLASS_001, CLASS_002, ..., PROP_001, PROP_002, ...)
- Include rdfs:label for human-readable names for EVERY entity
- Include rdfs:comment with detailed definitions for EVERY entity
- Create comprehensive class hierarchy with rdfs:subClassOf
- Include both object properties (relationships) and data properties (attributes)
- Generate VALID XML that can be parsed
- MUST declare xsd namespace: xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
- Use xsd:string, xsd:integer, xsd:boolean, xsd:decimal for datatype properties

CRITICAL: This is a LARGE document ({len(text)} chars). Create a COMPREHENSIVE ontology with at least {estimated_classes} classes.
DO NOT create a minimal 5-10 class example. Continue generating until the full document scope is captured.
A proper ontology for this document should be several thousand lines of XML.

Output ONLY the complete OWL/XML file, starting with <?xml version="1.0"?>
Do not include any explanations before or after the XML."""

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
