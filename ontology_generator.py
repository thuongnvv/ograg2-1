#!/usr/bin/env python3
"""
Ontology Generator - Auto-generate OWL from documents/URLs using LLM

Supports:
- PDF files
- DOCX files
- Web URLs

Uses local LLM (Ollama) to analyze content and generate OWL ontology
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
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
                # Add timeout for external APIs
                kwargs["timeout"] = 60.0
            self.llm_client = OpenAI(**kwargs)
        
        self.model = model
        self.use_ollama = use_ollama
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text from web page"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
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
        # Truncate text if too long (keep first 8000 chars for context)
        if len(text) > 8000:
            text = text[:8000] + "\n...[truncated]"
        
        prompt = f"""You are an expert ontology engineer. Analyze the following text and create a well-structured OWL ontology.

Domain: {domain}

Text to analyze:
{text}

Instructions:
1. Identify key concepts (classes) from the text
2. Identify important properties and relationships
3. Create hierarchical structure (subclass relationships)
4. Add annotations (labels, definitions, comments)
5. Generate valid OWL/XML format

Requirements:
- Use meaningful IDs (e.g., CLASS_001, PROP_001)
- Include rdfs:label for human-readable names
- Include rdfs:comment or IAO:0000115 for definitions
- Create proper class hierarchy with rdfs:subClassOf
- Include object and data properties
- Follow OBO Foundry best practices if applicable
- Generate VALID XML that can be parsed

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
                max_tokens=4000
            )
            
            owl_content = response.choices[0].message.content.strip()
            
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
