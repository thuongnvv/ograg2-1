#!/usr/bin/env python3
"""
Scraping Strategies - Multiple approaches for different site types

Implements various scraping strategies to handle different types of web pages
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
from pathlib import Path


@dataclass
class ScrapingConfig:
    """Configuration for scraping"""
    timeout: int
    user_agent: str
    wait_time: int = 2
    max_retries: int = 3
    verify_ssl: bool = True
    headers: Optional[Dict] = None


class ScrapingStrategy(ABC):
    """Base class for scraping strategies"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
    
    @abstractmethod
    def extract(self, url: str) -> str:
        """
        Extract text content from URL
        
        Args:
            url: Target URL
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If extraction fails
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass


class StaticHTMLStrategy(ScrapingStrategy):
    """Fast BeautifulSoup-based extraction for static HTML sites"""
    
    def get_name(self) -> str:
        return "StaticHTML"
    
    def extract(self, url: str) -> str:
        """Extract from static HTML using BeautifulSoup"""
        headers = self.config.headers or {
            'User-Agent': self.config.user_agent
        }
        
        # Try with SSL verification first
        try:
            response = requests.get(
                url, 
                timeout=self.config.timeout,
                headers=headers,
                verify=self.config.verify_ssl
            )
            response.raise_for_status()
        except requests.exceptions.SSLError:
            # Retry without SSL verification
            import warnings
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')
            response = requests.get(
                url,
                timeout=self.config.timeout,
                headers=headers,
                verify=False
            )
            response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text


class JavaScriptStrategy(ScrapingStrategy):
    """Playwright-based extraction for JavaScript-heavy sites"""
    
    def get_name(self) -> str:
        return "JavaScript"
    
    def extract(self, url: str) -> str:
        """Extract from JavaScript-rendered content using Playwright"""
        import sys
        
        # Skip Playwright entirely on Windows (asyncio subprocess issue)
        if sys.platform == 'win32':
            print("   ⚠️  Playwright unavailable on Windows, using static HTML...")
            static_strategy = StaticHTMLStrategy(self.config)
            return static_strategy.extract(url)
        
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright is required for JavaScript strategy. "
                "Install with: pip install playwright && playwright install chromium"
            )
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set user agent
                page.set_extra_http_headers({
                    'User-Agent': self.config.user_agent
                })
                
                # Navigate with domcontentloaded wait
                # (networkidle can hang on analytics scripts)
                page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout * 1000)
                
                # Wait for dynamic content to render
                page.wait_for_timeout(self.config.wait_time * 1000)
                
                # Get text from body
                text = page.inner_text("body")
                browser.close()
                
                return text
                
        except NotImplementedError:
            # Windows asyncio subprocess issue - fallback to static HTML
            print("   ⚠️  Playwright unavailable (Windows asyncio issue), falling back to static HTML...")
            static_strategy = StaticHTMLStrategy(self.config)
            return static_strategy.extract(url)


class HybridStrategy(ScrapingStrategy):
    """Try static first, fallback to JavaScript if needed"""
    
    def get_name(self) -> str:
        return "Hybrid"
    
    def extract(self, url: str) -> str:
        """Try static extraction first, fallback to JS if content is insufficient"""
        # Try static first
        static_strategy = StaticHTMLStrategy(self.config)
        
        try:
            text = static_strategy.extract(url)
            
            # Check if content is sufficient
            if len(text) >= 500:  # Reasonable content length
                return text
            
            # Content too short, try JavaScript
            print(f"   ⚠️  Static extraction yielded only {len(text)} chars, trying JavaScript...")
            
        except Exception as e:
            print(f"   ⚠️  Static extraction failed: {e}, trying JavaScript...")
        
        # Fallback to JavaScript
        js_strategy = JavaScriptStrategy(self.config)
        return js_strategy.extract(url)


class PDFStrategy(ScrapingStrategy):
    """Extract text from PDF documents"""
    
    def get_name(self) -> str:
        return "PDF"
    
    def extract(self, url: str) -> str:
        """Download and extract text from PDF"""
        from pypdf import PdfReader
        import tempfile
        
        headers = self.config.headers or {
            'User-Agent': self.config.user_agent
        }
        
        # Download PDF
        response = requests.get(url, timeout=self.config.timeout, headers=headers)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        try:
            # Extract text
            reader = PdfReader(tmp_path)
            text = ""
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception:
                    continue
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            return text.strip()
            
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)


class StealthStrategy(ScrapingStrategy):
    """Advanced strategy with anti-detection techniques"""
    
    def get_name(self) -> str:
        return "Stealth"
    
    def extract(self, url: str) -> str:
        """
        Extract with stealth techniques to bypass anti-bot protection
        
        Note: This is a basic implementation. For heavy protection,
        consider using specialized tools like playwright-stealth or undetected-chromedriver
        """
        import sys
        
        # Skip Playwright entirely on Windows (asyncio subprocess issue)
        if sys.platform == 'win32':
            print("   ⚠️  Playwright unavailable on Windows, using static HTML...")
            static_strategy = StaticHTMLStrategy(self.config)
            return static_strategy.extract(url)
        
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("Playwright required for stealth strategy")
        
        try:
            with sync_playwright() as p:
                # Use more stealthy browser context
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )
                
                context = browser.new_context(
                    user_agent=self.config.user_agent,
                    viewport={'width': 1920, 'height': 1080},
                    locale='en-US',
                )
                
                page = context.new_page()
                
                # Add stealth scripts
                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """)
                
                # Navigate
                page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout * 1000)
                
                # Random wait to appear more human-like
                import random
                wait_time = random.uniform(2, 4)
                page.wait_for_timeout(int(wait_time * 1000))
                
                # Get content
                text = page.inner_text("body")
                browser.close()
                
                return text
                
        except NotImplementedError:
            # Windows asyncio subprocess issue - fallback to static HTML
            print("   ⚠️  Playwright unavailable (Windows asyncio issue), falling back to static HTML...")
            static_strategy = StaticHTMLStrategy(self.config)
            return static_strategy.extract(url)


# Strategy factory
def create_strategy(strategy_name: str, config: ScrapingConfig) -> ScrapingStrategy:
    """
    Factory function to create scraping strategy
    
    Args:
        strategy_name: Name of strategy (static, javascript, hybrid, pdf, stealth)
        config: Scraping configuration
        
    Returns:
        ScrapingStrategy instance
    """
    strategies = {
        'static': StaticHTMLStrategy,
        'javascript': JavaScriptStrategy,
        'hybrid': HybridStrategy,
        'pdf': PDFStrategy,
        'stealth': StealthStrategy,
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        # Default to hybrid
        strategy_class = HybridStrategy
    
    return strategy_class(config)
