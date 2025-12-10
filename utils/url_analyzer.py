#!/usr/bin/env python3
"""
URL Analyzer - Phân tích và phân loại URL trước khi scraping

Tự động detect loại trang web và đề xuất scraping strategy phù hợp
"""

import requests
from typing import Dict, Optional, List
from enum import Enum
from urllib.parse import urlparse
from dataclasses import dataclass


class SiteType(Enum):
    """Các loại trang web"""
    STATIC_HTML = "static_html"
    JAVASCRIPT_SPA = "javascript_spa"
    DYNAMIC_CONTENT = "dynamic_content"
    API_ENDPOINT = "api_endpoint"
    PDF_DOCUMENT = "pdf_document"
    UNKNOWN = "unknown"


class ProtectionType(Enum):
    """Các loại bảo vệ anti-bot"""
    NONE = "none"
    CLOUDFLARE = "cloudflare"
    RECAPTCHA = "recaptcha"
    RATE_LIMIT = "rate_limit"
    AUTH_REQUIRED = "auth_required"
    UNKNOWN = "unknown"


@dataclass
class URLAnalysis:
    """Kết quả phân tích URL"""
    url: str
    site_type: SiteType
    protection_type: ProtectionType
    requires_javascript: bool
    estimated_timeout: int
    content_type: Optional[str]
    is_robots_allowed: bool
    recommended_strategy: str
    metadata: Dict


class URLAnalyzer:
    """Phân tích URL và đề xuất scraping strategy"""
    
    # Danh sách các framework JavaScript SPA phổ biến
    SPA_INDICATORS = [
        'react', 'vue', 'angular', 'next.js', 'nuxt',
        '__NEXT_DATA__', '__NUXT__', 'ng-version'
    ]
    
    # Danh sách các dấu hiệu của anti-bot protection
    PROTECTION_INDICATORS = {
        ProtectionType.CLOUDFLARE: ['cloudflare', 'cf-ray', 'cf_clearance'],
        ProtectionType.RECAPTCHA: ['recaptcha', 'g-recaptcha'],
    }
    
    def __init__(self, timeout: int = 10):
        """
        Initialize analyzer
        
        Args:
            timeout: Timeout cho initial HEAD request
        """
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
    
    def analyze(self, url: str) -> URLAnalysis:
        """
        Phân tích URL đầy đủ
        
        Args:
            url: URL cần phân tích
            
        Returns:
            URLAnalysis object với đầy đủ thông tin
        """
        metadata = {}
        
        # 1. Parse URL
        parsed = urlparse(url)
        metadata['domain'] = parsed.netloc
        metadata['path'] = parsed.path
        
        # 2. Check if it's a direct PDF link
        if url.lower().endswith('.pdf'):
            return URLAnalysis(
                url=url,
                site_type=SiteType.PDF_DOCUMENT,
                protection_type=ProtectionType.NONE,
                requires_javascript=False,
                estimated_timeout=15,
                content_type='application/pdf',
                is_robots_allowed=True,
                recommended_strategy='pdf',
                metadata=metadata
            )
        
        # 3. Check if it's an API endpoint
        if self._is_api_endpoint(url):
            return URLAnalysis(
                url=url,
                site_type=SiteType.API_ENDPOINT,
                protection_type=ProtectionType.NONE,
                requires_javascript=False,
                estimated_timeout=5,
                content_type='application/json',
                is_robots_allowed=True,
                recommended_strategy='api',
                metadata=metadata
            )
        
        # 4. Perform HEAD request to get headers
        try:
            response = requests.head(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True
            )
            content_type = response.headers.get('Content-Type', '').lower()
            metadata['status_code'] = response.status_code
            metadata['content_type'] = content_type
            
            # Check for PDF in headers
            if 'application/pdf' in content_type:
                return URLAnalysis(
                    url=url,
                    site_type=SiteType.PDF_DOCUMENT,
                    protection_type=ProtectionType.NONE,
                    requires_javascript=False,
                    estimated_timeout=15,
                    content_type=content_type,
                    is_robots_allowed=True,
                    recommended_strategy='pdf',
                    metadata=metadata
                )
            
            # Detect protection from headers
            protection = self._detect_protection_from_headers(response.headers)
            metadata['protection'] = protection.value
            
        except Exception as e:
            # HEAD request failed, fallback to GET
            metadata['head_request_error'] = str(e)
            content_type = None
            protection = ProtectionType.NONE
        
        # 5. Quick GET request to analyze content
        site_type = SiteType.UNKNOWN
        requires_js = False
        
        try:
            # Quick GET with small timeout to sample content
            response = requests.get(
                url,
                headers=self.headers,
                timeout=5,
                stream=True  # Stream to avoid downloading everything
            )
            
            # Only read first 50KB to analyze
            content_sample = b''
            for chunk in response.iter_content(chunk_size=8192):
                content_sample += chunk
                if len(content_sample) >= 50000:
                    break
            
            content_str = content_sample.decode('utf-8', errors='ignore').lower()
            
            # Detect site type from content
            site_type = self._detect_site_type(content_str)
            requires_js = site_type in [SiteType.JAVASCRIPT_SPA, SiteType.DYNAMIC_CONTENT]
            
            # Update protection detection from content
            protection_from_content = self._detect_protection_from_content(content_str)
            if protection_from_content != ProtectionType.NONE:
                protection = protection_from_content
            
            metadata['content_sample_size'] = len(content_sample)
            
        except Exception as e:
            metadata['get_request_error'] = str(e)
            # Conservative fallback: assume needs JS
            site_type = SiteType.DYNAMIC_CONTENT
            requires_js = True
        
        # 6. Check robots.txt (simplified - can be enhanced)
        is_robots_allowed = self._check_robots_txt(url)
        
        # 7. Determine recommended strategy and timeout
        recommended_strategy, estimated_timeout = self._determine_strategy(
            site_type, protection, requires_js
        )
        
        return URLAnalysis(
            url=url,
            site_type=site_type,
            protection_type=protection,
            requires_javascript=requires_js,
            estimated_timeout=estimated_timeout,
            content_type=content_type,
            is_robots_allowed=is_robots_allowed,
            recommended_strategy=recommended_strategy,
            metadata=metadata
        )
    
    def _is_api_endpoint(self, url: str) -> bool:
        """Check if URL looks like an API endpoint"""
        api_indicators = ['/api/', '/v1/', '/v2/', '/graphql', '/rest/']
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in api_indicators)
    
    def _detect_site_type(self, content: str) -> SiteType:
        """Detect site type from HTML content"""
        # Check for SPA indicators
        spa_score = sum(1 for indicator in self.SPA_INDICATORS if indicator in content)
        if spa_score >= 2:
            return SiteType.JAVASCRIPT_SPA
        
        # Check for heavy JavaScript usage
        script_count = content.count('<script')
        if script_count > 10:
            return SiteType.DYNAMIC_CONTENT
        
        # Check if mostly static content
        if script_count <= 3 and '<html' in content:
            return SiteType.STATIC_HTML
        
        # Default to dynamic
        return SiteType.DYNAMIC_CONTENT
    
    def _detect_protection_from_headers(self, headers: Dict) -> ProtectionType:
        """Detect anti-bot protection from response headers"""
        headers_str = str(headers).lower()
        
        for protection_type, indicators in self.PROTECTION_INDICATORS.items():
            if any(indicator in headers_str for indicator in indicators):
                return protection_type
        
        return ProtectionType.NONE
    
    def _detect_protection_from_content(self, content: str) -> ProtectionType:
        """Detect anti-bot protection from page content"""
        for protection_type, indicators in self.PROTECTION_INDICATORS.items():
            if any(indicator in content for indicator in indicators):
                return protection_type
        
        return ProtectionType.NONE
    
    def _check_robots_txt(self, url: str) -> bool:
        """
        Simplified robots.txt check
        
        Returns:
            True if scraping is allowed (or can't determine)
        """
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            
            response = requests.get(robots_url, timeout=3)
            if response.status_code == 200:
                # Very basic check - can be enhanced with robotparser
                robots_content = response.text.lower()
                if 'disallow: /' in robots_content and 'user-agent: *' in robots_content:
                    return False
            
            # If can't find robots.txt or can't parse, assume allowed
            return True
            
        except Exception:
            # Default to allowed if check fails
            return True
    
    def _determine_strategy(
        self, 
        site_type: SiteType, 
        protection: ProtectionType,
        requires_js: bool
    ) -> tuple[str, int]:
        """
        Determine recommended strategy and timeout
        
        Returns:
            (strategy_name, estimated_timeout)
        """
        # Handle PDF and API specially
        if site_type == SiteType.PDF_DOCUMENT:
            return ('pdf', 15)
        if site_type == SiteType.API_ENDPOINT:
            return ('api', 5)
        
        # Handle protected sites
        if protection in [ProtectionType.CLOUDFLARE, ProtectionType.RECAPTCHA]:
            return ('stealth', 45)
        
        # Handle based on site type
        if site_type == SiteType.STATIC_HTML and not requires_js:
            return ('static', 10)
        
        if site_type == SiteType.JAVASCRIPT_SPA:
            return ('javascript', 30)
        
        if site_type == SiteType.DYNAMIC_CONTENT:
            return ('hybrid', 20)
        
        # Conservative fallback
        return ('hybrid', 25)


def main():
    """CLI for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python url_analyzer.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    analyzer = URLAnalyzer()
    
    print(f"\n🔍 Analyzing: {url}\n")
    result = analyzer.analyze(url)
    
    print(f"Site Type: {result.site_type.value}")
    print(f"Protection: {result.protection_type.value}")
    print(f"Requires JS: {result.requires_javascript}")
    print(f"Recommended Strategy: {result.recommended_strategy}")
    print(f"Estimated Timeout: {result.estimated_timeout}s")
    print(f"Robots.txt Allowed: {result.is_robots_allowed}")
    print(f"\nMetadata: {result.metadata}")


if __name__ == "__main__":
    main()
