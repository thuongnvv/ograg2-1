#!/usr/bin/env python3
"""
Content Validator - Validate extracted content quality

Checks if extracted content is meaningful and sufficient
"""

from typing import Optional
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """Result of content validation"""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    issues: list[str]
    cleaned_text: Optional[str] = None


class ContentValidator:
    """Validate and clean extracted content"""
    
    # Common error page indicators
    ERROR_INDICATORS = [
        '404 not found',
        '403 forbidden',
        '500 internal server error',
        'page not found',
        'access denied',
        'error occurred',
    ]
    
    # Common boilerplate patterns
    BOILERPLATE_PATTERNS = [
        r'cookie policy',
        r'privacy policy',
        r'terms of service',
        r'copyright \d{4}',
        r'all rights reserved',
        r'subscribe to.*newsletter',
        r'follow us on',
    ]
    
    def __init__(
        self,
        min_length: int = 200,
        min_quality_score: float = 0.3,
        max_boilerplate_ratio: float = 0.7
    ):
        """
        Initialize validator
        
        Args:
            min_length: Minimum text length (adaptive based on context)
            min_quality_score: Minimum quality score to pass
            max_boilerplate_ratio: Maximum ratio of boilerplate to content
        """
        self.min_length = min_length
        self.min_quality_score = min_quality_score
        self.max_boilerplate_ratio = max_boilerplate_ratio
    
    def validate(self, text: str, url: str = "") -> ValidationResult:
        """
        Validate extracted content
        
        Args:
            text: Extracted text to validate
            url: Source URL (for context)
            
        Returns:
            ValidationResult with is_valid, quality_score, and issues
        """
        issues = []
        
        # Basic length check
        if not text or len(text.strip()) < self.min_length:
            issues.append(f"Content too short: {len(text)} chars (min: {self.min_length})")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=issues
            )
        
        text_lower = text.lower()
        
        # Check for error pages
        if self.is_error_page(text_lower):
            issues.append("Appears to be an error page")
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                issues=issues
            )
        
        # Calculate quality metrics
        quality_score = self.calculate_content_quality(text)
        
        if quality_score < self.min_quality_score:
            issues.append(f"Quality score too low: {quality_score:.2f} (min: {self.min_quality_score})")
        
        # Check boilerplate ratio
        boilerplate_ratio = self._calculate_boilerplate_ratio(text_lower)
        if boilerplate_ratio > self.max_boilerplate_ratio:
            issues.append(f"Too much boilerplate: {boilerplate_ratio:.2%}")
            quality_score *= 0.7  # Reduce quality score
        
        # Clean the text
        cleaned_text = self.remove_boilerplate(text)
        
        is_valid = len(issues) == 0 and quality_score >= self.min_quality_score
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            cleaned_text=cleaned_text
        )
    
    def is_error_page(self, text: str) -> bool:
        """Check if content appears to be an error page"""
        text_lower = text.lower()
        
        # Check for error indicators
        error_count = sum(
            1 for indicator in self.ERROR_INDICATORS
            if indicator in text_lower
        )
        
        # If multiple error indicators and short content
        if error_count >= 2 and len(text) < 1000:
            return True
        
        return False
    
    def calculate_content_quality(self, text: str) -> float:
        """
        Calculate content quality score (0.0 to 1.0)
        
        Based on:
        - Word count and diversity
        - Sentence structure
        - Meaningful content indicators
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # 1. Word count and length (max 0.3)
        words = text.split()
        word_count = len(words)
        
        if word_count >= 100:
            score += 0.3
        elif word_count >= 50:
            score += 0.15
        
        # 2. Vocabulary diversity (max 0.3)
        if word_count > 0:
            unique_words = len(set(w.lower() for w in words))
            diversity = unique_words / word_count
            score += min(diversity * 0.5, 0.3)
        
        # 3. Has sentences (max 0.2)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if len(s.strip()) > 10])
        
        if sentence_count >= 5:
            score += 0.2
        elif sentence_count >= 2:
            score += 0.1
        
        # 4. Has paragraphs (max 0.2)
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if len(p.strip()) > 50])
        
        if paragraph_count >= 3:
            score += 0.2
        elif paragraph_count >= 1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_boilerplate_ratio(self, text: str) -> float:
        """Calculate ratio of boilerplate content"""
        if not text:
            return 1.0
        
        boilerplate_chars = 0
        
        for pattern in self.BOILERPLATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                boilerplate_chars += len(match.group())
        
        return boilerplate_chars / len(text)
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate content
        
        Note: This is a basic implementation. For better results,
        consider using libraries like boilerpy3 or jusText
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Split into lines
        lines = text.split('\n')
        
        # Filter out likely boilerplate lines
        filtered_lines = []
        for line in lines:
            line_lower = line.lower()
            
            # Skip if line matches boilerplate patterns
            is_boilerplate = any(
                re.search(pattern, line_lower)
                for pattern in self.BOILERPLATE_PATTERNS
            )
            
            if not is_boilerplate and len(line.strip()) > 0:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)


def main():
    """CLI for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python content_validator.py <text_file>")
        sys.exit(1)
    
    text_file = sys.argv[1]
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    validator = ContentValidator()
    result = validator.validate(text)
    
    print(f"\n📊 Content Validation Results\n")
    print(f"Valid: {result.is_valid}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"\nIssues: {len(result.issues)}")
    for issue in result.issues:
        print(f"  - {issue}")
    
    if result.cleaned_text:
        print(f"\nCleaned text length: {len(result.cleaned_text)} chars")


if __name__ == "__main__":
    main()
