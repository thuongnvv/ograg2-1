#!/usr/bin/env python3
"""
Extract ALL Q&A pairs from Creative Commons FAQ page
to create comprehensive ground truth dataset for RAG evaluation
"""

import re
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path


def extract_faq_qa_pairs(url: str = "https://creativecommons.org/faq/"):
    """
    Scrape and extract all Q&A pairs from Creative Commons FAQ
    
    Returns:
        List of {question, answer, category, id} dicts
    """
    print(f"Fetching FAQ page: {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main content
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    
    if not main_content:
        print("Warning: Could not find main content, using entire body")
        main_content = soup.find('body')
    
    qa_pairs = []
    current_category = "General"
    question_id = 1
    
    # Find all h2 (categories) and h3/h4 (questions)
    for element in main_content.find_all(['h2', 'h3', 'h4']):
        tag_name = element.name
        text = element.get_text(strip=True)
        
        if tag_name == 'h2':
            # This is a category
            current_category = text
            print(f"\nCategory: {current_category}")
        
        elif tag_name in ['h3', 'h4']:
            # This is a question
            question = text
            
            # Skip non-question headings
            if not question or len(question) < 10:
                continue
            
            # Extract answer (all text between this heading and next heading)
            answer_parts = []
            next_element = element.find_next_sibling()
            
            while next_element and next_element.name not in ['h2', 'h3', 'h4']:
                # Get text from paragraphs, lists, etc.
                if next_element.name in ['p', 'ul', 'ol', 'blockquote']:
                    text_content = next_element.get_text(strip=True)
                    if text_content:
                        answer_parts.append(text_content)
                
                next_element = next_element.find_next_sibling()
            
            answer = '\n\n'.join(answer_parts)
            
            # Only add if we have a substantial answer
            if answer and len(answer) > 50:
                qa_pairs.append({
                    'id': f"q{question_id:03d}",
                    'category': current_category,
                    'question': question,
                    'reference_answer': answer,
                    'difficulty': classify_difficulty(question, answer),
                    'key_concepts': extract_key_concepts(answer)
                })
                print(f"  ✓ Q{question_id:03d}: {question[:60]}...")
                question_id += 1
    
    return qa_pairs


def classify_difficulty(question: str, answer: str) -> str:
    """Classify question difficulty based on heuristics"""
    answer_length = len(answer)
    
    # Check for complexity indicators
    complex_terms = ['however', 'although', 'exception', 'depends', 'may require', 
                     'jurisdiction', 'legal advice', 'varies', 'circumstances']
    
    complexity_score = sum(1 for term in complex_terms if term.lower() in answer.lower())
    
    if answer_length > 800 or complexity_score >= 3:
        return "hard"
    elif answer_length > 400 or complexity_score >= 1:
        return "medium"
    else:
        return "easy"


def extract_key_concepts(answer: str) -> list:
    """Extract key concepts from answer text"""
    # Find capitalized terms and phrases in quotes
    concepts = []
    
    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', answer)
    concepts.extend(quoted[:5])  # Limit to top 5
    
    # Extract capitalized phrases (2-3 words)
    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', answer)
    concepts.extend(cap_phrases[:3])
    
    # Common important terms
    important_terms = ['license', 'copyright', 'attribution', 'commercial', 
                       'derivative', 'public domain', 'permission', 'rights']
    
    for term in important_terms:
        if term.lower() in answer.lower() and term not in concepts:
            concepts.append(term)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for c in concepts:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique_concepts.append(c)
    
    return unique_concepts[:8]  # Max 8 concepts


def main():
    print("=" * 70)
    print("Creative Commons FAQ Ground Truth Extraction")
    print("=" * 70)
    
    # Extract all Q&A pairs
    qa_pairs = extract_faq_qa_pairs()
    
    print(f"\n{'=' * 70}")
    print(f"Total questions extracted: {len(qa_pairs)}")
    print(f"{'=' * 70}")
    
    # Count by category
    categories = {}
    for qa in qa_pairs:
        cat = qa['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nQuestions by category:")
    for cat, count in categories.items():
        print(f"  • {cat}: {count}")
    
    # Count by difficulty
    difficulties = {}
    for qa in qa_pairs:
        diff = qa['difficulty']
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print("\nQuestions by difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        count = difficulties.get(diff, 0)
        print(f"  • {diff}: {count}")
    
    # Create dataset
    dataset = {
        "dataset_info": {
            "source": "https://creativecommons.org/faq/",
            "created_date": "2026-01-09",
            "description": "Complete ground truth Q&A pairs from Creative Commons FAQ for RAG evaluation",
            "total_questions": len(qa_pairs),
            "categories": list(categories.keys()),
            "extraction_method": "BeautifulSoup HTML parsing"
        },
        "questions": qa_pairs
    }
    
    # Save to file
    output_file = Path(__file__).parent.parent / "evaluation" / "faq_ground_truth_complete.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved complete dataset to: {output_file}")
    print(f"   Total size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Print sample questions
    print("\n" + "=" * 70)
    print("Sample Questions:")
    print("=" * 70)
    for i, qa in enumerate(qa_pairs[:5], 1):
        print(f"\n{i}. [{qa['category']}] {qa['question']}")
        print(f"   Answer length: {len(qa['reference_answer'])} chars")
        print(f"   Difficulty: {qa['difficulty']}")
        print(f"   Key concepts: {', '.join(qa['key_concepts'][:3])}")


if __name__ == "__main__":
    main()
