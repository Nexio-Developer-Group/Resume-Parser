import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SectionType(Enum):
    ABOUT = "about"
    EDUCATION = "education"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    PROJECTS = "projects"
    CERTIFICATIONS = "certifications"
    UNKNOWN = "unknown"

@dataclass
class Section:
    section_type: SectionType
    header: str
    content: str
    start_line: int
    end_line: int
    confidence: float

@dataclass
class HeaderCandidate:
    line_no: int
    line: str
    candidates: List[Dict]
    status: str

class EnhancedResumeParser:
    def __init__(self, threshold=0.5, max_words_in_header=4):
        self.threshold = threshold
        self.max_words_in_header = max_words_in_header
        
        # Enhanced section grammar with more variations
        self.section_grammar = {
            SectionType.ABOUT: [
                "about me", "about", "summary", "profile", "personal summary", 
                "professional summary", "objective", "career objective",
                "personal statement", "introduction", "bio", "biography",
                "overview", "career summary", "executive summary"
            ],
            SectionType.EDUCATION: [
                "education", "educational background", "academic background", 
                "qualifications", "academic qualifications", "degrees", 
                "academic history", "schooling", "university", "college",
                "studies", "coursework"
            ],
            SectionType.EXPERIENCE: [
                "experience", "work experience", "professional experience", 
                "employment history", "career history", "work history", 
                "job experience", "professional background", "employment",
                "career", "positions", "roles", "jobs", "professional roles"
            ],
            SectionType.SKILLS: [
                "skills", "technical skills", "core skills", "key skills", 
                "competencies", "expertise", "technologies", "programming languages",
                "tools", "software skills", "hard skills", "soft skills",
                "abilities", "proficiencies", "capabilities", "skills proficiencies"
            ],
            SectionType.PROJECTS: [
                "projects", "personal projects", "professional projects", 
                "notable projects", "key projects", "portfolio", "work samples",
                "achievements", "accomplishments", "notable work", "contributions",
                "other projects"
            ],
            SectionType.CERTIFICATIONS: [
                "certifications", "certificates", "professional certifications", 
                "licenses", "credentials", "awards", "honors", "recognition",
                "professional development", "training", "certification",
                "awards honors"
            ]
        }
        
        self.grammar_keywords = []
        self.labels = []
        self.vectorizer = None
        self.grammar_vectors = None
        self.normalized_exact_headers = {}
        
        # Date pattern for filtering out date-containing lines
        self.date_pattern = re.compile(
            r'\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]?\d{0,4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', 
            re.I
        )
        
        # Time pattern for filtering out time-containing lines
        self.time_pattern = re.compile(
            r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm|AM|PM))\b'
        )
        
        # Day pattern for filtering out day-containing lines
        self.day_pattern = re.compile(
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b', 
            re.I
        )
        
        self._build_grammar_model()
    
    def _build_grammar_model(self):
        """Build the keyword list and TF-IDF model from section grammar"""
        for section_type, keywords in self.section_grammar.items():
            for keyword in keywords:
                self.grammar_keywords.append(keyword)
                self.labels.append(section_type)
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer().fit(self.grammar_keywords)
        self.grammar_vectors = self.vectorizer.transform(self.grammar_keywords)
        
        # Create normalized exact headers for quick matching
        self.normalized_exact_headers = {
            keyword.lower().strip(): self.labels[idx] 
            for idx, keyword in enumerate(self.grammar_keywords)
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def detect_section_headers(self, text: str) -> Tuple[List[HeaderCandidate], List[str]]:
        """Enhanced header detection using TF-IDF and cosine similarity"""
        lines = text.splitlines()
        header_candidates = []
        found_section_types = set()  # Track already found section types
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            line_clean = re.sub(r"[^a-zA-Z ]", " ", line_lower).strip()
            words = line_clean.split()
            
            # Skip empty lines or lines with too many words
            if not line_clean or len(words) > self.max_words_in_header:
                continue
            
            # Reject lines with periods (full stops) - headers shouldn't have periods
            if '.' in line.strip():
                continue
            
            # Reject lines with too many digits or dates/times/days
            digit_count = sum(c.isdigit() for c in line)
            if digit_count > 3:
                continue
            if self.date_pattern.search(line):
                continue
            if self.time_pattern.search(line):
                continue
            if self.day_pattern.search(line):
                continue
            
            # Check for exact match first
            if line_lower in self.normalized_exact_headers:
                label = self.normalized_exact_headers[line_lower]
                
                # Skip if we already found this section type (prevent duplicates)
                if label in found_section_types:
                    continue
                    
                header_candidates.append(HeaderCandidate(
                    line_no=i,
                    line=line.strip(),
                    candidates=[{"label": label.value, "score": 1.0}],
                    status="confident"
                ))
                found_section_types.add(label)
                continue
            
            # Use cosine similarity for fuzzy matching
            try:
                line_vector = self.vectorizer.transform([line_clean])
                similarities = cosine_similarity(line_vector, self.grammar_vectors).flatten()
                
                label_score_map = {}
                for idx, score in enumerate(similarities):
                    if score >= self.threshold:
                        label = self.labels[idx]
                        # Skip if we already found this section type
                        if label in found_section_types:
                            continue
                        
                        # Additional validation: prevent single word partial matches
                        # If the line is a single word and the matched keyword is multi-word,
                        # require higher similarity threshold
                        matched_keyword = self.grammar_keywords[idx]
                        if len(words) == 1 and len(matched_keyword.split()) > 1:
                            # For single word matching multi-word phrases, require exact match or very high similarity
                            if score < 0.95:
                                continue
                        
                        if label not in label_score_map or score > label_score_map[label]:
                            label_score_map[label] = score
                
                if label_score_map:
                    candidates = [
                        {"label": label.value, "score": round(score, 3)} 
                        for label, score in label_score_map.items()
                    ]
                    status = "confident" if len(candidates) == 1 else "ambiguous"
                    
                    # Add the best candidate's section type to found_section_types
                    if candidates:
                        best_section_type = None
                        for section_type in SectionType:
                            if section_type.value == candidates[0]["label"]:
                                best_section_type = section_type
                                break
                        if best_section_type:
                            found_section_types.add(best_section_type)
                    
                    header_candidates.append(HeaderCandidate(
                        line_no=i,
                        line=line.strip(),
                        candidates=sorted(candidates, key=lambda x: -x["score"]),
                        status=status
                    ))
            except Exception as e:
                # Handle cases where vectorizer fails
                continue
        
        return header_candidates, lines
    
    def resolve_ambiguous_headers(self, header_candidates: List[HeaderCandidate]) -> List[HeaderCandidate]:
        """Resolve ambiguous headers by taking the highest scoring candidate"""
        resolved_candidates = []
        
        for candidate in header_candidates:
            if candidate.status == "ambiguous" and candidate.candidates:
                # Take the highest scoring candidate
                best_candidate = candidate.candidates[0]
                resolved_candidate = HeaderCandidate(
                    line_no=candidate.line_no,
                    line=candidate.line,
                    candidates=[best_candidate],
                    status="resolved"
                )
                resolved_candidates.append(resolved_candidate)
            else:
                resolved_candidates.append(candidate)
        
        return resolved_candidates
    
    def extract_sections_from_headers(self, header_candidates: List[HeaderCandidate], lines: List[str]) -> Dict[SectionType, Section]:
        """Extract sections based on detected headers"""
        sections = {}
        resolved_headers = self.resolve_ambiguous_headers(header_candidates)
        
        # Sort headers by line number
        resolved_headers.sort(key=lambda x: x.line_no)
        
        for i, header_candidate in enumerate(resolved_headers):
            if not header_candidate.candidates:
                continue
            
            # Get the best candidate
            best_candidate = header_candidate.candidates[0]
            section_type_str = best_candidate["label"]
            confidence = best_candidate["score"]
            
            # Convert string back to enum
            try:
                section_type = SectionType(section_type_str)
            except ValueError:
                continue
            
            # Determine content boundaries
            start_line = header_candidate.line_no + 1
            
            # Find end line (next header or end of document)
            if i < len(resolved_headers) - 1:
                end_line = resolved_headers[i + 1].line_no
            else:
                end_line = len(lines)
            
            # Extract content
            content_lines = []
            for j in range(start_line, end_line):
                if j < len(lines):
                    line = lines[j].strip()
                    if line and not self._is_section_delimiter(line):
                        content_lines.append(lines[j])
            
            content = '\n'.join(content_lines).strip()
            
            # Only add sections with content
            if content:
                section = Section(
                    section_type=section_type,
                    header=header_candidate.line,
                    content=content,
                    start_line=header_candidate.line_no,
                    end_line=end_line - 1,
                    confidence=confidence
                )
                
                # If multiple sections of same type, keep the one with higher confidence
                if section_type not in sections or confidence > sections[section_type].confidence:
                    sections[section_type] = section
        
        return sections
    
    def _is_section_delimiter(self, line: str) -> bool:
        """Check if line is a section delimiter"""
        delimiters = [r'^={3,}', r'^-{3,}', r'^\*{3,}', r'^_{3,}']
        for pattern in delimiters:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                return True
        return False
    
    def parse_resume(self, resume_text: str) -> Dict:
        """Main method to parse resume and return structured data"""
        # Detect headers
        header_candidates, lines = self.detect_section_headers(resume_text)
        
        # Extract sections
        sections = self.extract_sections_from_headers(header_candidates, lines)
        
        result = {
            "sections_found": len(sections),
            "header_candidates_detected": len(header_candidates),
            "sections": {},
            "debug_info": {
                "header_candidates": [
                    {
                        "line_no": hc.line_no,
                        "line": hc.line,
                        "candidates": hc.candidates,
                        "status": hc.status
                    } for hc in header_candidates
                ]
            }
        }
        
        for section_type, section in sections.items():
            result["sections"][section_type.value] = {
                "header": section.header,
                "content": section.content,
                "start_line": section.start_line,
                "end_line": section.end_line,
                "confidence": section.confidence
            }
        
        return result
    
    def print_parsed_sections(self, parsed_data: Dict) -> None:
        """Pretty print the parsed sections with confidence scores"""
        print(f"Found {parsed_data['sections_found']} sections from {parsed_data['header_candidates_detected']} header candidates:\n")
        print("=" * 60)
        
        # Print debug info
        print("\n🔍 DETECTED HEADER CANDIDATES:")
        for candidate in parsed_data["debug_info"]["header_candidates"]:
            print(f"Line {candidate['line_no']}: '{candidate['line']}' -> {candidate['candidates']} ({candidate['status']})")
        
        print("\n" + "=" * 60)
        print("📑 EXTRACTED SECTIONS:")
        
        for section_name, section_data in parsed_data["sections"].items():
            print(f"\n🔍 SECTION: {section_name.upper()}")
            print(f"Header: {section_data['header']}")
            print(f"Confidence: {section_data['confidence']:.3f}")
            print(f"Lines: {section_data['start_line']} - {section_data['end_line']}")
            print("Content:")
            print("-" * 30)
            print(section_data['content'][:200] + "..." if len(section_data['content']) > 200 else section_data['content'])
            print("=" * 60)