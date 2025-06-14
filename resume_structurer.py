import spacy
from collections import Counter
import re
import time
from datetime import datetime

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

class ResumeStructurer:
    def __init__(self, segmented_dict):
        # Accepts the dictionary under the "sections" key if present
        self.sections = segmented_dict.get("sections", segmented_dict)
        self.structured = {}

    def extract_about_me(self):
        about = self.sections.get("about")
        if about and about.get("content"):
            result = extract_about_me(about["content"])
            self.structured["about_me"] = result
            return result
        return None

    def extract_most_recent_education(self):
        education = self.sections.get("education")
        if education and education.get("content"):
            result = extract_most_recent_education(education["content"])
            self.structured["education"] = result
            return result
        return None

    def extract_most_recent_experience(self):
        experience = self.sections.get("experience")
        if experience and experience.get("content"):
            result = extract_most_recent_experience(experience["content"])
            self.structured["experience"] = result
            return result
        return None

    def extract_projects(self):
        projects = self.sections.get("projects")
        if projects and projects.get("content"):
            result = extract_top_2_projects(projects["content"])
            self.structured["projects"] = result
            return result
        return None

    def get_structured_resume(self):
        # Only extract if section exists
        self.extract_about_me()
        self.extract_most_recent_education()
        self.extract_most_recent_experience()
        self.extract_projects()
        return self.structured

def extract_relevant_lines_and_link(text: str):

    # POS tags typically associated with grammatical "glue" rather than content
    grammar_pos_tags = {"ADP", "PRON", "AUX", "CCONJ", "SCONJ", "DET", "PART", "INTJ", "VERB"}
    modals = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}

    if not text.strip():
        return {"first_line": None, "first_link": None}

    # Split by common delimiters
    split_pattern = re.compile(r'[\n\-;:|]')
    lines = [l.strip() for l in split_pattern.split(text) if l.strip()]

    # Filter lines with minimal grammatical words
    lines_without_grammar = []
    for line in lines:
        doc = nlp(line)
        has_grammar_word = False
        for token in doc:
            if token.pos_ in grammar_pos_tags:
                if token.pos_ == "AUX" and token.lemma_.lower() in modals:
                    has_grammar_word = True
                    break
                elif token.pos_ != "AUX":
                    has_grammar_word = True
                    break
        if not has_grammar_word:
            lines_without_grammar.append(line)

    # Deduplicate lines
    line_counts = Counter(lines_without_grammar)
    unique_lines = [line for line in lines_without_grammar if line_counts[line] == 1]

    if not unique_lines:
        return {"first_line": None, "first_link": None}

    first_line = unique_lines[0]

    # Find the first valid link (if within 5 lines of first line)
    link_pattern = re.compile(r'https?://\S+|www\.\S+|\b\S+\.\S+\b')
    first_link = None
    link_line_num = None

    for idx, line in enumerate(unique_lines):
        match = link_pattern.search(line)
        if match:
            first_link = match.group(0)
            link_line_num = idx
            break

    if link_line_num is not None and (link_line_num - 0) > 5:
        first_link = None

    return {"first_line": first_line, "first_link": first_link}

def extract_most_recent_experience(text):
    """
    Extract the most recent experience from raw experience section text
    
    Args:
        text (str): Raw text from the experience section
    
    Returns:
        dict: {
            'organization_position': str,  # Company name and position combined
            'time_span': str              # Time period of employment
        }
    """
    
    # Define grammar-related POS tags that are likely in job descriptions
    # These are words we want to filter out as they're usually descriptive content
    description_pos_tags = {"ADP", "PRON", "AUX", "CCONJ", "SCONJ", "DET", "PART", "INTJ"}
    description_verbs = {
        "developed", "created", "managed", "led", "implemented", "designed", 
        "worked", "collaborated", "maintained", "optimized", "built", "handled",
        "responsible", "achieved", "delivered", "coordinated", "supervised"
    }
    
    start_time = time.time()
    
    # Split by common separators
    split_pattern = re.compile(r'[\n\-;:|•]')
    lines = [l.strip() for l in split_pattern.split(text) if l.strip()]
    
    # Filter out lines with heavy grammatical/descriptive content
    filtered_lines = []
    for line in lines:
        doc = nlp(line)
        
        # Count grammar words and description indicators
        grammar_count = 0
        total_words = len([token for token in doc if not token.is_space])
        
        if total_words == 0:
            continue
            
        for token in doc:
            # Skip if it's a description-heavy line
            if (token.pos_ in description_pos_tags or 
                token.lemma_.lower() in description_verbs or
                token.text.lower() in ["the", "and", "or", "but", "with", "for", "in", "on", "at"]):
                grammar_count += 1
        
        # If less than 40% grammar words and has substantial content, keep it
        grammar_ratio = grammar_count / total_words if total_words > 0 else 1
        if grammar_ratio < 0.4 and total_words >= 2:
            filtered_lines.append(line)
    
    print(f"Filtered lines (likely company/position/date lines):")
    for i, line in enumerate(filtered_lines):
        print(f"{i+1}: {line}")
    
    # Date patterns to match various formats
    date_patterns = [
        # Full dates with ranges
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to]\s*(?:Present|Current|Ongoing|Now)',
        r'\d{4}\s*[-–—to]\s*\d{4}',
        r'\d{4}\s*[-–—to]\s*(?:Present|Current|Ongoing|Now)',
        r'\d{1,2}/\d{4}\s*[-–—to]\s*\d{1,2}/\d{4}',
        r'\d{1,2}/\d{4}\s*[-–—to]\s*(?:Present|Current|Ongoing|Now)',
        
        # Single dates
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
        r'\d{4}',
        r'\d{1,2}/\d{4}',
        r'(?:Present|Current|Ongoing|Now)',
        
        # Multi-line date ranges (dates split across lines)
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—]?\s*$',
    ]
    
    # Find all lines with dates and their positions
    date_info = []
    
    for idx, line in enumerate(filtered_lines):
        for pattern in date_patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                date_text = match.group(0)
                
                # Check if this might be part of a multi-line date
                if idx < len(filtered_lines) - 1:
                    next_line = filtered_lines[idx + 1]
                    # Check if next line starts with a date that could complete this one
                    continuation_pattern = r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|^(?:Present|Current|Ongoing|Now)'
                    if re.match(continuation_pattern, next_line, re.IGNORECASE):
                        date_text += " " + next_line
                
                # Extract year for sorting (most recent first)
                year_match = re.search(r'\d{4}', date_text)
                year = int(year_match.group()) if year_match else 0
                
                # Handle ongoing/current positions (give them current year + 1 for sorting)
                if re.search(r'(?:Present|Current|Ongoing|Now)', date_text, re.IGNORECASE):
                    year = datetime.now().year + 1
                
                date_info.append({
                    'line_idx': idx,
                    'line_text': line,
                    'date_text': date_text,
                    'year': year,
                    'full_match': match
                })
    
    if not date_info:
        print("No dates found in experience section")
        return {'organization_position': '', 'time_span': ''}
    
    # Sort by year (most recent first), then by line position (earlier in text first)
    date_info.sort(key=lambda x: (-x['year'], x['line_idx']))
    
    print(f"\nFound dates (sorted by recency):")
    for info in date_info:
        print(f"Year: {info['year']}, Line {info['line_idx']+1}: {info['line_text']} -> Date: {info['date_text']}")
    
    # Get the most recent experience
    most_recent = date_info[0]
    most_recent_line_idx = most_recent['line_idx']
    time_span = most_recent['date_text'].strip()
    
    # Find organization and position
    # Look in the current line and nearby lines for company/position info
    org_position_candidates = []
    
    # Check current line (remove the date part)
    current_line = filtered_lines[most_recent_line_idx]
    line_without_date = re.sub(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}(?:\s*[-–—to]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4}|Present|Current|Ongoing|Now)?|\d{4}(?:\s*[-–—to]\s*(?:\d{4}|Present|Current|Ongoing|Now))?|\d{1,2}/\d{4}(?:\s*[-–—to]\s*(?:\d{1,2}/\d{4}|Present|Current|Ongoing|Now))?|Present|Current|Ongoing|Now', '', current_line, flags=re.IGNORECASE).strip()
    
    if line_without_date and len(line_without_date) > 3:
        org_position_candidates.append((current_line, line_without_date))
    
    # Check previous line (often company name comes before date)
    if most_recent_line_idx > 0:
        prev_line = filtered_lines[most_recent_line_idx - 1]
        if not re.search(r'\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', prev_line, re.IGNORECASE):
            org_position_candidates.append((prev_line, prev_line))
    
    # Check next line (sometimes position comes after date)
    if most_recent_line_idx < len(filtered_lines) - 1:
        next_line = filtered_lines[most_recent_line_idx + 1]
        if not re.search(r'\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line, re.IGNORECASE):
            org_position_candidates.append((next_line, next_line))
    
    # Choose the best organization/position candidate
    organization_position = ""
    if org_position_candidates:
        # Prefer longer, more descriptive text
        best_candidate = max(org_position_candidates, key=lambda x: len(x[1]))
        organization_position = best_candidate[1]
    else:
        # Fallback: use the line with date but cleaned
        organization_position = line_without_date if line_without_date else current_line
    
    # Clean up the organization_position
    organization_position = re.sub(r'^\W+|\W+$', '', organization_position)  # Remove leading/trailing punctuation
    organization_position = re.sub(r'\s+', ' ', organization_position)  # Normalize whitespace
    
    end_time = time.time()
    
    result = {
        'organization_position': organization_position,
        'time_span': time_span
    }
    
    print(f"\nExtraction completed in {end_time - start_time:.4f} seconds")
    print(f"Most recent experience found at line {most_recent_line_idx + 1}")
    
    return result

def extract_top_2_projects(text):
    """
    Extract top 2 projects from raw project section text
    
    Args:
        text (str): Raw text from the projects section
    
    Returns:
        list: [
            {'project_name': str, 'project_link': str},
            {'project_name': str, 'project_link': str}
        ]
    """
    
    # Enhanced grammar-related POS tags for filtering descriptions
    grammar_pos_tags = {"ADP", "PRON", "AUX", "CCONJ", "SCONJ", "DET", "PART", "INTJ", "VERB"}
    
    # Common modals and description words to filter out
    modals = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
    description_words = {
        "developed", "created", "built", "designed", "implemented", "used", "utilizing",
        "including", "featuring", "contains", "provides", "allows", "enables", "helps",
        "supports", "offers", "delivers", "focuses", "aims", "targets", "designed",
        "this", "that", "these", "those", "which", "where", "when", "what", "how",
        "the", "and", "or", "but", "with", "for", "in", "on", "at", "by", "from"
    }
    
    start_time = time.time()
    
    # Enhanced splitting pattern - more comprehensive separators
    split_pattern = re.compile(r'[\n\r\-;:|•★▪▫‣⁃◦∙]|(?:\d+\.)')
    lines = [l.strip() for l in split_pattern.split(text) if l.strip()]
    
    # Remove very short lines (likely artifacts)
    lines = [line for line in lines if len(line) > 2]
    
    print(f"Initial lines after splitting ({len(lines)} lines):")
    for i, line in enumerate(lines[:10]):  # Show first 10 for debugging
        print(f"{i+1}: {line}")
    
    # Filter out lines with heavy grammatical content
    lines_without_grammar = []
    
    for line in lines:
        doc = nlp(line)
        total_tokens = len([token for token in doc if not token.is_space and not token.is_punct])
        
        if total_tokens == 0:
            continue
            
        grammar_count = 0
        description_count = 0
        
        for token in doc:
            token_lower = token.lemma_.lower()
            
            # Count grammar words
            if token.pos_ in grammar_pos_tags:
                if token.pos_ == "AUX" and token_lower in modals:
                    grammar_count += 1
                elif token.pos_ != "AUX":
                    grammar_count += 1
            
            # Count description words
            if token_lower in description_words:
                description_count += 1
        
        # Calculate ratios
        grammar_ratio = grammar_count / total_tokens
        description_ratio = description_count / total_tokens
        
        # Keep lines that are likely project names/titles
        # Filter out if too much grammar (>30%) or description words (>25%)
        if grammar_ratio <= 0.3 and description_ratio <= 0.25 and total_tokens >= 1:
            lines_without_grammar.append(line)
    
    print(f"\nFiltered lines without heavy grammar ({len(lines_without_grammar)} lines):")
    for i, line in enumerate(lines_without_grammar):
        print(f"{i+1}: {line}")
    
    # Remove duplicate lines
    line_counts = Counter(lines_without_grammar)
    unique_lines = [line for line in lines_without_grammar if line_counts[line] == 1]
    
    print(f"\nUnique lines after deduplication ({len(unique_lines)} lines):")
    for i, line in enumerate(unique_lines):
        print(f"{i+1}: {line}")
    
    # Enhanced link detection patterns
    link_patterns = [
        r'https?://[^\s<>"{}|\\^`\[\]]+',  # Standard HTTP/HTTPS URLs
        r'www\.[^\s<>"{}|\\^`\[\]]+',      # www URLs
        r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.(?:com|org|net|io|co|in|edu|gov|app|dev|tech|me|ly|it|us|uk|ca|de|fr|jp|au|br|ru|cn|live|site|online|website|web|digital|blog|info|biz|pro|name|mobi|cc|tv|fm|am|github\.io|herokuapp\.com|vercel\.app|netlify\.app|firebase\.app|github\.com|gitlab\.com|bitbucket\.org)(?:/[^\s<>"{}|\\^`\[\]]*)?',  # Domain-based URLs
        r'[a-zA-Z0-9.-]+\.(?:github\.io|herokuapp\.com|vercel\.app|netlify\.app|firebase\.app)',  # Specific hosting platforms
        r'github\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+',  # GitHub repositories
        r'gitlab\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+',  # GitLab repositories
    ]
    
    # Extract projects with their associated links
    projects = []
    
    for idx, line in enumerate(unique_lines):
        # Check for links in current line
        found_links = []
        for pattern in link_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            found_links.extend(matches)
        
        # If no link in current line, search in nearby lines (within 3 lines)
        if not found_links:
            search_range = range(max(0, idx-2), min(len(unique_lines), idx+4))
            for search_idx in search_range:
                if search_idx != idx:  # Don't search current line again
                    search_line = unique_lines[search_idx]
                    for pattern in link_patterns:
                        matches = re.findall(pattern, search_line, re.IGNORECASE)
                        if matches:
                            found_links.extend(matches)
                            break
                    if found_links:
                        break
        
        # Clean and validate project name
        project_name = line.strip()
        
        # Remove common prefixes/suffixes that might be artifacts
        project_name = re.sub(r'^(?:project\s*:?\s*|•\s*|\d+\.\s*)', '', project_name, flags=re.IGNORECASE)
        project_name = re.sub(r'\s*[:-]\s*$', '', project_name)
        
        # Skip if project name is too short or looks like a section header
        if (len(project_name) < 3 or 
            project_name.lower() in ['projects', 'project', 'work', 'portfolio', 'experience'] or
            len(project_name.split()) > 8):  # Too long, likely a description
            continue
        
        # Get the best link (prefer https, then www, then others)
        best_link = ""
        if found_links:
            # Sort links by preference
            https_links = [link for link in found_links if link.startswith('https://')]
            www_links = [link for link in found_links if link.startswith('www.')]
            other_links = [link for link in found_links if not link.startswith(('https://', 'www.'))]
            
            if https_links:
                best_link = https_links[0]
            elif www_links:
                best_link = www_links[0]
            elif other_links:
                best_link = other_links[0]
            
            # Clean the link
            best_link = best_link.rstrip('.,;:!?)')
        
        projects.append({
            'project_name': project_name,
            'project_link': best_link,
            'line_index': idx
        })
    
    # Remove projects with very similar names (likely duplicates)
    filtered_projects = []
    for project in projects:
        is_duplicate = False
        for existing in filtered_projects:
            # Check similarity based on common words
            name1_words = set(project['project_name'].lower().split())
            name2_words = set(existing['project_name'].lower().split())
            
            if name1_words and name2_words:
                similarity = len(name1_words & name2_words) / len(name1_words | name2_words)
                if similarity > 0.6:  # 60% similarity threshold
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_projects.append(project)
    
    # Sort projects by their appearance order (topmost first)
    filtered_projects.sort(key=lambda x: x['line_index'])
    
    # Return top 2 projects
    top_2_projects = filtered_projects[:2]
    
    # Format result
    result = []
    for project in top_2_projects:
        result.append({
            'project_name': project['project_name'],
            'project_link': project['project_link'] if project['project_link'] else ""
        })
    # Fill with empty entries if less than 2 projects found
    while len(result) < 2:
        result.append({'project_name': "", 'project_link': ""})

    end_time = time.time()

    print(f"\nProject extraction completed in {end_time - start_time:.4f} seconds")
    print(f"Found {len(filtered_projects)} total projects, returning top 2")

    return {"top_2_projects": result}

def extract_about_me(text):
    """
    Extract about me section from raw text using grammar-based analysis
    
    Args:
        text (str): Raw text from the about me/introduction section
    
    Returns:
        str: Extracted about me content
    """
    
    # Define grammar patterns that are safe/common in self-introductions
    # We want to identify lines with personal descriptive language
    intro_pos_tags = {"VERB", "ADJ", "NOUN", "PROPN"}  # Key content words
    
    # Personal pronouns and descriptive words that indicate self-introduction
    personal_indicators = {
        "i", "my", "me", "myself", "am", "was", "have", "had", "been", "being",
        "passionate", "experienced", "dedicated", "skilled", "enthusiastic",
        "motivated", "creative", "professional", "developer", "engineer",
        "student", "graduate", "specialist", "expert"
    }
    
    # Verbs commonly used in self-introduction
    intro_verbs = {
        "am", "work", "specialize", "focus", "enjoy", "love", "like", "pursue",
        "study", "learn", "develop", "create", "build", "design", "manage"
    }
    
    start_time = time.time()
    
    # Split text by lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return {"about_me": ""}

    # Find the first line that contains introduction-style grammar
    intro_start_idx = None
    
    for idx, line in enumerate(lines):
        doc = nlp(line.lower())
        
        # Check if line contains personal introduction patterns
        has_personal_indicator = False
        has_descriptive_content = False
        
        # Look for personal indicators
        for token in doc:
            if token.lemma_.lower() in personal_indicators:
                has_personal_indicator = True
                break
        
        # Look for descriptive content (adjectives, verbs, nouns)
        content_words = 0
        for token in doc:
            if token.pos_ in intro_pos_tags and not token.is_stop:
                content_words += 1
            if token.lemma_.lower() in intro_verbs:
                has_descriptive_content = True
        
        # Consider it an intro line if it has personal indicators or 
        # descriptive content with reasonable length
        if (has_personal_indicator or 
            (has_descriptive_content and content_words >= 2 and len(line.split()) >= 4)):
            intro_start_idx = idx
            break
    
    # If no clear intro pattern found, check for lines with first-person pronouns
    if intro_start_idx is None:
        for idx, line in enumerate(lines):
            line_lower = line.lower()
            if any(pronoun in line_lower.split() for pronoun in ["i", "my", "me"]):
                intro_start_idx = idx
                break
    
    # If still no pattern found, start from the first substantial line
    if intro_start_idx is None:
        for idx, line in enumerate(lines):
            if len(line.split()) >= 5:  # At least 5 words
                intro_start_idx = idx
                break
    
    # If nothing found, return the first line
    if intro_start_idx is None:
        intro_start_idx = 0
    
    # Extract from the identified start line to the end
    about_me_content = '\n'.join(lines[intro_start_idx:])
    
    end_time = time.time()
    
    print(f"About Me extraction time: {end_time - start_time:.4f} seconds")
    print(f"Started extraction from line {intro_start_idx + 1}")
    
    return {"about_me": about_me_content}

def extract_most_recent_education(text):
    """
    Extract the most recent education from raw education section text
    
    Args:
        text (str): Raw text from the education section
    
    Returns:
        dict: {
            'organization': str,     # Institute/University name
            'degree': str,          # Degree and subject
            'time_span': str        # Time period
        }
    """
    
    # Comprehensive degree collection from around the world
    degrees = {
        # Bachelor's Degrees
        'bachelor': ['bachelor', 'bachelors', 'b.tech', 'btech', 'b.e', 'be', 'b.sc', 'bsc', 'b.com', 'bcom', 
                    'b.a', 'ba', 'b.des', 'bdes', 'b.arch', 'barch', 'b.pharm', 'bpharm', 'b.ed', 'bed',
                    'b.b.a', 'bba', 'b.c.a', 'bca', 'b.f.a', 'bfa', 'b.voc', 'bvoc', 'b.lib', 'blib',
                    'bs', 'b.s', 'ab', 'sb', 'beng', 'b.eng', 'bachelor of technology', 'bachelor of engineering',
                    'bachelor of science', 'bachelor of arts', 'bachelor of commerce', 'bachelor of computer applications',
                    'bachelor of business administration', 'bachelor of fine arts', 'bachelor of education',
                    'bachelor of architecture', 'bachelor of pharmacy', 'bachelor of design', 'licence', 'licenciatura'],
        
        # Master's Degrees
        'master': ['master', 'masters', 'm.tech', 'mtech', 'm.e', 'me', 'm.sc', 'msc', 'm.com', 'mcom',
                  'm.a', 'ma', 'm.des', 'mdes', 'm.arch', 'march', 'm.pharm', 'mpharm', 'm.ed', 'med',
                  'm.b.a', 'mba', 'm.c.a', 'mca', 'm.f.a', 'mfa', 'm.lib', 'mlib', 'm.phil', 'mphil',
                  'ms', 'm.s', 'meng', 'm.eng', 'master of technology', 'master of engineering', 'master of science',
                  'master of arts', 'master of commerce', 'master of computer applications', 'master of business administration',
                  'master of fine arts', 'master of education', 'master of architecture', 'master of pharmacy',
                  'master of design', 'master of philosophy', 'master of library science', 'magistr', 'maestria'],
        
        # Doctoral Degrees
        'doctoral': ['phd', 'ph.d', 'doctor', 'doctorate', 'd.phil', 'dphil', 'sc.d', 'scd', 'd.sc', 'dsc',
                    'ed.d', 'edd', 'doctor of philosophy', 'doctor of science', 'doctor of education',
                    'doctor of medicine', 'md', 'm.d', 'mbbs', 'm.b.b.s', 'doctor of dental surgery',
                    'dds', 'd.d.s', 'doctor of veterinary medicine', 'dvm', 'd.v.m', 'juris doctor', 'jd', 'j.d'],
        
        # Diploma & Certificate
        'diploma': ['diploma', 'certificate', 'cert', 'advanced diploma', 'graduate diploma', 'postgraduate diploma',
                   'pgd', 'p.g.d', 'associate', 'associates', 'a.a', 'aa', 'a.s', 'as', 'a.a.s', 'aas',
                   'associate of arts', 'associate of science', 'associate of applied science', 'diplome'],
        
        # Professional Degrees
        'professional': ['ca', 'c.a', 'chartered accountant', 'cs', 'c.s', 'company secretary', 'cma', 'c.m.a',
                        'cost and management accountant', 'cfa', 'c.f.a', 'chartered financial analyst',
                        'frm', 'f.r.m', 'financial risk manager', 'pmp', 'p.m.p', 'project management professional',
                        'cissp', 'c.i.s.s.p', 'cisa', 'c.i.s.a', 'cpa', 'c.p.a', 'certified public accountant'],
        
        # International Degrees
        'international': ['llb', 'll.b', 'llm', 'll.m', 'bachelor of laws', 'master of laws', 'baccalaureat',
                         'abitur', 'a-levels', 'ib', 'international baccalaureate', 'gcse', 'o-levels',
                         'higher secondary', 'intermediate', 'hsc', 'h.s.c', 'class 12', '12th', 'grade 12',
                         'high school', 'secondary school', 'matriculation', 'ssc', 's.s.c', 'class 10', '10th']
    }
    
    # Flatten all degrees into a single list for easier matching
    all_degrees = []
    for category, degree_list in degrees.items():
        all_degrees.extend(degree_list)
    
    # Create regex pattern for degree matching
    degree_pattern = r'\b(?:' + '|'.join(re.escape(degree) for degree in all_degrees) + r')\b'
    
    start_time = time.time()
    
    # Split text into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    print(f"Processing {len(lines)} lines from education section:")
    for i, line in enumerate(lines[:10]):  # Show first 10 for debugging
        print(f"{i+1}: {line}")
    
    # Date patterns for education (similar to experience but adapted)
    date_patterns = [
        # Graduation years and ranges
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to]\s*(?:Present|Current|Ongoing|Now|Expected)',
        r'\d{4}\s*[-–—to]\s*\d{4}',
        r'\d{4}\s*[-–—to]\s*(?:Present|Current|Ongoing|Now|Expected)',
        r'(?:Graduated|Graduation|Completed|Expected|Class of)\s+(?:in\s+)?\d{4}',
        r'(?:Batch|Year)\s+(?:of\s+)?\d{4}',
        
        # Single years
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
        r'\b\d{4}\b(?!\s*[-–—])',  # Year not followed by dash (to avoid splitting ranges)
        r'(?:Graduating|Expected)\s+\d{4}',
    ]
    
    # Find lines with degrees and dates
    education_entries = []
    
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        
        # Check for degree in current line
        degree_matches = re.findall(degree_pattern, line_lower, re.IGNORECASE)
        
        # Check for dates in current line
        date_matches = []
        for pattern in date_patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            date_matches.extend(matches)
        
        # If degree found, look for associated date and organization
        if degree_matches:
            degree_text = degree_matches[0]  # Take the first/best match
            date_text = ""
            organization = ""
            
            # Look for date in current line first
            if date_matches:
                date_text = date_matches[0]
            else:
                # Search nearby lines for dates (within 2 lines)
                search_range = range(max(0, idx-2), min(len(lines), idx+3))
                for search_idx in search_range:
                    if search_idx != idx:
                        search_line = lines[search_idx]
                        for pattern in date_patterns:
                            matches = re.findall(pattern, search_line, re.IGNORECASE)
                            if matches:
                                date_text = matches[0]
                                break
                        if date_text:
                            break
            
            # Extract year for sorting
            year_match = re.search(r'\d{4}', date_text)
            year = int(year_match.group()) if year_match else 0
            
            # Handle expected/current (give them current year + 1 for sorting)
            if re.search(r'(?:Present|Current|Ongoing|Now|Expected|Graduating)', date_text, re.IGNORECASE):
                year = datetime.now().year + 1
            
            # Find organization name
            # Remove degree and date from current line to get organization
            line_clean = line
            for match in degree_matches:
                line_clean = re.sub(re.escape(match), '', line_clean, flags=re.IGNORECASE)
            if date_text:
                line_clean = re.sub(re.escape(date_text), '', line_clean, flags=re.IGNORECASE)
            
            # Clean organization name
            line_clean = re.sub(r'[-–—,;:|]', ' ', line_clean)  # Replace separators with space
            line_clean = re.sub(r'\s+', ' ', line_clean).strip()  # Normalize whitespace
            line_clean = re.sub(r'^\W+|\W+$', '', line_clean)  # Remove leading/trailing punctuation
            
            if line_clean and len(line_clean) > 2:
                organization = line_clean
            else:
                # Look in adjacent lines for organization
                search_range = [idx-1, idx+1]
                for search_idx in search_range:
                    if 0 <= search_idx < len(lines):
                        search_line = lines[search_idx]
                        # Check if this line doesn't contain degree or date patterns
                        has_degree = bool(re.search(degree_pattern, search_line, re.IGNORECASE))
                        has_date = any(re.search(pattern, search_line, re.IGNORECASE) for pattern in date_patterns)
                        
                        if not has_degree and not has_date and len(search_line.strip()) > 3:
                            organization = search_line.strip()
                            break
            
            # Create full degree description (include subject if available)
            full_degree = line
            if organization:
                # Remove organization from degree description
                full_degree = re.sub(re.escape(organization), '', full_degree, flags=re.IGNORECASE)
            if date_text:
                # Remove date from degree description
                full_degree = re.sub(re.escape(date_text), '', full_degree, flags=re.IGNORECASE)
            
            full_degree = re.sub(r'[-–—,;:|]', ' ', full_degree)
            full_degree = re.sub(r'\s+', ' ', full_degree).strip()
            full_degree = re.sub(r'^\W+|\W+$', '', full_degree)
            
            education_entries.append({
                'line_idx': idx,
                'degree': full_degree if full_degree else degree_text,
                'organization': organization,
                'time_span': date_text,
                'year': year
            })
    
    if not education_entries:
        print("No education entries found")
        return {'organization': '', 'degree': '', 'time_span': ''}
    
    # Sort by year (most recent first), then by line position
    education_entries.sort(key=lambda x: (-x['year'], x['line_idx']))
    
    print(f"\nFound education entries (sorted by recency):")
    for entry in education_entries:
        print(f"Year: {entry['year']}, Line {entry['line_idx']+1}")
        print(f"  Degree: {entry['degree']}")
        print(f"  Organization: {entry['organization']}")
        print(f"  Time: {entry['time_span']}")
        print("-" * 40)
    
    # Get the most recent education
    most_recent = education_entries[0]
    
    # Clean up the results
    organization = most_recent['organization'].strip()
    degree = most_recent['degree'].strip()
    time_span = most_recent['time_span'].strip()
    
    # Final cleanup
    if not organization and degree:
        # Try to extract organization from degree if they're combined
        parts = degree.split('-')
        if len(parts) > 1:
            # Check if one part looks like an organization
            for part in parts:
                part = part.strip()
                if (len(part) > 10 and 
                    not re.search(degree_pattern, part, re.IGNORECASE) and
                    ('university' in part.lower() or 'college' in part.lower() or 
                     'institute' in part.lower() or 'school' in part.lower())):
                    organization = part
                    degree = degree.replace(part, '').replace('-', '').strip()
                    break
    
    result = {
        'organization': organization,
        'degree': degree,
        'time_span': time_span
    }
    
    end_time = time.time()
    print(f"\nEducation extraction completed in {end_time - start_time:.4f} seconds")
    
    return result
