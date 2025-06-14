import spacy
import time
from collections import Counter
import re

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Your input text
text = """SPSU SPORTS ERP(LIVE PROJECT) 2024\nWe developed a comprehensive Sports ERP system designed to automate\nrequests, efficiently track inventory, and manage overdue items. This\ninnovative solution offers cost-effective services that streamline\noperations, saving valuable time and improving decision-making processes.\nResponsible For:\nLed a team of 4 in workflow management and development.\nHandled front-end development, serverless Firebase functions, API\nintegration, and database design.\nLink: sportserp.spsu.ac.in\nHandFlow Home Automation (TESTED PROJECT) 2023-2024\nImplemented a real-time hand gesture recognition system for home\nautomation, utilizing an ESP32 microcontroller and a custom model\n'HandFlow' based on Google MediaPipe. User control electronic devices\nthrough gestures via a web interface, with signals transmitted to Firebase\nand actions executed by the ESP32.\nResponsible For:\nCo-Led a team of 5 in the development and integration of AI models.\nCollaborated closely with IoT device experts to establish a hardware\ntesting environment.\nDeveloped a website for seamless integration of AI and hardware,\nutilizing Firebase Realtime Database for control and data\nmanagement.\nLink: self-nasu.github.io/HandFlow/\nDropout Rate Factor Identification - 2023\nSmart India Hackathon\nWe proposed a solution that aims to address the critical issue of high\ndropout rates in schools across Gujarat. We have developed an ML-\npowered solution that can predict dropout rates and recommend targeted\ninterventions to reduce dropout rates effectively.\nResponsible For:\nDeveloped the front-end UI/UX, focusing on user-friendly design.\nUtilized modular libraries to create dynamic bar and table charts for\neffective data visualization on the user dashboard.\nContributed as a team member in developing a machine learning-\npowered solution.\nLink: self-nasu.github.io/droprate.github.io/\nOTHER PROJECTS\nAvailable on GitHub\nPersonal Portfolio Website - React js and Node Js\n(Currently Working on)\nEasyAI Python Lib (New Open Source) - available on\npip packages. (Currently Working on)\nJRSC Jhadol Website - Responsive. (Freelance work)\nMusic player with dynamic music uploads and file\nfetching using NodeJS Backend.\nApp Store Clone - Responsive.\nVerdure: Explore Medicinal Plants with Advanced\nSearch, 3D Models using MySQL and NodeJS.\nYoLo Object Detection using OpenCV."""

# Define grammar-related POS tags for the specified categories
# Prepositions: ADP
# Pronouns: PRON
# Auxiliary verbs: AUX
# Conjunctions: CCONJ, SCONJ
# Determiners: DET
# Modals: (subset of AUX, handled by lemma check)
grammar_pos_tags = {"ADP", "PRON", "AUX", "CCONJ", "SCONJ", "DET", "PART", "INTJ", "VERB"}
modals = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}

start_time = time.time()

# Split by lines using \n, -, ;, :, |
split_pattern = re.compile(r'[\n\-;:|]')
lines = [l.strip() for l in split_pattern.split(text) if l.strip()]
lines_without_grammar = []

for line in lines:
    doc = nlp(line)
    has_grammar_word = False
    for token in doc:
        if token.pos_ in grammar_pos_tags:
            # For AUX, check if it's a modal
            if token.pos_ == "AUX" and token.lemma_.lower() in modals:
                has_grammar_word = True
                break
            elif token.pos_ != "AUX":
                has_grammar_word = True
                break
    if not has_grammar_word:
        lines_without_grammar.append(line)

end_time = time.time()

# Remove lines that appear more than once
line_counts = Counter(lines_without_grammar)
unique_lines = [line for line in lines_without_grammar if line_counts[line] == 1]

# Output result
print("Lines without grammar-related words:")
for line in unique_lines:
    print(line)

print(f"\nFunction execution time: {end_time - start_time:.4f} seconds")

def main():
    # Use the filtered, deduplicated lines
    filtered_lines = unique_lines
    first_line = filtered_lines[0].strip() if filtered_lines else None
    link_pattern = re.compile(r'https?://\S+|www\.\S+|\b\S+\.\S+\b')
    link_line_num = None
    first_link = None
    for idx, line in enumerate(filtered_lines):
        match = link_pattern.search(line)
        if match:
            first_link = match.group(0)
            link_line_num = idx
            break
    # If the link is much farther than the first line, set to None (threshold: 5 lines)
    if link_line_num is not None and (link_line_num - 0) > 5:
        first_link = None
    print(f"First line: {first_line}")
    print(f"First link: {first_link}")

if __name__ == "__main__":
    main()
