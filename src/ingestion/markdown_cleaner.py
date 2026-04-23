import re
import html
import logging

logger = logging.getLogger(__name__)

class MarkdownCleaner:
    """Utility to clean up OCR and parsing artifacts from Docling markdown/JSON output."""

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""

        # 1. Unescape HTML entities (e.g., &#124; -> |, &amp; -> &)
        text = html.unescape(text)

        # 2. Remove Devanagari (Hindi) characters
        text = re.sub(r'[\u0900-\u097F]+', '', text)

        # 3. Remove non-printable / garbage ASCII symbols and specific OCR noise
        # This keeps common punctuation but removes things like ƒ, Œ, ‡, ˆ, ¡, ¤, etc.
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # 4. Fix "spaced-out" words (e.g., "s o u r c e" -> "source")
        # Heuristic: Find sequences of single characters separated by single spaces
        # We look for 2 or more single chars in a row.
        
        # Join sequences like "E l e c t r i c a l" or "s o u r c e"
        # Match a letter, then 1-3 repetitions of (space + letter)
        def join_match(m):
            return m.group(0).replace(" ", "")

        # Pass 1: Handle lowercase sequences (most common in OCR)
        # Match word char, then multiple (space + word char)
        text = re.sub(r'\b[a-zA-Z](?: [a-zA-Z]){2,}\b', join_match, text)
        
        # Pass 2: Specific broken words found in the user's files
        broken_word_artifacts = [
            r'sour c e', r'd a te', r'v e r s i o n', r's e v e r i t y', 
            r'p a ge _ n u m b e r', r's e c t i o n _ p a t h', r'c h u n k _ i d',
            r'm a r k d o w n', r's e m a n t i c', r'k e y w o r d', r's e a r c h',
            r't a g s', r'i s s u e d', r'do c ling', r'do c _ type'
        ]
        for art in broken_word_artifacts:
            # Replace spaces in the artifact pattern with " " and match case-insensitively
            pattern = art.replace(" ", r" ?")
            text = re.sub(pattern, lambda m: m.group(0).replace(" ", ""), text, flags=re.IGNORECASE)

        # Pass 3: Handle things like "ca l" -> "cal" in "Electrical"
        # Match 1-2 chars surrounded by letters
        text = re.sub(r'(?<=[a-zA-Z]) ([a-zA-Z]{1,2}) (?=[a-zA-Z])', r'\1', text)
        text = re.sub(r'(?<=[a-zA-Z]) ([a-zA-Z]{1,2})\b', r'\1', text)

        # 5. Clean up corrupted markdown headers (e.g., ## "|Heading")
        text = re.sub(r'##\s*["\'|]+', '## ', text)
        text = re.sub(r'#+\s*["\'|]+', '## ', text) # Normalize to ## for simplicity or keep #

        # 6. Normalize whitespace (but keep pipes for table structure)
        text = re.sub(r'[ \t]{2,}', ' ', text) # Collapse multiple spaces/tabs to one
        
        # 7. Remove weird random single-char noise like "f g " or " u " if they are surrounded by pipes
        # Only if it's truly noise (e.g., single letters that aren't abbreviations)
        # We'll be conservative here to avoid deleting valid narrow columns (like "M/F")
        # text = re.sub(r'\|\s*[a-zA-Z]\s*\|', '|', text) # Commented out: too aggressive for directories
        
        # 8. Fix malformed markdown tables (missing headers, mismatched column separators)
        text = MarkdownCleaner.fix_markdown_tables(text)
        
        # 9. Final trim
        return text.strip()

    @staticmethod
    def fix_markdown_tables(text: str) -> str:
        """Ensures every markdown table has a valid separator row matching the header columns."""
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        import re
        while i < len(lines):
            line = lines[i]
            line_str = line.strip()
            # Heuristic for table row: starts with | and has at least one other |
            is_table_line = line_str.startswith('|') and line_str.count('|') >= 2
            
            if is_table_line:
                table_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    next_str = next_line.strip()
                    if next_str.startswith('|') and next_str.count('|') >= 2:
                        table_lines.append(next_line)
                        i += 1
                    else:
                        break
                
                # Fix the table_lines block
                header = table_lines[0].strip()
                if not header.endswith('|'):
                    header += '|'
                    table_lines[0] = header
                
                num_cols = header.count('|') - 1
                
                if len(table_lines) > 1:
                    second_line = table_lines[1].strip()
                    # Check if second line is a separator row
                    is_sep = bool(re.match(r'^[\s\|\-\:]+$', second_line)) and '-' in second_line
                    
                    new_sep = '|' + '|'.join(['---'] * num_cols) + '|'
                    if is_sep:
                        table_lines[1] = new_sep
                    else:
                        table_lines.insert(1, new_sep)
                else:
                    new_sep = '|' + '|'.join(['---'] * num_cols) + '|'
                    table_lines.append(new_sep)
                
                # Ensure all other rows end with '|'
                for j in range(2 if len(table_lines) > 1 else 1, len(table_lines)):
                    row = table_lines[j].strip()
                    if row != new_sep and not row.endswith('|'):
                        table_lines[j] = table_lines[j] + '|'
                
                fixed_lines.extend(table_lines)
            else:
                fixed_lines.append(line)
                i += 1
                
        return '\n'.join(fixed_lines)

    @classmethod
    def clean_json(cls, obj):
        """Recursively apply clean_text to all strings in a JSON-like structure."""
        if isinstance(obj, str):
            return cls.clean_text(obj)
        if isinstance(obj, dict):
            return {k: cls.clean_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls.clean_json(item) for item in obj]
        return obj

if __name__ == "__main__":
    # Test cases
    test_text = "sour c e (document name), d a te\\_issued , t a gs (e.g., fire, cooling), version , p a ge\\_num b er , se c tion\\_p a th"
    print(f"Original: {test_text}")
    print(f"Cleaned:  {MarkdownCleaner.clean_text(test_text)}")
    
    test_table = "| 1 | w &#124;Product Name : EXIDE | f g |"
    print(f"Original Table: {test_table}")
    print(f"Cleaned Table:  {MarkdownCleaner.clean_text(test_table)}")
