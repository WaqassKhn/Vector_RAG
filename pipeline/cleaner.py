import re
import unicodedata
from typing import List, Dict, Any

class DocumentCleaner:
    """
    Robust text and table cleaner designed for noisy corporate and financial documents.
    Fixes PDF line splits, normalizes Unicode symbols, repairs damaged table structures,
    and preserves numerical metadata relationships.
    """

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize Unicode characters, smart quotes, dashes, and whitespace."""
        if not text:
            return ""
        # NFKC normalization brings compatibility characters to standard forms
        text = unicodedata.normalize("NFKC", text)
        
        # Replace non-breaking spaces, zero-width spaces, and odd white spaces
        text = re.sub(r'[\xa0\u200b\u200e\u200f\u202f]', ' ', text)
        
        # Normalize quotes and dashes
        text = re.sub(r'[“”„]', '"', text)
        text = re.sub(r'[‘’`]', "'", text)
        text = re.sub(r'[—–−]', '-', text)
        
        return text

    @staticmethod
    def fix_line_breaks(text: str) -> str:
        """
        Fix broken lines caused by PDF wrapping or OCR formatting.
        Connects split words (e.g. 're- \\n revenue' -> 'rerevenue' or 'cor- \\n poration' -> 'corporation')
        and merges soft line breaks inside sentences.
        """
        if not text:
            return ""

        # Repair hyphenated word breaks across newlines
        text = re.sub(r'(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)', r'\1\2', text)
        
        # Replace single newlines within sentences with space, keeping double newlines (paragraphs)
        # Match lines not ending with sentence punctuation (.!?):
        lines = text.split('\n')
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                cleaned_lines.append("")
                i += 1
                continue
            
            # If line ends without sentence-ending punctuation and next line starts with lowercase or digit
            if (i + 1 < len(lines) and 
                lines[i + 1].strip() and 
                not line.endswith(('.', ':', ';', '!', '?', '|')) and 
                not lines[i + 1].strip().startswith(('#', '|', '-', '*', 'Table', 'TABLE'))):
                
                # Check if it looks like a tabular row
                if not DocumentCleaner._is_table_row(line):
                    line = line + " " + lines[i + 1].strip()
                    i += 1 # skip next line as it's merged
            
            cleaned_lines.append(line)
            i += 1

        res = "\n".join(cleaned_lines)
        # Collapse multi-spaces (except inside markdown tables)
        res = re.sub(r'[ \t]+', ' ', res)
        # Collapse 3+ newlines into 2
        res = re.sub(r'\n{3,}', '\n\n', res)
        return res.strip()

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if a string line appears to be a table row or tabular data."""
        if '|' in line:
            return True
        # Check if line contains 2 or more distinct numeric values separated by multiple spaces
        numbers = re.findall(r'\b\$?\d+(?:\.\d+)?%?\b', line)
        if len(numbers) >= 2 and re.search(r'\s{2,}', line):
            return True
        return False

    @staticmethod
    def format_raw_table_to_markdown(table_rows: List[List[Any]]) -> str:
        """
        Convert extracted 2D table arrays (e.g. from pdfplumber or pandas) into clean Markdown tables.
        Ensures numerical columns stay clearly aligned with row headers.
        """
        if not table_rows:
            return ""
        
        # Clean each cell
        cleaned_table = []
        for row in table_rows:
            cleaned_row = []
            for cell in row:
                cell_str = str(cell) if cell is not None else ""
                cell_str = cell_str.replace('\n', ' ').strip()
                cell_str = DocumentCleaner.normalize_unicode(cell_str)
                # Escape pipe characters inside cells
                cell_str = cell_str.replace('|', '\\|')
                cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""

        headers = cleaned_table[0]
        # Make unique non-empty headers
        headers = [h if h else f"Col_{idx+1}" for idx, h in enumerate(headers)]
        
        md_lines = []
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in cleaned_table[1:]:
            # Pad or trim row to match header length
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            md_lines.append("| " + " | ".join(row) + " |")
            
        return "\n".join(md_lines)

    @classmethod
    def clean_document_text(cls, text: str) -> str:
        """Main entry point to clean raw document text."""
        if not text:
            return ""
        text = cls.normalize_unicode(text)
        text = cls.fix_line_breaks(text)
        return text
