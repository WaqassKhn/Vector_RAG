import re
from typing import List, Dict, Any
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class DocumentChunker:
    """
    Header & Table Aware Document Chunker.
    Keeps numerical values, table rows, and structural context together across chunks.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_parsed_document(self, parsed_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes output from DocumentParser and chunks it while preserving page numbers and section metadata.
        """
        filename = parsed_doc.get("filename", "unknown_doc")
        pages = parsed_doc.get("pages", [])
        
        chunks = []
        chunk_idx = 0

        for page in pages:
            page_num = page.get("page_number", 1)
            cleaned_text = page.get("cleaned_text", "")
            
            if not cleaned_text.strip():
                continue

            # Split text by double newlines or table blocks
            blocks = self._split_into_structural_blocks(cleaned_text)
            
            current_chunk_text = ""
            current_section_header = f"Document: {filename} (Page {page_num})"

            for block in blocks:
                # Track latest header if block is a section title
                if block.startswith('#') or block.isupper() and len(block) < 60:
                    current_section_header = f"Doc: {filename} | Section: {block.strip('# ').strip()}"

                # If single block is larger than chunk_size, split by sentences/lines
                if len(block) > self.chunk_size:
                    if current_chunk_text.strip():
                        chunks.append(self._create_chunk_object(
                            chunk_id=f"{filename}_p{page_num}_c{chunk_idx}",
                            text=current_chunk_text.strip(),
                            filename=filename,
                            page_num=page_num,
                            header_context=current_section_header
                        ))
                        chunk_idx += 1
                        current_chunk_text = ""
                    
                    sub_blocks = self._split_large_block(block)
                    for sb in sub_blocks:
                        chunks.append(self._create_chunk_object(
                            chunk_id=f"{filename}_p{page_num}_c{chunk_idx}",
                            text=sb.strip(),
                            filename=filename,
                            page_num=page_num,
                            header_context=current_section_header
                        ))
                        chunk_idx += 1
                    continue

                if len(current_chunk_text) + len(block) + 2 <= self.chunk_size:
                    current_chunk_text += ("\n\n" + block if current_chunk_text else block)
                else:
                    chunks.append(self._create_chunk_object(
                        chunk_id=f"{filename}_p{page_num}_c{chunk_idx}",
                        text=current_chunk_text.strip(),
                        filename=filename,
                        page_num=page_num,
                        header_context=current_section_header
                    ))
                    chunk_idx += 1
                    
                    # Create overlap from end of previous chunk
                    overlap_text = current_chunk_text[-self.chunk_overlap:] if len(current_chunk_text) > self.chunk_overlap else ""
                    current_chunk_text = overlap_text + "\n\n" + block if overlap_text else block

            if current_chunk_text.strip():
                chunks.append(self._create_chunk_object(
                    chunk_id=f"{filename}_p{page_num}_c{chunk_idx}",
                    text=current_chunk_text.strip(),
                    filename=filename,
                    page_num=page_num,
                    header_context=current_section_header
                ))
                chunk_idx += 1

        return chunks

    def _split_into_structural_blocks(self, text: str) -> List[str]:
        """Separates text by paragraph breaks or markdown table boundaries."""
        raw_blocks = text.split("\n\n")
        blocks = []
        in_table = False
        table_buffer = []

        for b in raw_blocks:
            lines = b.strip().split("\n")
            # If paragraph contains table lines
            if any("|" in line for line in lines):
                if not in_table:
                    in_table = True
                table_buffer.append(b)
            else:
                if in_table:
                    blocks.append("\n\n".join(table_buffer))
                    table_buffer = []
                    in_table = False
                blocks.append(b)

        if table_buffer:
            blocks.append("\n\n".join(table_buffer))

        return [b.strip() for b in blocks if b.strip()]

    def _split_large_block(self, text: str) -> List[str]:
        """Splits a large paragraph by sentences or line breaks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sub_chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= self.chunk_size:
                current += (" " + s if current else s)
            else:
                if current:
                    sub_chunks.append(current)
                current = s
        if current:
            sub_chunks.append(current)
        return sub_chunks

    def _create_chunk_object(self, chunk_id: str, text: str, filename: str, page_num: int, header_context: str) -> Dict[str, Any]:
        """Formats chunk dict with numerical metadata count."""
        has_table = "|" in text or "---" in text
        numbers = re.findall(r'\b\$?\d+(?:\.\d+)?%?\b', text)
        
        # Enforce header context prepend if not present
        full_context_text = f"[{header_context}]\n{text}"

        return {
            "chunk_id": chunk_id,
            "text": full_context_text,
            "raw_body": text,
            "filename": filename,
            "page_number": page_num,
            "has_table": has_table,
            "numeric_count": len(numbers),
            "header_context": header_context
        }
