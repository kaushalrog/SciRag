from typing import List

class SemanticChunker:
    """
    Chunks scientific documents into semantically coherent segments.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of roughly `chunk_size` words with `overlap` words.
        """
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.overlap
            if i >= len(words) - self.overlap: # avoid tiny trailing chunks
                break
        return chunks
