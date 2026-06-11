import json
import os
import sys

# Ensure the root of the project is in the Python path regardless of where the script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tqdm import tqdm
from src.ingestion.pdf_extractor import PDFExtractor
from src.embeddings.chunker import SemanticChunker
from src.embeddings.embedder import DocumentEmbedder
from src.vectorstore.chroma_store import ChromaManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

class IngestionPipeline:
    """
    Orchestrates: PDF Extraction -> Chunking -> Embedding -> Vector Store Upsert
    """
    def __init__(self, metadata_path: str = None, persist_directory: str = None):
        self.metadata_path = metadata_path or os.path.join(PROJECT_ROOT, "data/raw/metadata.json")
        self.extractor = PDFExtractor()
        self.chunker = SemanticChunker(chunk_size=300, overlap=50)
        self.embedder = DocumentEmbedder()
        self.vector_store = ChromaManager(persist_directory=persist_directory or os.path.join(PROJECT_ROOT, "data/chroma_db"))

    def run(self):
        if not os.path.exists(self.metadata_path):
            print("Metadata file not found. Run arxiv_collector.py first.")
            return

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"Processing {len(metadata)} papers for ingestion...")
        
        all_ids = []
        all_embeddings = []
        all_documents = []
        all_metadatas = []

        # Process each paper
        for paper in tqdm(metadata, desc="Chunking and Embedding"):
            pdf_path = paper.get("pdf_path")
            if not pdf_path:
                continue
                
            if not os.path.isabs(pdf_path):
                pdf_path = os.path.join(PROJECT_ROOT, pdf_path)
                
            if not os.path.exists(pdf_path):
                continue

            text = self.extractor.extract_text(pdf_path)
            if not text.strip():
                continue

            chunks = self.chunker.chunk_text(text)
            if not chunks:
                continue

            # Embed chunks
            embeddings = self.embedder.embed(chunks)

            # Prepare for ChromaDB
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{paper['id']}_chunk_{i}"
                
                chunk_meta = {
                    "source": paper.get("category", "unknown"),
                    "paper_id": paper["id"],
                    "title": paper.get("title", ""),
                    "year": paper.get("year", 0),
                    # Mock citation and recency scores for fusion
                    "citation_score": min(1.0, float(paper.get("year", 2020) - 2000) / 25.0),
                    "recency_score": min(1.0, float(paper.get("year", 2020) - 2015) / 10.0) 
                }

                all_ids.append(chunk_id)
                all_documents.append(chunk)
                all_embeddings.append(embedding.tolist())
                all_metadatas.append(chunk_meta)

        print(f"Generated {len(all_ids)} total chunks. Upserting to ChromaDB...")
        
        # Batch upsert to prevent Chroma limits
        batch_size = 5000
        for i in range(0, len(all_ids), batch_size):
            end = i + batch_size
            self.vector_store.add_chunks(
                ids=all_ids[i:end],
                embeddings=all_embeddings[i:end],
                documents=all_documents[i:end],
                metadatas=all_metadatas[i:end]
            )

        print("Ingestion pipeline complete! Vector store is ready.")

if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
