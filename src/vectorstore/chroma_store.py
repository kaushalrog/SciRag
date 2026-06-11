import chromadb
from typing import List, Dict

class ChromaManager:
    """
    Manages the ChromaDB collections for the multi-source vector store.
    """
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="scirag_corpus")
        
    def add_chunks(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]):
        """
        Adds document chunks to the collection.
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """
        Queries the collection for the nearest neighbors.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
