import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path, chunks_path, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the retriever with FAISS index and text chunks.
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        # Load corresponding text chunks
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        # Load embedding model
        self.embedder = SentenceTransformer(model_name)

    def search(self, query, top_k=3):
        """
        Perform semantic search over the vector DB.
        Returns top_k most relevant text chunks.
        """
        # Generate embedding for query
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        # Search FAISS index
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        # Retrieve chunks
        results = [self.chunks[i] for i in indices[0]]
        return results
