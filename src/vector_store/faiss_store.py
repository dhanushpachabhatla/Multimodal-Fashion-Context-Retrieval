import faiss
import numpy as np
import json
import os

class FAISSStore:
    def __init__(self, index_path=None, meta_path=None, dim=512):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Inner Product (Cosine Similarity if normalized)
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = {} # map index_id to dict
            self.current_id = 0

    def add_vectors(self, vectors, meta_list):
        """
        vectors: np.array of shape (N, dim)
        meta_list: list of dicts of length N
        """
        if len(vectors) == 0:
            return
            
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
        for meta in meta_list:
            self.metadata[str(self.current_id)] = meta
            self.current_id += 1

    def search(self, query_vector, k=10):
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "score": float(distances[0][i]),
                    "meta": self.metadata[str(idx)]
                })
        return results

    def save(self, index_path, meta_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
