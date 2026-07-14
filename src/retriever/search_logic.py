import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)

from src.indexer.feature_extraction import CLIPExtractor
from src.vector_store.faiss_store import FAISSStore
from src.retriever.query_parser import QueryParser
from collections import defaultdict

class RetrievalSystem:
    def __init__(self, indices_dir):
        print("Loading Retrieval System...")
        self.extractor = CLIPExtractor()
        self.parser = QueryParser()
        
        self.context_store = FAISSStore(
            index_path=os.path.join(indices_dir, "context.index"),
            meta_path=os.path.join(indices_dir, "context_meta.json")
        )
        
        self.item_store = FAISSStore(
            index_path=os.path.join(indices_dir, "item.index"),
            meta_path=os.path.join(indices_dir, "item_meta.json")
        )

    def search(self, query, top_k=5, alpha=0.5):
        """
        Retrieves top_k images based on the query.
        alpha: weight for context score vs item score (0.0 = only items, 1.0 = only context)
        """
        parsed = self.parser.parse_query(query)
        print(f"Parsed Query: {parsed}")
        
        image_scores = defaultdict(float)
        
        context_text = parsed.get("context", "")
        items = parsed.get("items", [])
        
        # Dynamically adjust weights if context or items are missing
        actual_alpha = alpha
        if not context_text and items:
            actual_alpha = 0.0  # Weight items fully
        elif context_text and not items:
            actual_alpha = 1.0  # Weight context fully
        
        # 1. Search Context
        if context_text:
            context_emb = self.extractor.get_text_embeddings([context_text])
            context_results = self.context_store.search(context_emb, k=top_k * 5) # Fetch more for aggregation
            
            for res in context_results:
                img_id = res['meta']['image_id']
                image_scores[img_id] += actual_alpha * res['score']
                
        # 2. Search Items
        if items:
            for item in items:
                item_emb = self.extractor.get_text_embeddings([item])
                item_results = self.item_store.search(item_emb, k=top_k * 10)
                
                # To prevent a single image with many bounding boxes dominating, 
                # we track the max score per item for each image
                max_item_score_per_img = defaultdict(float)
                for res in item_results:
                    img_id = res['meta']['image_id']
                    if res['score'] > max_item_score_per_img[img_id]:
                        max_item_score_per_img[img_id] = res['score']
                        
                for img_id, score in max_item_score_per_img.items():
                    image_scores[img_id] += ((1.0 - actual_alpha) / len(items)) * score
                    
        # Sort and return top K
        sorted_results = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format output
        final_results = []
        for img_id, score in sorted_results[:top_k]:
            # We just need the file_name, can get it from context store metadata
            # It's slightly inefficient to loop but ok for MVP
            file_name = None
            for meta in self.context_store.metadata.values():
                if meta['image_id'] == img_id:
                    file_name = meta['file_name']
                    break
            final_results.append({"image_id": img_id, "file_name": file_name, "score": score})
            
        return final_results
