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
            image_total_item_score = defaultdict(float)
            # Track the geometric product of scores. Initialize to 1.0.
            image_item_product = defaultdict(lambda: 1.0)
            # Track how many items this image was actually found in
            image_item_counts = defaultdict(int)
            
            for item in items:
                item_emb = self.extractor.get_text_embeddings([item])
                # Fetch a large number of crops to prevent false negatives
                item_results = self.item_store.search(item_emb, k=1000)
                
                max_item_score_per_img = defaultdict(float)
                for res in item_results:
                    img_id = res['meta']['image_id']
                    if res['score'] > max_item_score_per_img[img_id]:
                        max_item_score_per_img[img_id] = res['score']
                        
                for img_id, score in max_item_score_per_img.items():
                    # HARD THRESHOLD: If cosine similarity is below 0.16 on the 768-dim model, assume False Positive
                    if score < 0.16:
                        score = 0.01
                        
                    # Use a baseline of 0.01 to prevent multiplying by negative or zero
                    safe_score = max(0.01, score)
                    image_total_item_score[img_id] += safe_score
                    image_item_product[img_id] *= safe_score
                    
                    if safe_score > 0.01:
                        image_item_counts[img_id] += 1
                    
            # Apply Geometric Penalty
            for img_id in list(image_total_item_score.keys()):
                # If the image was missing entirely from an item's Top K, penalize the product heavily
                missing_items = len(items) - image_item_counts[img_id]
                final_product = image_item_product[img_id] * (0.01 ** missing_items)
                
                # We blend the additive score (for stability) with the geometric product (for intersection)
                final_item_score = (image_total_item_score[img_id] / len(items)) * (final_product ** (1.0 / len(items)))
                
                image_scores[img_id] += (1.0 - actual_alpha) * final_item_score
                    
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
