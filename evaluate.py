import os
import json
from src.retriever.search_logic import RetrievalSystem

def evaluate_retrieval(indices_dir, metadata_path):
    print("Loading Retrieval System for Evaluation...")
    retriever = RetrievalSystem(indices_dir)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    # Create a mapping of image_id -> list of category names present in the image
    image_to_categories = {}
    for img_id, data in metadata.items():
        categories = [crop['category'].lower() for crop in data.get('crops', [])]
        image_to_categories[img_id] = categories

    # Define test cases: Query -> List of required item groups (AND logic). 
    # Inside each group, we use OR logic (e.g. ['backpack', 'bag']).
    test_cases = [
        # Easy / Single Item
        {"query": "A person wearing a jacket", "targets": [["jacket"]]},
        {"query": "Someone with a skirt", "targets": [["skirt"]]},
        {"query": "Wearing sunglasses", "targets": [["sunglasses", "glasses", "eyewear"]]},
        
        # Medium / Multiple Items (Strict AND)
        {"query": "A red tie and a white shirt", "targets": [["tie", "necktie"], ["shirt"]]},
        {"query": "Someone wearing a hat and carrying a purse", "targets": [["hat", "headwear"], ["purse", "bag"]]},
        {"query": "A long coat with a scarf", "targets": [["coat"], ["scarf"]]},
        
        # Hard / Tricky (Implicit or easily confused attributes)
        {"query": "Business attire with a belt and watch", "targets": [["belt"], ["watch"]]},
        {"query": "A woman wearing a necklace and a dress", "targets": [["necklace", "jewelry"], ["dress"]]},
    ]
    
    print("\n--- Starting Metric Evaluation ---")
    
    top_k = 5
    successful_queries = 0
    
    for case in test_cases:
        query = case["query"]
        target_groups = case["targets"]
        
        results = retriever.search(query, top_k=top_k, alpha=0.5)
        
        hit = False
        # We check if ANY of the top K images satisfy ALL the required item groups
        for res in results:
            img_id = str(res["image_id"])
            img_cats = image_to_categories.get(img_id, [])
            
            all_groups_found = True
            for group in target_groups:
                # Check if this specific group (OR logic) is satisfied by the image
                group_satisfied = any(t in cat for t in group for cat in img_cats)
                if not group_satisfied:
                    all_groups_found = False
                    break
                    
            if all_groups_found:
                hit = True
                break
                
        if hit:
            successful_queries += 1
            print(f"[SUCCESS] Query: '{query}' -> Found {target_groups} in Top {top_k}")
        else:
            print(f"[FAILED]  Query: '{query}' -> Did not find {target_groups} in Top {top_k}")
            
    accuracy = (successful_queries / len(test_cases)) * 100
    print("-" * 35)
    print(f"Overall Category Recall@{top_k}: {accuracy:.1f}% ({successful_queries}/{len(test_cases)})")
    
if __name__ == "__main__":
    indices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'indices'))
    metadata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'metadata.json'))
    evaluate_retrieval(indices_dir, metadata_path)
