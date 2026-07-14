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
        
    test_cases = [
        # ==========================================
        # PREVIOUS GUARANTEED DATASET TEST CASES
        # ==========================================
        # Easy / Single Item (Guaranteed to exist)
        {"query": "A person wearing a jacket", "targets": [["jacket"]]},
        {"query": "Someone with a skirt", "targets": [["skirt"]]},
        {"query": "Wearing glasses", "targets": [["glasses", "eyewear"]]},
        {"query": "A person in a jumpsuit", "targets": [["jumpsuit"]]},
        
        # Medium / 2 Items (Guaranteed to exist)
        {"query": "A person wearing a dress and a cape", "targets": [["dress"], ["cape"]]},
        {"query": "Someone wearing a jacket and a dress", "targets": [["jacket"], ["dress"]]},
        {"query": "A shirt and a skirt", "targets": [["shirt, blouse", "shirt"], ["skirt"]]},
        {"query": "A top and shorts", "targets": [["top, t-shirt, sweatshirt", "top"], ["shorts"]]},
        
        # Hard / 3+ Items (Guaranteed to exist)
        {"query": "A person with a scarf, a blouse, and a skirt carrying a bag", "targets": [["scarf"], ["shirt, blouse", "shirt", "blouse"], ["skirt"], ["bag, wallet", "bag"]]},
        {"query": "Someone wearing glasses, a hat, and a sweatshirt carrying a bag", "targets": [["glasses"], ["hat"], ["top, t-shirt, sweatshirt", "sweatshirt", "top"], ["bag, wallet", "bag"]]},
        {"query": "A person wearing a jacket over a shirt with glasses", "targets": [["jacket"], ["shirt, blouse", "shirt"], ["glasses"]]},
        {"query": "Someone wearing a coat, a belt, and a skirt", "targets": [["coat"], ["belt"], ["skirt"]]},
        {"query": "A person wearing a hat, a belt, and a skirt", "targets": [["hat"], ["belt"], ["skirt"]]},
        {"query": "A dress with a headband and ruffle details", "targets": [["dress"], ["headband, head covering, hair accessory", "headband"], ["ruffle"]]},

        # ==========================================
        # OFFICIAL ASSIGNMENT QUERIES (Dataset Mismatch Expected)
        # ==========================================
        {"query": "A person in a bright yellow raincoat.", "targets": [["coat", "jacket"]]},
        {"query": "Professional business attire inside a modern office.", "targets": [["jacket", "suit", "blazer", "tie", "pants"]]},
        {"query": "Someone wearing a blue shirt sitting on a park bench.", "targets": [["shirt, blouse", "shirt"]]},
        {"query": "Casual weekend outfit for a city walk.", "targets": [["jeans", "pants", "top, t-shirt, sweatshirt", "sneaker", "shoe"]]},
        {"query": "A guy with blue tie and a jacket in a formal setting.", "targets": [["tie", "necktie"], ["shirt, jacket"]]}
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
