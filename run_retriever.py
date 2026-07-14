import os
import sys
import logging
from tqdm import tqdm

from src.retriever.search_logic import RetrievalSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Initializing Retrieval Pipeline...")
    indices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'indices'))
    
    if not os.path.exists(indices_dir):
        logger.error("Indices not found. Please run run_indexer.py first.")
        sys.exit(1)
        
    logger.info("Loading FAISS Indices and CLIP Extractor...")
    retriever = RetrievalSystem(indices_dir)
    
    queries = [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting.",
        "A red shirt and blue pants.",
        "A black dress with white shoes in an outdoor setting."
    ]
    
    logger.info(f"Loaded {len(queries)} evaluation queries. Starting semantic search...")
    print("\n" + "="*50)
    print("--- EVALUATION RESULTS ---")
    print("="*50 + "\n")
    
    for query in tqdm(queries, desc="Processing Queries", unit="query"):
        logger.info(f"Querying: '{query}'")
        results = retriever.search(query, top_k=3, alpha=0.5)
        
        print(f"\n[QUERY]: {query}")
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['file_name']} (Score: {res['score']:.4f})")
        print("-" * 50)
        
    logger.info("Retrieval Evaluation Completed!")
