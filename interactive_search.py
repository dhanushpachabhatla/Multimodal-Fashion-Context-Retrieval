import os
import sys
import logging
import time
from src.retriever.search_logic import RetrievalSystem
from src.retriever.vlm_reranker import VLMReranker

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the Glance Fashion Retrieval Engine!")
    print("="*50)
    print("Loading AI models and FAISS indices into GPU memory...")
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    indices_dir = os.path.join(base_dir, 'indices')
    dataset_dir = os.path.abspath(os.path.join(base_dir, '..', 'initial_dataset'))
    
    if not os.path.exists(indices_dir):
        print("[ERROR] Indices not found. Please run run_indexer.py first.")
        sys.exit(1)
        
    retriever = RetrievalSystem(indices_dir)
    vlm = VLMReranker()
    print("Models loaded successfully!\n")
    
    while True:
        try:
            query = input("\nEnter your search query (or type 'quit' to exit): ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting search engine. Goodbye!")
                break
                
            if not query:
                continue
                
            print(f"Searching for: '{query}'...")
            
            # FAISS Search
            t0 = time.time()
            results = retriever.search(query, top_k=10, alpha=0.5)
            t1 = time.time()
            faiss_latency = t1 - t0
            
            print(f"\n--- Top 10 FAISS Matches (Latency: {faiss_latency:.3f}s) ---")
            for i, res in enumerate(results):
                print(f"  {i+1}. {res['file_name']} (Score: {res['score']:.4f})")
                
            # VLM Reranking
            print("\nApplying VLM Reranker...")
            t2 = time.time()
            filtered_results = vlm.filter_results(query, results, dataset_dir, os.path.dirname(base_dir))
            t3 = time.time()
            vlm_latency = t3 - t2
            
            print(f"\n--- Final Reranked Matches (VLM Latency: {vlm_latency:.3f}s) ---")
            if not filtered_results:
                print("  No matches survived the VLM filter!")
            else:
                for i, res in enumerate(filtered_results):
                    print(f"  {i+1}. {res['file_name']} (Score: {res['score']:.4f})")
            print("-" * 19)
            
        except KeyboardInterrupt:
            print("\nExiting search engine. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
