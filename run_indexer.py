import os
import logging
from src.indexer.data_setup import setup_metadata
from src.indexer.build_index import build_indices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Indexing Pipeline...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(base_dir, 'initial_dataset')
    
    metadata_path = os.path.join(os.path.dirname(__file__), 'data', 'metadata.json')
    indices_dir = os.path.join(os.path.dirname(__file__), 'indices')
    
    # In case data_setup wasn't run separately
    if not os.path.exists(metadata_path):
        logger.info("Metadata not found. Generating metadata from COCO annotations...")
        annotation_path = os.path.join(base_dir, 'instances_attributes_val2020.json')
        setup_metadata(dataset_dir, annotation_path, metadata_path)
    else:
        logger.info(f"Using existing metadata at {metadata_path}")
        
    logger.info("Initializing embedding generation and FAISS index construction...")
    build_indices(dataset_dir, metadata_path, indices_dir)
    logger.info("Indexing Pipeline Completed Successfully!")
