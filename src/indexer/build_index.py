import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)

from src.indexer.feature_extraction import CLIPExtractor
from src.vector_store.faiss_store import FAISSStore

def build_indices(dataset_dir, metadata_path, indices_dir, batch_size=32):
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    extractor = CLIPExtractor()
    
    # We use 768 for standard CLIP ViT-L/14
    context_store = FAISSStore(dim=768)
    item_store = FAISSStore(dim=768)
    
    test_dir = os.path.join(dataset_dir, 'test')
    
    img_ids = list(metadata.keys())
    
    context_batch_imgs = []
    context_batch_meta = []
    
    item_batch_imgs = []
    item_batch_meta = []
    
    def process_context_batch(imgs, metas):
        if not imgs: return
        emb = extractor.get_image_embeddings(imgs)
        context_store.add_vectors(emb, metas)
        
    def process_item_batch(imgs, metas):
        if not imgs: return
        emb = extractor.get_image_embeddings(imgs)
        item_store.add_vectors(emb, metas)
        
    print(f"Indexing {len(img_ids)} images and their crops...")
    
    for img_id in tqdm(img_ids):
        img_info = metadata[img_id]
        img_path = os.path.join(test_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue
            
        # Add to context batch
        context_batch_imgs.append(image)
        context_batch_meta.append({
            "image_id": img_id,
            "file_name": img_info['file_name']
        })
        
        # Process crops
        for crop_info in img_info['crops']:
            # bbox is [x, y, w, h]
            x, y, w, h = crop_info['bbox']
            # PIL crop uses (left, upper, right, lower)
            crop_box = (x, y, x + w, y + h)
            try:
                cropped_img = image.crop(crop_box)
                # To save memory and time, we can resize if it's too big, but CLIP handles it
                item_batch_imgs.append(cropped_img)
                item_batch_meta.append({
                    "image_id": img_id,
                    "file_name": img_info['file_name'],
                    "category": crop_info['category'],
                    "attributes": crop_info['attributes']
                })
            except Exception as e:
                pass
                
        # Flush batches
        if len(context_batch_imgs) >= batch_size:
            process_context_batch(context_batch_imgs, context_batch_meta)
            context_batch_imgs, context_batch_meta = [], []
            
        if len(item_batch_imgs) >= batch_size:
            process_item_batch(item_batch_imgs, item_batch_meta)
            item_batch_imgs, item_batch_meta = [], []
            
    # Process remaining
    process_context_batch(context_batch_imgs, context_batch_meta)
    process_item_batch(item_batch_imgs, item_batch_meta)
    
    # Save indices
    print("Saving indices...")
    os.makedirs(indices_dir, exist_ok=True)
    context_store.save(os.path.join(indices_dir, "context.index"), os.path.join(indices_dir, "context_meta.json"))
    item_store.save(os.path.join(indices_dir, "item.index"), os.path.join(indices_dir, "item_meta.json"))
    print("Indexing complete.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    dataset_dir = os.path.join(base_dir, 'initial_dataset')
    
    metadata_path = os.path.join(project_dir, 'data', 'metadata.json')
    indices_dir = os.path.join(project_dir, 'indices')
    
    build_indices(dataset_dir, metadata_path, indices_dir)
