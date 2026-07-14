import json
import os
from collections import defaultdict
from pathlib import Path

def setup_metadata(dataset_dir, annotation_path, output_path):
    print("Loading annotations...")
    with open(annotation_path, 'r') as f:
        data = json.load(f)
        
    categories = {c['id']: c['name'] for c in data['categories']}
    attributes = {a['id']: a['name'] for a in data['attributes']}
    
    # Get list of files in the test directory
    test_dir = os.path.join(dataset_dir, 'test')
    available_files = set(os.listdir(test_dir))
    
    print(f"Found {len(available_files)} images in the test directory.")
    
    # Map image id to file name
    img_id_to_file = {}
    for img in data['images']:
        if img['file_name'] in available_files:
            img_id_to_file[img['id']] = img['file_name']
            
    print(f"Matched {len(img_id_to_file)} images with annotations.")
    
    # Map annotations to images
    img_annotations = defaultdict(list)
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id in img_id_to_file:
            # Gather category and attributes
            cat_name = categories.get(ann['category_id'], "unknown")
            attr_names = [attributes.get(attr_id, "unknown") for attr_id in ann.get('attribute_ids', [])]
            
            img_annotations[img_id].append({
                'annotation_id': ann['id'],
                'bbox': ann['bbox'], # [x, y, width, height]
                'category': cat_name,
                'attributes': attr_names
            })
            
    # Compile final metadata
    metadata = {}
    for img_id, file_name in img_id_to_file.items():
        metadata[str(img_id)] = {
            'file_name': file_name,
            'crops': img_annotations[img_id]
        }
        
    print(f"Writing metadata for {len(metadata)} images to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    dataset_dir = os.path.join(base_dir, 'initial_dataset')
    annotation_path = os.path.join(base_dir, 'instances_attributes_val2020.json')
    
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    output_path = os.path.join(project_dir, 'data', 'metadata.json')
    
    setup_metadata(dataset_dir, annotation_path, output_path)
