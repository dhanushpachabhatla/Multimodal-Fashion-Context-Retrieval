import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import os

class VLMReranker:
    def __init__(self, model_name="Salesforce/blip-vqa-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading VLM Reranker ({model_name}) on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def filter_results(self, raw_query, search_results, dataset_dir, base_dir):
        """
        Takes the FAISS search results and filters them using BLIP VQA.
        search_results: list of dicts like {'image_id': '...', 'file_name': '...', 'score': ...}
        """
        filtered_results = []
        
        # Break complex queries into multiple questions to prevent BLIP hallucinations
        questions = []
        if " and " in raw_query.lower():
            parts = raw_query.lower().replace("a person with", "").split(" and ")
            for p in parts:
                questions.append(f"Question: Is there a {p.strip()} in this picture? Answer:")
        else:
            questions.append(f"Question: Is there {raw_query} in this picture? Answer:")
            
        for res in search_results:
            file_name = res.get('file_name', '')
            
            # Determine correct image path
            if file_name.startswith('red_scarf'):
                # Custom image on desktop
                img_path = os.path.join(base_dir, file_name)
            else:
                img_path = os.path.join(dataset_dir, 'test', file_name)
                
            if not os.path.exists(img_path):
                continue
                
            try:
                image = Image.open(img_path).convert('RGB')
                
                passed_all = True
                debug_answers = []
                
                for question in questions:
                    inputs = self.processor(image, question, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(**inputs, max_new_tokens=5)
                    answer = self.processor.decode(out[0], skip_special_tokens=True).lower().strip()
                    debug_answers.append(f"Q: '{question}' -> {answer}")
                    
                    if "no" in answer:
                        passed_all = False
                        break # Optimization: stop asking if one condition fails
                
                print(f"  [VLM Debug] {file_name}: {debug_answers}")
                
                # Only keep if it passed every single question
                if passed_all:
                    res['vlm_answer'] = str(debug_answers)
                    filtered_results.append(res)
            except Exception as e:
                print(f"Error processing {file_name} with VLM: {e}")
                
        return filtered_results
