import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model ({model_name}) on {self.device}...")
        
        # We use a specific revision (refs/pr/66) that contains the safetensors format 
        # to bypass the PyTorch 2.6 security restriction on .bin files
        revision = "refs/pr/66" if model_name == "openai/clip-vit-base-patch32" else "main"
        
        self.model = CLIPModel.from_pretrained(model_name, revision=revision).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_image_embeddings(self, images):
        """
        Takes a list of PIL Images, returns normalized numpy embeddings.
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        
        if not isinstance(image_features, torch.Tensor):
            if hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            else:
                image_features = image_features[0]
                
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    @torch.no_grad()
    def get_text_embeddings(self, texts):
        """
        Takes a list of strings, returns normalized numpy embeddings.
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, "text_embeds"):
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            else:
                text_features = text_features[0]
                
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()
