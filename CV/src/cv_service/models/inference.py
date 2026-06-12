import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging
from cv_service.models.architecture import DamageClassifier

logger = logging.getLogger(__name__)

class DamageInference:
    """
    Wrapper for ResNet50 damage classification model inference.
    """
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading model from {model_path} to {self.device}")
        
        # Load architecture
        self.model = DamageClassifier(num_classes=4, pretrained=False)
        
        # Load state dict
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class mapping for internal use (aligning with XBDDataset)
        self.classes = ['no_damage', 'minor_damage', 'major_damage', 'destroyed']

    def predict(self, image: Image.Image) -> dict:
        """
        Runs inference on a PIL image.
        """
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Extract results
        prob_list = probabilities.cpu().numpy().tolist()
        class_probs = {self.classes[i]: prob_list[i] for i in range(len(self.classes))}
        
        # Predicted class
        pred_idx = torch.argmax(probabilities).item()
        damage_label = self.classes[pred_idx]
        
        # Continuous damage score (weighted average of classes)
        # Weights: no=0.0, minor=0.33, major=0.66, destroyed=1.0
        weights = np.array([0.0, 0.33, 0.66, 1.0])
        damage_score = np.dot(prob_list, weights)
        
        return {
            "damage_label": damage_label,
            "damage_score": float(damage_score),
            "model_probability": float(probabilities[pred_idx].item()),
            "class_probabilities": class_probs
        }
