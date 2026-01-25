import torch
import torch.nn as nn
from typing import List, Optional
from transformers import ViTModel
from fmllpsm.domain.interfaces import FeatureExtractor

class DINOv1Extractor(nn.Module):
    """
    Extracts features from DINOv1 (ViT-base) using the Transformers library.
    """
    def __init__(self, model_name: str = 'facebook/dino-vitb16', layer_indices: Optional[List[int]] = None):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
        # Default to capturing features every 3 layers if not specified
        self.layer_indices = layer_indices or [2, 5, 8, 11]
        
        # Freeze the backbone for efficiency during research/inference
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (N, C, H, W)
        outputs = self.model(x, output_hidden_states=True)
        # hidden_states: tuple of (initial_embeddings, layer_1, ..., layer_n)
        hidden_states = outputs.hidden_states
        
        # Extract requested layers (hidden_states[idx+1] corresponds to layer idx output)
        features = []
        for idx in self.layer_indices:
            # ViT hidden states are (N, T, D). 
            # We treat T as flattened spatial dims (H*W) + 1 (CLS)
            feat = hidden_states[idx + 1]
            features.append(feat)
            
        return features

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward(x)
