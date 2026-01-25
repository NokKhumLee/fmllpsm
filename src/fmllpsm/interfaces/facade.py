import torch
import torch.nn as nn
from fmllpsm.infrastructure.extractors.dino import DINOv1Extractor
from fmllpsm.infrastructure.metrics.learned import LearnedMetric
from fmllpsm.application.services import QualityService

class FMLLPSM(nn.Module):
    """
    High-level facade for the Foundational Model Low-Level Perceptual Similarity Metric.
    """
    def __init__(self, model_type: str = "DINOv1", device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        
        if model_type == "DINOv1":
            extractor = DINOv1Extractor().to(self.device)
            # DINOv1 base has 768 channels
            channels = [768] * len(extractor.layer_indices)
            metric = LearnedMetric(channels).to(self.device)
            
            # Initialize metric with "unit" weights for stable start
            for m in metric.lin_layers:
                if isinstance(m[1], nn.Conv1d):
                    nn.init.constant_(m[1].weight, 1.0 / len(extractor.layer_indices))
            
            self.service = QualityService(extractor, metric, name="DINOv1-LPIPS")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, ref: torch.Tensor, dis: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity score between reference and distorted images.
        """
        # Ensure tensors are on the correct device
        ref = ref.to(self.device)
        dis = dis.to(self.device)
        
        # Use autocast for performance
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type):
            score = self.service.evaluate(ref, dis)
            
        # Return as a scalar if batch size is 1, else return (N,)
        res = score.value
        if res.shape[0] == 1:
            return res.squeeze()
        return res

    def __call__(self, ref: torch.Tensor, dis: torch.Tensor) -> torch.Tensor:
        return self.forward(ref, dis)
