import torch
import torch.nn as nn
import torch.nn.functional as F
from fmllpsm.infrastructure.extractors.dino import DINOv1Extractor
from fmllpsm.infrastructure.metrics.learned import LearnedMetric
from fmllpsm.application.services import QualityService


class FMLLPSM(nn.Module):
    """
    High-level facade for the Foundational Model Low-Level Perceptual Similarity Metric.
    """

    def __init__(self, model_type: str = "DINOv1", device: str | None = None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_type == "DINOv1":
            self.extractor = DINOv1Extractor().to(self.device)
            # DINOv1 base has 768 channels
            channels = [768] * len(self.extractor.layer_indices)
            self.metric = LearnedMetric(channels).to(self.device)

            # Initialize metric with "unit" weights for stable start
            for m in self.metric.lin_layers:
                if isinstance(m[1], nn.Conv1d):
                    nn.init.constant_(
                        m[1].weight, 1.0 / len(self.extractor.layer_indices)
                    )

            self.service = QualityService(
                self.extractor, self.metric, name="DINOv1-LPIPS"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def compile(self, **kwargs):
        """
        Optimizes the model using torch.compile (requires PyTorch 2.0+).
        Recommended for training loops to improve performance.
        """
        self.extractor = torch.compile(self.extractor, **kwargs)
        self.metric = torch.compile(self.metric, **kwargs)
        return self

    def forward(self, ref: torch.Tensor, dis: torch.Tensor) -> torch.Tensor:
        """
        Computes the similarity score between reference and distorted images.
        """
        # Ensure inputs match the current device of the model
        model_device = next(self.parameters()).device
        ref = ref.to(model_device)
        dis = dis.to(model_device)

        # Use autocast for performance
        device_type = "cuda" if model_device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=device_type):
            # Preprocess: resize to the expected resolution of the backbone (224, 224)
            # This ensures stability and significantly improves training speed
            if ref.shape[-2:] != (224, 224):
                ref = F.interpolate(
                    ref, size=(224, 224), mode="bilinear", align_corners=False
                )
            if dis.shape[-2:] != (224, 224):
                dis = F.interpolate(
                    dis, size=(224, 224), mode="bilinear", align_corners=False
                )

            score = self.service.evaluate(ref, dis)

        # Return as a scalar if batch size is 1, else return (N,)
        res = score.value
        if res.shape[0] == 1:
            return res.squeeze()
        return res

    def __call__(self, ref: torch.Tensor, dis: torch.Tensor) -> torch.Tensor:
        return self.forward(ref, dis)
