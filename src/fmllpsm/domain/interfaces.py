from typing import List, Protocol, runtime_checkable
import torch

@runtime_checkable
class FeatureExtractor(Protocol):
    """
    Interface for Feature Extractors.
    """
    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns a list of feature maps from different layers.
        """
        ...

@runtime_checkable
class SimilarityMetric(Protocol):
    """
    Interface for distance calculation methodologies.
    """
    def __call__(self, ref_feats: List[torch.Tensor], dis_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates similarity scores between reference and distorted features.
        """
        ...
