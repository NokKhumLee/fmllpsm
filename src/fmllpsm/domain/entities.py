from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class QualityScore:
    """
    Entity representing the calculated quality score.
    """
    value: torch.Tensor
    metric_name: str
    
    def __float__(self) -> float:
        return float(self.value.item())
    
    def __repr__(self) -> str:
        return f"QualityScore(value={self.value.item():.4f}, metric='{self.metric_name}')"
