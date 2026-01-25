import torch
import torch.nn as nn
from typing import List
from fmllpsm.domain.interfaces import SimilarityMetric

class LearnedMetric(nn.Module):
    """
    LPIPS-style learned distance. Calculates:
    sum_l ( weight_l * mean_spatial ( (ref_l - dis_l)^2 ) )
    """
    def __init__(self, channels_list: List[int]):
        super().__init__()
        # Linear layers to map features to a scalar relevance score per spatial location
        self.lin_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(),
                nn.Conv1d(c, 1, kernel_size=1, stride=1, padding=0, bias=False) 
            ) for c in channels_list
        ])
        
    def _normalize_tensor(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + eps)
        return x / norm_factor

    def forward(self, ref_feats: List[torch.Tensor], dis_feats: List[torch.Tensor]) -> torch.Tensor:
        total_dist = 0.0
        
        for i, (ref, dis) in enumerate(zip(ref_feats, dis_feats)):
            # Normalize across the feature dimension (D)
            ref = self._normalize_tensor(ref)
            dis = self._normalize_tensor(dis)
            
            diff = (ref - dis)**2 # (N, T, D)
            
            # Application of learned weights. 
            # diff is (N, T, D), transpose to (N, D, T) for Conv1d
            weighted_diff = self.lin_layers[i](diff.transpose(1, 2)) # (N, 1, T)
            # Mean over tokens (T)
            total_dist += weighted_diff.mean(dim=[1, 2])
            
        return total_dist

    def __call__(self, ref_feats: List[torch.Tensor], dis_feats: List[torch.Tensor]) -> torch.Tensor:
        return self.forward(ref_feats, dis_feats)
