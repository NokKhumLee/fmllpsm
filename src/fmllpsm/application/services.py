import torch
from fmllpsm.domain.interfaces import FeatureExtractor, SimilarityMetric
from fmllpsm.domain.entities import QualityScore

class QualityService:
    """
    Application service that coordinates quality evaluation.
    """
    def __init__(self, extractor: FeatureExtractor, metric: SimilarityMetric, name: str):
        self.extractor = extractor
        self.metric = metric
        self.name = name

    def evaluate(self, ref: torch.Tensor, dis: torch.Tensor) -> QualityScore:
        """
        Evaluates the quality of a distorted image relative to a reference image.
        """
        # Feature extraction
        ref_feats = self.extractor(ref)
        dis_feats = self.extractor(dis)
        
        # Similarity calculation
        score_tensor = self.metric(ref_feats, dis_feats)
        
        return QualityScore(value=score_tensor, metric_name=self.name)
