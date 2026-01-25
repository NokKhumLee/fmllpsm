# fmllpsm: Foundational Model Low-Level Perceptual Similarity Metric

[![PyPI version](https://img.shields.io/pypi/v/fmllpsm.svg)](https://pypi.org/project/fmllpsm/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

**fmllpsm** is a Python library that provides a high-level interface for calculating image similarity using deep features from foundational models (e.g., DINOv1). It follows a modular architecture based on **Domain-Driven Design (DDD)** and **Clean Architecture**, making it easy for researchers to extend with new backbones or statistical metrics.

This project is a refactored and production-ready implementation of the research conducted in the [ZS-IQA](https://github.com/abhijay9/ZS-IQA) repository.

---

## 🚀 Quick Start

### Installation

```bash
pip install fmllpsm
```

*Note: Ensure you have PyTorch and Transformers installed. See [pyproject.toml](pyproject.toml) for version requirements.*

### Basic Usage

```python
import torch
from fmllpsm import FMLLPSM

# 1. Initialize the metric (DINOv1 is supported natively)
# Device is automatically handled, but you can specify "cuda" or "cpu"
fllpsm = FMLLPSM("DINOv1", device="cuda")

# 2. Prepare your images (Tensors: N x C x H x W, normalized to [-1, 1] or [0, 1])
ref_img = torch.randn(1, 3, 224, 224).to("cuda")
gen_img = torch.randn(1, 3, 224, 224).to("cuda")

# 3. Compute the similarity loss
loss = fllpsm(ref_img, gen_img)

print(f"Similarity Loss: {loss.item():.6f}")
```

---

## 🏗️ Architecture

The package is built using **Domain-Driven Design (DDD)** principles to ensure high maintainability and extensibility for researchers:

- **Domain Layer**: Defines pure interfaces for `FeatureExtractor` and `SimilarityMetric`.
- **Infrastructure Layer**: Contains concrete implementations like `DINOv1Extractor` and `LearnedMetric` (LPIPS-style).
- **Application Layer**: Orchestrates the evaluation logic through a `QualityService`.
- **Interfaces Layer**: Provides a simplified `FMLLPSM` facade for common use cases.

```mermaid
graph TD
    User["Researcher/User"] --> Facade["FMLLPSM Facade"]
    Facade --> App["Application Service"]
    App --> Domain["Domain Interfaces"]
    Domain -.-> Infra["Infrastructure Impls (DINO, LPIPS)"]
```

---

## 🔬 Scientific Background

The core metric uses a **Learned LPIPS-style distance** calculated over DINOv1 ViT-Base features. The formula involves:

1.  **Normalization** of features on a unit hypersphere.
2.  **Weighted Squared Difference** across multiple layers (2, 5, 8, 11).
3.  **Spatial Averaging** to produce a robust perceptual score.

This approach allows for zero-shot image quality assessment that correlates highly with human subjective scores (MOS) across standard datasets like LIVE, TID2013, and PIPAL.

---

## 📝 Roadmap

- [x] DINOv1 Backbone Support
- [x] LPIPS-style Learned Metric
- [ ] Statistical Proof Metrics (KL Divergence, Wasserstein Distance)
- [ ] Support for CLIP and ConvNeXt backbones
- [ ] Pre-trained weights for specific downstream tasks

---

## ⚖️ License

Distributed under the **BSD 2-Clause License**. See `LICENSE` for more information.

---

## 🤝 Citation

If you use this work in your research, please cite the original ZS-IQA paper:

```bibtex
@article{fmllpsm2024,
  title={Foundational Model Low-Level Perceptual Similarity Metrics},
  author={Chogerlate and Ghildyal, Abhijay},
  journal={TBD},
  year={2026}
}
```
