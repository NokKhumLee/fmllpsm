import torch
import torchvision.transforms as T
from PIL import Image
from fmllpsm import FMLLPSM

def run_demo():
    # 1. Initialize the metric
    # It will automatically download the DINOv1 weights on the first run
    print("🚀 Initializing FMLLPSM (DINOv1)...")
    model = FMLLPSM("DINOv1", device="cpu") # Use "cuda" if available
    
    # 2. Create dummy data or load real images
    # Convention: Tensors should be [N, 3, H, W] in range [0, 1] or [-1, 1]
    # DINO usually prefers 224x224
    print("🖼️ Preparing dummy images...")
    ref = torch.rand(1, 3, 224, 224)
    dist = ref + 0.1 * torch.randn(1, 3, 224, 224) # Add some noise
    
    # 3. Compute similarity
    print("📏 Calculating perceptual similarity...")
    with torch.no_grad():
        score = model(ref, dist)
    
    print(f"✅ Similarity Loss: {score.item():.6f}")
    print("\nNote: A lower score means higher similarity (closer to the reference).")

if __name__ == "__main__":
    run_demo()
