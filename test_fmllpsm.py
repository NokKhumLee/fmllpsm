import torch
from fmllpsm import FMLLPSM

def test_facade():
    print("Initializing FMLLPSM('DINOv1')...")
    fllpsm = FMLLPSM("DINOv1")
    
    # Create two dummy images (N, C, H, W)
    # ViT usually expects 224x224
    ref_img = torch.randn(1, 3, 224, 224)
    gen_img = torch.randn(1, 3, 224, 224)
    
    print("Computing loss...")
    loss = fllpsm(ref_img, gen_img)
    
    print(f"Calculated loss: {loss.item():.6f}")
    
    if isinstance(loss, torch.Tensor) and loss.ndim == 0:
        print("Success: Loss is a scalar tensor.")
    else:
        print(f"Failure: Loss shape is {loss.shape}")

if __name__ == "__main__":
    test_facade()
