
import torch
import xworldmodels.nn as xnn


# A Vision Transformer
vit = xnn.VisionTransformer().eval().cuda()

# Random image for test
# B=1, C=3, H=224, W=224
x = torch.randn(1, 3, 224, 224).cuda()

# Forward
x = vit(x)
