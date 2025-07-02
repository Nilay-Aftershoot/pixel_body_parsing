import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class RegNetX400MF_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load RegNetX-400MF from timm
        self.backbone = timm.create_model('regnetx_004.pycls_in1k', pretrained=pretrained, features_only=True)
        # This gives us feature maps at different stages

    def forward(self, x):
        feats = self.backbone(x)
        # feats is a list: [C2, C3, C4, C5]
        # Let's match your ResNet API: feat8, feat16, feat32
        # Typically, feats[1], feats[2], feats[3] correspond to 1/8, 1/16, 1/32 resolutions
        feat8 = feats[1]
        feat16 = feats[2]
        feat32 = feats[3]
        return feat8, feat16, feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

if __name__ == "__main__":
    net = RegNetX400MF_Backbone()
    x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].shape)  # feat8
    print(out[1].shape)  # feat16
    print(out[2].shape)  # feat32
    net.get_params()
