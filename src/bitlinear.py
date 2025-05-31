import torch
from torch import nn
import torch.nn.functional as F


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinearWithRMS(nn.Linear):
    """Linear layer with RMSNorm and ternary weight/activation quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        # elementwise_affine=False -> no learned weight in the norm
        self.rms = nn.RMSNorm(in_features, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        x_norm = self.rms(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant, self.bias)
        return y

def replace_linear_with_bitlinear(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, BitLinearWithRMS):
            new_layer = BitLinearWithRMS(child.in_features, child.out_features, bias=(child.bias is not None))
            with torch.no_grad():
                new_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_bitlinear(child)
