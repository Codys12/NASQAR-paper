import torch
from torch import nn
import torch.nn.functional as F
from .norm import RMSNorm


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Activation quantization with Straight-Through Estimator."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Ternary weight quantization with Straight-Through Estimator."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinearWithRMS(nn.Linear):
    """Linear layer preceded by RMS norm and ternary quantization."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.rms = RMSNorm(in_features, weight_scaling=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS norm and STE-based ternary quantization."""
        x_norm = self.rms(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return F.linear(x_quant, w_quant, self.bias)

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
