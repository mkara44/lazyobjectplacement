import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


def get_hole(tokens: torch.Tensor, mask: torch.Tensor, num_patches: int) -> torch.Tensor:
    B, N, D = tokens.shape

    context_mask = F.max_pool2d(mask, kernel_size=2, stride=2)
    patch_mask = context_mask.flatten(start_dim=2).squeeze(1)
    bool_mask = patch_mask > 0.5

    hole_tokens_list = []
    for b in range(B):
        sel = bool_mask[b]
        sel_tokens = tokens[b][sel]
        n_hole = sel_tokens.shape[0]
        
        if n_hole == 0:
            pad = torch.zeros(num_patches, D, device=tokens.device, dtype=tokens.dtype)
            hole_tokens = pad.unsqueeze(0)
        else:
            if n_hole < num_patches:
                hole_tokens = torch.zeros(num_patches, D, device=tokens.device, dtype=tokens.dtype)
                hole_tokens[sel] = sel_tokens
                hole_tokens = hole_tokens.unsqueeze(0)
            else:
                hole_tokens = sel_tokens[:num_patches].to(dtype=tokens.dtype).unsqueeze(0)
        hole_tokens_list.append(hole_tokens)
    hole_tokens = torch.cat(hole_tokens_list, dim=0)
    return hole_tokens, bool_mask

class Dinov2FeatureExtractor(nn.Module):
    def __init__(self, model_name='facebook/dinov2-small'):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)

    def forward(self, x):
        outputs = self.model(x)
        return outputs.last_hidden_state

class LearnedMaskDownsampler(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, intermediate_channels=16, input_size=512, target_size=64):
        super().__init__()
        assert input_size % target_size == 0, "input_size must be divisible by target_size"
        
        num_downsamples = int(math.log2(input_size // target_size))

        layers = []
        in_ch = in_channels
        out_ch = intermediate_channels
        
        for i in range(num_downsamples - 1):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.Sigmoid())
            in_ch = out_ch
            out_ch *= 2
        
        layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=2, padding=1))        
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, mask):
        return self.net(mask)
