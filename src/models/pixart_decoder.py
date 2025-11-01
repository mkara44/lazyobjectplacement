import torch
import torch.nn as nn 

from .utils.pixart_transformer_2d import PixArtTransformer2DModel
from .utils.helper import get_hole


class PixArtDecoder(nn.Module):
    def __init__(self, 
                 hidden_size = 1152,
                 object_dim = 384,
                 pixart_pretrained_model="PixArt-alpha/PixArt-XL-2-512x512"):
        super().__init__()

        self.object_projection = nn.Linear(object_dim, hidden_size)
        self.context_projection = nn.Linear(hidden_size * 2, hidden_size)

        self.pixart_transformer = PixArtTransformer2DModel.from_pretrained(
            pixart_pretrained_model,
            subfolder="transformer",
            use_safetensors=True,
        )

    def patchify(self, x: torch.Tensor):
        x, height, width = self.pixart_transformer.patchify(x)
        return x, height, width

    def forward(
            self, 
            x: torch.Tensor, 
            timestep: torch.Tensor, 
            t_hole: torch.Tensor, 
            mask: torch.Tensor,
            object_features: torch.Tensor,
            padded: torch.Tensor = False,
        ):
        x_all, height, width = self.patchify(x)
        x_hole, bool_mask = get_hole(x_all, mask, height * width)

        # Hidden States
        hidden_states = self.context_projection(torch.cat([x_hole, t_hole], dim=-1))

        # Object Embeddings
        object_features = self.object_projection(object_features).unsqueeze(1)

        out = self.pixart_transformer(
            hidden_states=hidden_states,
            mask=mask,
            timestep=timestep,
            encoder_hidden_states=object_features,
        )[0]

        if not padded:
            out_full = torch.zeros(out.shape[0], bool_mask.shape[1], out.shape[2], device=out.device)
            out_full[bool_mask] = out.reshape(-1, out.shape[2])
        else:
            out_full = out

        out_full = self.pixart_transformer.unpatchify(out_full, height, width)
        out_full = out_full.chunk(2, dim=1)[0]
        return out_full