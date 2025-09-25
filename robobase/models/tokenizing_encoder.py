import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

from robobase.models.core import EncoderModule


class TokenizingEncoder(EncoderModule):
    """A ResNet-based encoder that tokenizes image inputs."""

    def __init__(self, d_model: int, frame_stack: int, input_shape: dict):
        super().__init__(input_shape)
        self.d_model = d_model
        self.frame_stack = frame_stack

        # All views are assumed to have the same shape
        example_shape = next(iter(input_shape.values()))
        single_frame_channels = example_shape[0] // frame_stack

        # Use a single shared backbone for efficiency
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.conv1 = nn.Conv2d(
            single_frame_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Projector to map ResNet features (512) to d_model
        self.projector = nn.Conv2d(512, d_model, kernel_size=1)

        self.view_names = sorted(input_shape.keys())

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        all_tokens = []
        batch_size = next(iter(x.values())).shape[0]

        for view_name in self.view_names:
            view_tensor = x[view_name]
            b, cf, h, w = view_tensor.shape
            c = cf // self.frame_stack

            # Reshape to (B * F, C, H, W) to process all frames at once
            frames = view_tensor.view(b * self.frame_stack, c, h, w)

            # Get feature maps -> (B * F, 512, H_out, W_out)
            feature_map = self.backbone(frames)
            # Project to d_model -> (B * F, d_model, H_out, W_out)
            projected_map = self.projector(feature_map)

            # Tokenize by flattening spatial dimensions
            # (B * F, d_model, N_tokens)
            tokens = projected_map.flatten(2)

            # Reshape back to (B, F, d_model, N_tokens)
            _, _, n_tokens = tokens.shape
            tokens = tokens.view(b, self.frame_stack, self.d_model, n_tokens)

            # Permute and flatten to (B, F * N_tokens, d_model)
            tokens = tokens.permute(0, 1, 3, 2).contiguous()
            tokens = tokens.view(b, self.frame_stack * n_tokens, self.d_model)
            all_tokens.append(tokens)

        # Concatenate tokens from all views -> (B, V * F * N_tokens, d_model)
        final_tokens = torch.cat(all_tokens, dim=1)
        return final_tokens
