import torch
import torch.nn as nn

class DummyEncoder(nn.Module):
    def __init__(self, d_model, frame_stack, input_shape):
        super().__init__()
        self.d_model = d_model
        self.dummy_layer = nn.Linear(1, 1)  # Add a parameter for the optimizer

    def forward(self, x):
        # Dummy forward
        return torch.randn(x.shape[0], 147, self.d_model, device=x.device)

class DiTPolicyModel(nn.Module):
    def __init__(self, proprio_dim: int, d_model: int, num_encoder_layers: int, num_decoder_layers: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.dummy_param = nn.Parameter(torch.rand(1))

    def forward(self, image_tokens, qpos, action, timestep, **kwargs):
        return torch.randn_like(action) * self.dummy_param 