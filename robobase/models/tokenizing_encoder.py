import torch
import torch.nn as nn
from robobase.models.encoder import EncoderModule
from robobase.utils import weight_init

# Lightweight CNN Backbone
class LightweightCNNBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class TokenizingEncoder(EncoderModule):
    def __init__(self, input_shape, d_model: int, frame_stack: int):
        super().__init__(input_shape)
        self.d_model = d_model
        num_views = input_shape[0]
        view_in_channels = input_shape[1]
        cnn_channels = 32

        self.backbones = nn.ModuleList(
            [LightweightCNNBackbone(in_channels=view_in_channels, channels=cnn_channels) for _ in range(num_views)]
        )
        
        self.projector = nn.Conv2d(cnn_channels, d_model, kernel_size=1)
        
        # Calculate output shape
        test_input = torch.randn(1, view_in_channels, input_shape[2], input_shape[3])
        with torch.no_grad():
            test_output = self.backbones[0](test_input)
            spatial_dim = test_output.shape[2] * test_output.shape[3]
        
        self._output_shape = (num_views * spatial_dim, d_model)
        self.apply(weight_init)

    def forward(self, x):
        # x shape: (B, V, C_in, H, W) where C_in = T*C
        all_view_tokens = []
        for i, view_x in enumerate(x.unbind(1)):
            # view_x shape: (B, C_in, H, W)
            feature_map = self.backbones[i](view_x)
            projected_map = self.projector(feature_map)
            tokens = projected_map.flatten(2).permute(0, 2, 1)
            all_view_tokens.append(tokens)
            
        final_tokens = torch.cat(all_view_tokens, dim=1)
        return final_tokens

    @property
    def output_shape(self):
        return self._output_shape
