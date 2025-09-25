import torch
import torch.nn as nn
import math

# Helper for sinusoidal positional encoding for sequences
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.permute(1, 0, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Correctly implemented time embedding MLP
class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.freqs = nn.Parameter(torch.randn(d_model // 2) * 10, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B,)
        args = t[:, None].float() * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)

# AdaLN-Zero Decoder Block
class DiTDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.d_model = d_model
        self.norm1_gate = nn.Linear(d_model, 2 * d_model)
        self.attn_gate = nn.Linear(d_model, d_model)
        self.norm2_gate = nn.Linear(d_model, 2 * d_model)
        self.mlp_gate = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        nn.init.constant_(self.attn_gate.weight, 0)
        nn.init.constant_(self.attn_gate.bias, 0)
        nn.init.constant_(self.mlp_gate.weight, 0)
        nn.init.constant_(self.mlp_gate.bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1 = self.norm1_gate(cond).chunk(2, dim=1)
        alpha1 = self.attn_gate(cond)
        h_norm1 = self.norm1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h_attn, _ = self.attn(h_norm1, h_norm1, h_norm1)
        x = x + alpha1.unsqueeze(1) * h_attn

        gamma2, beta2 = self.norm2_gate(cond).chunk(2, dim=1)
        alpha2 = self.mlp_gate(cond)
        h_norm2 = self.norm2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        h_mlp = self.mlp(h_norm2)
        x = x + alpha2.unsqueeze(1) * h_mlp
        return x

# Main Policy Model
class DiTPolicyModel(nn.Module):
    def __init__(self, proprio_dim: int, d_model: int, num_encoder_layers: int, num_decoder_layers: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.proprio_projector = nn.Linear(proprio_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4, batch_first=True) for _ in range(num_encoder_layers)]
        )
        self.time_mlp = TimestepEmbedding(d_model)
        self.action_projector = nn.Linear(16, d_model)
        self.decoder_blocks = nn.ModuleList(
            [DiTDecoderBlock(d_model, num_heads) for _ in range(num_decoder_layers)]
        )
        self.final_layer = nn.Linear(d_model, 16)

    def forward(self, image_tokens, qpos, action, timestep, **kwargs):
        proprio_tokens = self.proprio_projector(qpos).unsqueeze(1)
        encoder_input = torch.cat([image_tokens, proprio_tokens], dim=1)
        h = self.pos_encoding(encoder_input)
        encoder_outputs = []
        for layer in self.encoder_layers:
            h = layer(h)
            encoder_outputs.append(h)

        t_emb = self.time_mlp(timestep)
        x_emb = self.action_projector(action)

        for i, decoder_block in enumerate(self.decoder_blocks):
            encoder_output = encoder_outputs[i % len(encoder_outputs)]
            cond = t_emb + encoder_output.mean(dim=1)
            x_emb = decoder_block(x_emb, cond)

        predicted_noise = self.final_layer(x_emb)
        return predicted_noise
