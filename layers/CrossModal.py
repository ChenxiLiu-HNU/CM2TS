import torch.nn as nn

from layers.Transformer_EncDec import DecoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class CrossModal(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        d_ff=None,
        scale=None,
        attn_dropout=0,
        dropout=0,
        activation="gelu",
        n_layers=1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    self_attention=None,
                    cross_attention=AttentionLayer(
                        FullAttention(
                            mask_flag=False,
                            scale=scale,
                            attention_dropout=attn_dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        d_keys,
                        d_values,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for mod in self.layers:
            output = mod(x, cross, x_mask, cross_mask, tau, delta)
        return output
