import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SwiGLUFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.linear1 = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.linear1(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.linear2(hidden)


class SwigluAttentionBlock(nn.Module):
    def __init__(self, embed_dim, tsfm_hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, tsfm_hidden_dim, embed_dim)

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        log_scale = np.log(inp_x.shape[-2])

        attn_output, _ = self.attn(log_scale * inp_x, inp_x, inp_x, need_weights=False)
        x = x + self.attn_dropout(attn_output)  # Apply dropout to the attention output
        x = x + self.ffn(self.layer_norm_2(x))
        return x


class ResidualBlock(nn.Module):
    # Follows "Identity Mappings in Deep Residual Networks", uses LayerNorm instead of BatchNorm, and LeakyReLU instead of ReLU
    def __init__(self, feat_in=128, feat_out=128, feat_hidden=256, drop_out=0.0, use_norm=True):
        super().__init__()
        # Define the residual block with or without normalization
        if use_norm:
            self.block = nn.Sequential(
                nn.LayerNorm(feat_in),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(p=drop_out),
                nn.Linear(feat_in, feat_hidden),
                nn.LayerNorm(feat_hidden),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(p=drop_out),
                nn.Linear(feat_hidden, feat_out)
            )
        else:
            self.block = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(p=drop_out),
                nn.Linear(feat_in, feat_hidden),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(p=drop_out),
                nn.Linear(feat_hidden, feat_out)
            )

        if feat_in != feat_out:
            self.bypass = nn.Linear(feat_in, feat_out)
        else:
            self.bypass = nn.Identity()

    def forward(self, input_data):
        return self.block(input_data) + self.bypass(input_data)


class ImageFeatureRegressionModel(nn.Module):

    def __init__(self,
                 input_dim=512, output_dim=512, internal_emb_dim=768,
                 num_tsfm_layers=12, tsfm_hidden_dim=1024, num_reg_tok=4, num_heads=8,
                 num_early_lyr=1, num_w_pred_layers=1, early_hidden_dim=1024 * 2, w_pred_hidden_dim=1024 * 2,
                 dropout=0.1, used_token_num=4):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.internal_emb_dim = internal_emb_dim
        self.tsfm_hidden_dim = tsfm_hidden_dim
        self.num_reg_tok = num_reg_tok
        self.num_heads = num_heads
        self.used_token_num = used_token_num
        self.num_tsfm_layers = num_tsfm_layers
        self.num_early_lyr = num_early_lyr
        assert self.num_early_lyr > 0, f'Number of early layer must > 0 to project embedding dimension to hidden embedding dimenstion. But num_early_lyr = {self.num_early_lyr}'
        self.early_hidden_dim = early_hidden_dim
        self.num_w_pred_layers = num_w_pred_layers
        assert self.num_w_pred_layers > 0, f'Number of early weight pred must > 0 to project hidden embedding dimension back to embedding dimenstion. But num_w_pred_lyr = {self.num_w_pred_layers}'
        self.w_pred_hidden_dim = w_pred_hidden_dim

        self.dropout = dropout
        print(f'[INFO] dropout here1={self.dropout}')
        # if self.dropout != 0:
        #     raise ValueError(f'dropout should be > 0, but got {self.dropout}')
        self.early_layers = nn.Sequential(
            ResidualBlock(feat_in=self.input_dim + 2, feat_out=self.internal_emb_dim, feat_hidden=self.early_hidden_dim,
                          drop_out=self.dropout, use_norm=True),
            *(ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.internal_emb_dim,
                            feat_hidden=self.early_hidden_dim,
                            drop_out=self.dropout, use_norm=True) for _ in range(self.num_early_lyr - 1)))

        cls_tensor = torch.randn(1, self.num_reg_tok, self.internal_emb_dim)
        cls_tensor = cls_tensor / (float(self.internal_emb_dim + 1) ** 0.5)
        self.cls_token = nn.Parameter(cls_tensor, requires_grad=True)

        # # === Attention Pooling over registers ===
        # self.pool_token = nn.Parameter(
        #     torch.randn(1, 1, self.internal_emb_dim) / (float(self.internal_emb_dim + 1) ** 0.5))
        # self.pool_ln_q = nn.LayerNorm(self.internal_emb_dim)
        # self.pool_ln_kv = nn.LayerNorm(self.internal_emb_dim)
        # self.pool_attn = nn.MultiheadAttention(
        #     embed_dim=self.internal_emb_dim,
        #     num_heads=self.num_heads,
        #     dropout=self.dropout,
        #     batch_first=True
        # )
        # self.pool_dropout = nn.Dropout(self.dropout)

        # Transformer Layersx
        self.input_dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *(SwigluAttentionBlock(self.internal_emb_dim, self.tsfm_hidden_dim, self.num_heads, dropout=self.dropout)
              for _ in range(self.num_tsfm_layers)))
        self.token_merger = nn.Sequential(
            nn.Linear(self.used_token_num * internal_emb_dim, self.used_token_num * internal_emb_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(self.used_token_num * internal_emb_dim, internal_emb_dim)
        )
        self.feature_regressor = nn.Sequential(
            *(ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.internal_emb_dim,
                            feat_hidden=self.early_hidden_dim,
                            drop_out=self.dropout, use_norm=True) for _ in range(self.num_early_lyr - 1)),
            ResidualBlock(feat_in=self.internal_emb_dim, feat_out=self.output_dim, feat_hidden=self.early_hidden_dim,
                          drop_out=self.dropout, use_norm=True))

    def forward(self, ic_beta, ic_weights, ic_bias):
        """
        Given in-context V nums of [ic_weights,ic_bias, ic_beta] (vol_fingerprint),regress this for the image embeddings.
        vol_fingerprint: (B, V, E+2)
          where B is batch size,
          V is the num of in context voxel samples,
          E is the dim of weights,+1 for bias, +1 for beta
        ic_beta: (B, V, 1), is the in-context neural activations for one image
        ic_weights: (B, V, E), is the voxel weights for incontext learning
        ic_bias: (B, V, 1), is the neural activation for incontext learning
        """
        voxel_fp = torch.cat([ic_weights, ic_bias, ic_beta], dim=-1)  # (B, V, E+2)
        B, V, _ = voxel_fp.shape
        x = self.early_layers(voxel_fp)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.input_dropout(x)
        x = self.transformer(x)
        k = self.used_token_num
        reg = x[:, :k, :]  # (B, R, D)
        reg = self.token_merger(reg.reshape(B, -1)).unsqueeze(1)
        pred_img_emb = self.feature_regressor(reg)
        return pred_img_emb