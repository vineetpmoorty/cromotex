import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    '''
    Patch Embedding for ECG data to be input to ECGTransformer.
    Converts (12, 1000) to 196 patches of dim patch_dim.
    Source: https://github.com/svthapa/MoRE/blob/main/utils/build_model.py
    '''
    def __init__(
        self, in_channels=12, patch_dim=256,
        intermediate_dim=128, kernel_size=5, stride1=5, stride2=1
    ):
        super(PatchEmbed, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=intermediate_dim,
                kernel_size=kernel_size, stride=stride1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=intermediate_dim, out_channels=patch_dim,
                kernel_size=kernel_size, stride=stride2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(patch_dim),
        )
        self.num_patches = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, patch_dim]
        self.num_patches = x.size(1)  # Save the number of patches
        return x

class ECGPatchTransformer(nn.Module):
    '''
    Transformer for ECG data to generate global
    embeddings for contrastive learning.
    '''
    def __init__(
        self, 
        cfg,
        in_channels=12, 
        seq_len=1000, 
        intermediate_dim=128, 
        kernel_size=5, 
        stride1=5, 
        stride2=1,
        embed_dim=256, 
        num_heads=4, 
        depth=4, 
        num_classes=1
        ):
        super(ECGPatchTransformer, self).__init__()

        if cfg is not None:
            kernel_size = cfg.cromotex.kernel_size
            stride1 = cfg.cromotex.stride1
            stride2 = cfg.cromotex.stride2
            intermediate_dim = cfg.cromotex.intermediate_dim
            embed_dim = cfg.cromotex.embed_dim
            
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_dim=embed_dim,
            intermediate_dim=intermediate_dim,
            kernel_size=kernel_size,
            stride1=stride1,
            stride2=stride2
        )
        self.cfg = cfg
        max_patches = 196
        self.positional_embedding = nn.Parameter(
            torch.randn(1, max_patches + 1, embed_dim)  # +1 for CLS token
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=depth
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        )

    def forward(self, x):
        # Generate patches and embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, patch_dim]

        if self.training and self.cfg.cromotex.patch_drop > 0:
            drop_mask = (
                torch.rand(x.size(0), x.size(1), device=x.device)
                < self.cfg.cromotex.patch_drop
            )
            drop_mask = drop_mask.unsqueeze(-1)
            x = x * (~drop_mask)
        
        num_patches = x.size(1)
        pos_embed = self.positional_embedding[:, :num_patches + 1, :]

        cls_token = self.cls_token.expand(x.size(0), -1, -1)  
        # [batch_size, 1, embed_dim]

        x = torch.cat((cls_token, x), dim=1)

        x = x + pos_embed  
        # Ensure positional embedding matches num_patches

        x = self.transformer_encoder(x)  # [batch_size, num_patches, embed_dim]

        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, num_patches]
        global_embedding = self.global_pool(x).squeeze(-1)  
        # [batch_size, embed_dim]

        # Optional classification
        if self.classifier is not None:
            logits = self.classifier(global_embedding)  
            # [batch_size, num_classes]
            return global_embedding, logits

        return global_embedding

