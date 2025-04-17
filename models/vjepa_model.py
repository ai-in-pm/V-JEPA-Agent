import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbed(nn.Module):
    """
    Video to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

    def forward(self, x):
        """
        x: [B, T, C, H, W] or [B, C, H, W]
        """
        # Handle different input shapes
        if len(x.shape) == 4:  # [B, C, H, W]
            # For single frame input, add time dimension
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        # Now we have [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        # Project patches
        x = self.proj(x)  # [B, E, T, H//P, W//P]

        # Flatten spatial dimensions
        x = x.flatten(3)  # [B, E, T, (H//P)*(W//P)]

        # Transpose to get [B, T, (H//P)*(W//P), E]
        x = x.permute(0, 2, 3, 1)

        return x

class Attention(nn.Module):
    """
    Multi-head Attention module
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    """
    MLP module
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """
    Transformer block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ContextEncoder(nn.Module):
    """
    Context Encoder for V-JEPA
    Processes masked video and outputs embeddings
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, 16, 1, embed_dim))  # Assuming max 16 frames

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        """
        x: [B, T, C, H, W] or [B, C, H, W] - Video frames
        mask: [B, T, H, W] or [B, H, W] - Binary mask (1 = masked, 0 = visible)
        """
        # Handle different input shapes
        if len(x.shape) == 4:  # [B, C, H, W] - Single frame or batch of frames
            B, C, H, W = x.shape
            # Reshape to [B, 1, C, H, W] to handle as a single-frame video
            x = x.unsqueeze(1)
            T = 1
            if mask is not None and len(mask.shape) == 3:  # [B, H, W]
                mask = mask.unsqueeze(1)  # [B, 1, H, W]
        elif len(x.shape) == 5:  # [B, T, C, H, W] - Video
            B, T, C, H, W = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected [B, C, H, W] or [B, T, C, H, W]")

        # Patch embedding
        x = self.patch_embed(x)  # [B, T, N, E]

        # Add position embeddings
        x = x + self.pos_embed[:, :, :x.size(2), :]
        x = x + self.temporal_embed[:, :T, :, :]

        # Replace masked patches with mask token
        if mask is not None:
            # Downsample mask to patch size
            P = self.patch_embed.patch_size
            mask = F.avg_pool2d(mask.float(), kernel_size=P, stride=P).bool()  # [B, T, H//P, W//P]
            mask = mask.flatten(2)  # [B, T, (H//P)*(W//P)]

            # Expand mask token to batch size
            mask_tokens = self.mask_token.expand(B, T, 1, -1)

            # Apply mask
            w = mask.unsqueeze(-1).type_as(mask_tokens)
            x = x * (1 - w) + mask_tokens * w

        # Reshape to sequence format for transformer
        x = x.reshape(B, T * x.size(2), -1)  # [B, T*N, E]

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Reshape back to [B, T, N, E]
        x = x.reshape(B, T, -1, x.size(-1))

        return x

class Predictor(nn.Module):
    """
    Predictor for V-JEPA
    Predicts embeddings for masked regions
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()

        # Transformer blocks (narrower than the encoder)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Projection head
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        x: [B, T, N, E] - Context embeddings
        mask: [B, T, H, W] - Binary mask (1 = masked, 0 = visible)
        """
        B, T, N, E = x.shape

        # Reshape to sequence format for transformer
        x = x.reshape(B, T * N, -1)  # [B, T*N, E]

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Project to target embedding space
        x = self.proj(x)

        # Reshape back to [B, T, N, E]
        x = x.reshape(B, T, N, -1)

        return x

class TargetEncoder(nn.Module):
    """
    Target Encoder for V-JEPA
    Processes unmasked video to provide target embeddings
    This is essentially the same as the context encoder but with EMA updates
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Position embedding
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, 16, 1, embed_dim))  # Assuming max 16 frames

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)

    def forward(self, x):
        """
        x: [B, T, C, H, W] or [B, C, H, W] - Video frames
        """
        # Handle different input shapes
        if len(x.shape) == 4:  # [B, C, H, W] - Single frame or batch of frames
            B, C, H, W = x.shape
            # Reshape to [B, 1, C, H, W] to handle as a single-frame video
            x = x.unsqueeze(1)
            T = 1
        elif len(x.shape) == 5:  # [B, T, C, H, W] - Video
            B, T, C, H, W = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected [B, C, H, W] or [B, T, C, H, W]")

        # Patch embedding
        x = self.patch_embed(x)  # [B, T, N, E]

        # Add position embeddings
        x = x + self.pos_embed[:, :, :x.size(2), :]
        x = x + self.temporal_embed[:, :T, :, :]

        # Reshape to sequence format for transformer
        x = x.reshape(B, T * x.size(2), -1)  # [B, T*N, E]

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Reshape back to [B, T, N, E]
        x = x.reshape(B, T, -1, x.size(-1))

        return x
