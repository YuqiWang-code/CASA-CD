"""
Siamese U-Net for Change Detection.

Baseline models:
- B0: Siamese U-Net (concat/diff features)
- B1: Siamese U-Net + Attention-only
- B2: Siamese U-Net + CoSA (cross-correlation guided attention)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# U-Net building blocks
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SiameseEncoder(nn.Module):
    """Shared encoder for T1 and T2 images."""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class SpatialAttention(nn.Module):
    """Simple spatial attention (B1 - no correlation)."""
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)
    
    def forward(self, x):
        # x: [B, C, H, W]
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        a = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * a


class ChannelSpatialAttention(nn.Module):
    """
    Channel-Spatial Attention Module.
    Combines channel attention and spatial attention for better feature refinement.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Fusion Module.
    Fuses features from multiple scales to capture both fine and coarse details.
    """
    def __init__(self, channels_list):
        super().__init__()
        # channels_list: list of channel numbers at different scales
        self.num_scales = len(channels_list)
        
        # Reduce all to same channel size (use min channels)
        min_channels = min(channels_list)
        self.reductions = nn.ModuleList([
            nn.Conv2d(ch, min_channels, 1) if ch != min_channels else nn.Identity()
            for ch in channels_list
        ])
        
        # Fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(min_channels * self.num_scales, min_channels, 3, padding=1),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of features at different scales [B, C_i, H_i, W_i]
        
        Returns:
            fused: [B, C_out, H, W] - Fused features at highest resolution
        """
        # Upsample all to highest resolution
        target_size = features_list[0].shape[2:]
        upsampled = []
        
        for i, feat in enumerate(features_list):
            # Reduce channels
            feat = self.reductions[i](feat)
            # Upsample to target size
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(feat)
        
        # Concatenate and fuse
        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion(fused)
        
        return fused


class CrossCorrelationModule(nn.Module):
    """
    Cross-correlation between T1 and T2 features.
    For change detection: low correlation = changed region.
    """
    def __init__(self, in_channels, kernel_size=3, topk=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.topk = min(topk, in_channels)
        self.padding = kernel_size // 2
        
        # Unfold for local correlation
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=self.padding)
    
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: [B, C, H, W] - T1 features
            feat2: [B, C, H, W] - T2 features
        
        Returns:
            corr_map: [B, topk, H, W] - Correlation map (high = unchanged, low = changed)
        """
        B, C, H, W = feat1.shape
        
        # Simple approach: compute correlation per channel at each location
        # Normalize features
        feat1_norm = F.normalize(feat1, p=2, dim=1)  # [B, C, H, W]
        feat2_norm = F.normalize(feat2, p=2, dim=1)  # [B, C, H, W]
        
        # Correlation: dot product per channel
        corr = (feat1_norm * feat2_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Expand to topk channels (for compatibility with gate)
        if self.topk > 1:
            corr_map = corr.repeat(1, self.topk, 1, 1)  # [B, topk, H, W]
        else:
            corr_map = corr
        
        return corr_map


class CoSABlockCD(nn.Module):
    """
    CoSA block for Change Detection (B2).
    Uses cross-correlation between T1 and T2 to guide attention.
    """
    def __init__(self, in_channels, kernel_size=3, topk=32, use_learnable_gate=True, gamma_init=0.0):
        super().__init__()
        self.corr_module = CrossCorrelationModule(in_channels, kernel_size, topk)
        # Gate: convert correlation to change attention
        # Low correlation → high change probability
        self.gate = nn.Conv2d(topk, 1, kernel_size=1, bias=False)
        
        # V3: Learnable residual gate (initialized to 0.0 so training starts as baseline)
        if use_learnable_gate:
            # Learnable scalar, init to provided value (default 0.0).
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.register_buffer('gamma', torch.tensor(2.0))  # Fixed scale (old behavior)
    
    def forward(self, feat1, feat2, fused_feat):
        """
        Args:
            feat1: [B, C, H, W] - T1 features
            feat2: [B, C, H, W] - T2 features
            fused_feat: [B, C, H, W] - Fused features (baseline)
        
        Returns:
            output: [B, C, H, W] - Gated features (baseline + γ * cosa_output)
            change_gate: [B, 1, H, W] - Change gate for visualization
        """
        # Compute cross-correlation
        corr_map = self.corr_module(feat1, feat2)  # [B, topk, H, W]
        
        # Convert correlation to change gate
        # Low correlation = high change probability
        change_gate = torch.sigmoid(self.gate(corr_map))  # [B, 1, H, W]
        # Invert: low corr → high gate (change regions)
        change_gate = 1.0 - change_gate
        
        # V3: Residual gate with learnable γ
        # cosa_output = fused_feat * change_gate (the enhancement)
        cosa_output = fused_feat * change_gate
        # output = baseline + γ * cosa_output (γ starts at 0.0)
        output = fused_feat + self.gamma * cosa_output
        
        return output, change_gate


class LocalCorrelationVolume(nn.Module):
    """
    Builds a local correlation volume between f1 and f2.
    For each location in f1, computes correlation with f2 at nearby offsets.
    """
    def __init__(self, radius=4):
        """
        Args:
            radius: Search radius (r). Total offsets = (2r+1)^2
        """
        super().__init__()
        self.radius = radius
        # Create offset grid: [(dx,dy) for dx,dy in [-r...r]]
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append([dx, dy])
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.float32))  # [K, 2]
        self.num_offsets = len(offsets)
    
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: [B, C, H, W] - T1 features
            feat2: [B, C, H, W] - T2 features (same shape)
        
        Returns:
            corr_volume: [B, K, H, W] - Correlation at each offset
        """
        B, C, H, W = feat1.shape
        
        # Normalize features (cosine similarity)
        feat1_norm = F.normalize(feat1, p=2, dim=1)  # [B, C, H, W]
        feat2_norm = F.normalize(feat2, p=2, dim=1)  # [B, C, H, W]
        
        # Build correlation volume
        # For each offset, compute correlation by shifting f2
        corr_list = []
        
        for k in range(self.num_offsets):
            dx, dy = self.offsets[k, 0].item(), self.offsets[k, 1].item()
            
            # Create grid for sampling f2 at offset (dx, dy)
            # Grid coordinates: [-1, 1] range, where (0,0) is center
            y_coords = torch.arange(H, dtype=torch.float32, device=feat1.device)
            x_coords = torch.arange(W, dtype=torch.float32, device=feat1.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Normalize to [-1, 1] and apply offset
            grid_y = (2.0 * grid_y / (H - 1)) - 1.0  # [H, W]
            grid_x = (2.0 * grid_x / (W - 1)) - 1.0  # [H, W]
            
            # Apply offset (in normalized coordinates)
            offset_y = 2.0 * dy / (H - 1) if H > 1 else 0.0
            offset_x = 2.0 * dx / (W - 1) if W > 1 else 0.0
            
            grid_y_offset = grid_y + offset_y
            grid_x_offset = grid_x + offset_x
            
            # Stack: [H, W, 2] where last dim is [y, x]
            grid = torch.stack([grid_y_offset, grid_x_offset], dim=-1)  # [H, W, 2]
            grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
            
            # Sample f2 at offset locations
            f2_shifted = F.grid_sample(
                feat2_norm, grid, mode='bilinear', padding_mode='border', align_corners=True
            )  # [B, C, H, W]
            
            # Compute correlation (dot product over channels)
            corr = (feat1_norm * f2_shifted).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            corr_list.append(corr)
        
        # Stack all correlations: [B, K, H, W]
        corr_volume = torch.cat(corr_list, dim=1)
        
        return corr_volume


class DisplacementEstimation(nn.Module):
    """
    Converts correlation volume into a displacement field using soft-argmax.
    """
    def __init__(self, radius=4, temperature=0.1):
        """
        Args:
            radius: Search radius (must match LocalCorrelationVolume)
            temperature: Temperature for softmax (lower = sharper distribution)
        """
        super().__init__()
        self.radius = radius
        self.temperature = temperature
        
        # Create offset grid (same as LocalCorrelationVolume)
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append([dx, dy])
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.float32))  # [K, 2]
        self.num_offsets = len(offsets)
    
    def forward(self, corr_volume):
        """
        Args:
            corr_volume: [B, K, H, W] - Correlation volume from LocalCorrelationVolume
        
        Returns:
            flow: [B, 2, H, W] - Displacement field (dx, dy)
        """
        B, K, H, W = corr_volume.shape
        
        # Convert correlation to probability distribution
        # Higher correlation = higher probability
        prob = F.softmax(corr_volume / self.temperature, dim=1)  # [B, K, H, W]
        
        # Compute expected displacement (soft-argmax)
        # offsets: [K, 2] -> [1, K, 1, 1, 2]
        offsets_expanded = self.offsets.view(1, K, 1, 1, 2)  # [1, K, 1, 1, 2]
        prob_expanded = prob.unsqueeze(-1)  # [B, K, H, W, 1]
        
        # Weighted sum: flow = Σ prob_k * offset_k
        flow = (prob_expanded * offsets_expanded).sum(dim=1)  # [B, H, W, 2]
        flow = flow.permute(0, 3, 1, 2)  # [B, 2, H, W]
        
        return flow


class FeatureAlignmentModule(nn.Module):
    """
    Alignment-first module: aligns T2 features to T1 before differencing.
    Uses local correlation + soft-argmax + warping.
    """
    def __init__(self, radius=4, temperature=0.1, use_learnable_gate=True):
        """
        Args:
            radius: Search radius for correlation (r=4 → 81 offsets)
            temperature: Temperature for softmax in displacement estimation
            use_learnable_gate: If True, use learnable γ (init 0.0), else fixed scale
        """
        super().__init__()
        self.corr_volume = LocalCorrelationVolume(radius=radius)
        self.displacement = DisplacementEstimation(radius=radius, temperature=temperature)
        
        # Residual gate (like CoSA v3)
        if use_learnable_gate:
            self.gamma = nn.Parameter(torch.zeros(1))  # Starts at 0.0
        else:
            self.register_buffer('gamma', torch.tensor(1.0))  # Fixed scale
    
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: [B, C, H, W] - T1 features
            feat2: [B, C, H, W] - T2 features
        
        Returns:
            diff_aligned: [B, C, H, W] - Aligned difference
            flow: [B, 2, H, W] - Displacement field (for visualization)
        """
        B, C, H, W = feat1.shape
        
        # Baseline difference (unaligned)
        diff_base = torch.abs(feat1 - feat2)  # [B, C, H, W]
        
        # Build correlation volume
        corr_vol = self.corr_volume(feat1, feat2)  # [B, K, H, W]
        
        # Estimate displacement field
        flow = self.displacement(corr_vol)  # [B, 2, H, W]
        
        # Warp f2 toward f1 using the displacement
        # Create sampling grid
        y_coords = torch.arange(H, dtype=torch.float32, device=feat1.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=feat1.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Normalize to [-1, 1]
        grid_y_norm = (2.0 * grid_y / (H - 1)) - 1.0 if H > 1 else grid_y * 0.0
        grid_x_norm = (2.0 * grid_x / (W - 1)) - 1.0 if W > 1 else grid_x * 0.0
        
        # Apply displacement (convert pixel offset to normalized coordinates)
        flow_y = flow[:, 0, :, :]  # [B, H, W]
        flow_x = flow[:, 1, :, :]  # [B, H, W]
        
        # Convert pixel offsets to normalized [-1, 1] range
        flow_y_norm = 2.0 * flow_y / (H - 1) if H > 1 else flow_y * 0.0  # [B, H, W]
        flow_x_norm = 2.0 * flow_x / (W - 1) if W > 1 else flow_x * 0.0  # [B, H, W]
        
        # Create grid with displacement
        # grid_y_norm, grid_x_norm are [H, W], expand to [B, H, W]
        grid_y_warp = grid_y_norm.unsqueeze(0).expand(B, -1, -1) + flow_y_norm  # [B, H, W]
        grid_x_warp = grid_x_norm.unsqueeze(0).expand(B, -1, -1) + flow_x_norm  # [B, H, W]
        
        # Stack: [B, H, W, 2] where last dim is [y, x]
        grid = torch.stack([grid_y_warp, grid_x_warp], dim=-1)  # [B, H, W, 2]
        
        # Warp f2
        feat2_warped = F.grid_sample(
            feat2, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # [B, C, H, W]
        
        # Compute aligned difference
        diff_aligned = torch.abs(feat1 - feat2_warped)  # [B, C, H, W]
        
        # Residual combination: diff_base + γ * (diff_aligned - diff_base)
        diff_final = diff_base + self.gamma * (diff_aligned - diff_base)
        
        return diff_final, flow


class SiameseUNet(nn.Module):
    """Baseline Siamese U-Net (B0) - FC-Siam-diff style."""
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, fusion='diff'):
        super().__init__()
        self.fusion = fusion  # Only 'diff' supported for now
        
        # Shared encoder
        self.encoder = SiameseEncoder(in_channels, base_channels)
        
        # Decoder (for diff fusion)
        # After diff: x5=1024, x4=512, x3=256, x2=128, x1=64
        factor = 2
        # up1: concat(x5_up, x4) = concat(1024, 512) = 1536 -> 512
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8 // factor, bilinear=True)
        # up2: concat(up1_out, x3) where up1_out=512, x3=256 -> 768 -> 256
        self.up2 = Up((base_channels * 8 // factor) + base_channels * 4, base_channels * 4 // factor, bilinear=True)
        # up3: concat(up2_out, x2) where up2_out=256, x2=128 -> 384 -> 128
        self.up3 = Up((base_channels * 4 // factor) + base_channels * 2, base_channels * 2 // factor, bilinear=True)
        # up4: concat(up3_out, x1) where up3_out=128, x1=64 -> 192 -> 64
        self.up4 = Up((base_channels * 2 // factor) + base_channels, base_channels // factor, bilinear=True)
        # Final upsample (from 32 channels to 64)
        self.up5_conv = DoubleConv(base_channels // factor, base_channels)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, img1, img2):
        """
        Args:
            img1: [B, 3, H, W] - T1 image
            img2: [B, 3, H, W] - T2 image
        
        Returns:
            logits: [B, 1, H, W] - Change prediction
        """
        # Encode both images
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.encoder(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.encoder(img2)
        
        # Fuse features (diff)
        x5 = torch.abs(f1_5 - f2_5)
        x4 = torch.abs(f1_4 - f2_4)
        x3 = torch.abs(f1_3 - f2_3)
        x2 = torch.abs(f1_2 - f2_2)
        x1 = torch.abs(f1_1 - f2_1)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final processing (x is already at input resolution after up4)
        x = self.up5_conv(x)
        logits = self.outc(x)
        
        return logits


class SiameseUNetAttention(nn.Module):
    """Siamese U-Net with Attention-only (B1)."""
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, fusion='diff'):
        super().__init__()
        self.fusion = fusion
        self.encoder = SiameseEncoder(in_channels, base_channels)
        
        # Attention at bottleneck (1/16 resolution)
        self.attention = SpatialAttention(k=7)
        
        # Decoder (for diff fusion)
        factor = 2
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8 // factor, bilinear=True)
        self.up2 = Up((base_channels * 8 // factor) + base_channels * 4, base_channels * 4 // factor, bilinear=True)
        self.up3 = Up((base_channels * 4 // factor) + base_channels * 2, base_channels * 2 // factor, bilinear=True)
        self.up4 = Up((base_channels * 2 // factor) + base_channels, base_channels // factor, bilinear=True)
        self.up5_conv = DoubleConv(base_channels // factor, base_channels)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, img1, img2):
        # Encode
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.encoder(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.encoder(img2)
        
        # Fuse (diff)
        x5 = torch.abs(f1_5 - f2_5)
        x4 = torch.abs(f1_4 - f2_4)
        x3 = torch.abs(f1_3 - f2_3)
        x2 = torch.abs(f1_2 - f2_2)
        x1 = torch.abs(f1_1 - f2_1)
        
        # Apply attention at bottleneck
        x5 = self.attention(x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final processing (x is already at input resolution after up4)
        x = self.up5_conv(x)
        logits = self.outc(x)
        
        return logits


class SiameseUNetCoSA(nn.Module):
    """Siamese U-Net with CoSA (B2) - Cross-correlation guided attention."""
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, fusion='diff', topk=32, use_multiscale=True, use_learnable_gate=True):
        super().__init__()
        self.fusion = fusion
        self.encoder = SiameseEncoder(in_channels, base_channels)
        self.use_multiscale = use_multiscale
        
        # V3: Multi-scale CoSA placement (1/8 and 1/16)
        if use_multiscale:
            # CoSA at 1/8 resolution (encoder stage 4)
            self.cosa_block_x4 = CoSABlockCD(base_channels * 8, kernel_size=3, topk=topk, use_learnable_gate=use_learnable_gate)
            # CoSA at 1/16 resolution (encoder stage 5, bottleneck)
            self.cosa_block_x5 = CoSABlockCD(base_channels * 16, kernel_size=3, topk=topk, use_learnable_gate=use_learnable_gate)
        else:
            # Original: Only at bottleneck
            self.cosa_block = CoSABlockCD(base_channels * 16, kernel_size=3, topk=topk, use_learnable_gate=use_learnable_gate)
        
        # Decoder (for diff fusion)
        factor = 2
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8 // factor, bilinear=True)
        self.up2 = Up((base_channels * 8 // factor) + base_channels * 4, base_channels * 4 // factor, bilinear=True)
        self.up3 = Up((base_channels * 4 // factor) + base_channels * 2, base_channels * 2 // factor, bilinear=True)
        self.up4 = Up((base_channels * 2 // factor) + base_channels, base_channels // factor, bilinear=True)
        self.up5_conv = DoubleConv(base_channels // factor, base_channels)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, img1, img2):
        # Encode
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.encoder(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.encoder(img2)
        
        # Fuse (diff)
        x5 = torch.abs(f1_5 - f2_5)
        x4 = torch.abs(f1_4 - f2_4)
        x3 = torch.abs(f1_3 - f2_3)
        x2 = torch.abs(f1_2 - f2_2)
        x1 = torch.abs(f1_1 - f2_1)
        
        # V3: Apply CoSA at multiple scales with residual gates
        if self.use_multiscale:
            # CoSA at 1/8 (x4) - mid-level features for building changes
            x4, gate4 = self.cosa_block_x4(f1_4, f2_4, x4)
            # CoSA at 1/16 (x5) - bottleneck
            x5, gate5 = self.cosa_block_x5(f1_5, f2_5, x5)
            change_gate = gate5  # Use bottleneck gate for visualization
        else:
            # Original: Only at bottleneck
            x5, change_gate = self.cosa_block(f1_5, f2_5, x5)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final processing (x is already at input resolution after up4)
        x = self.up5_conv(x)
        logits = self.outc(x)
        
        return logits, change_gate

    def forward_with_attention(self, img1, img2):
        """Forward that returns logits and attention maps for visualization."""
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.encoder(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.encoder(img2)
        x5 = torch.abs(f1_5 - f2_5)
        x4 = torch.abs(f1_4 - f2_4)
        x3 = torch.abs(f1_3 - f2_3)
        x2 = torch.abs(f1_2 - f2_2)
        x1 = torch.abs(f1_1 - f2_1)
        B, _, H, W = img1.shape
        attention_maps = {}
        if self.use_multiscale:
            x4, gate4 = self.cosa_block_x4(f1_4, f2_4, x4)
            x5, gate5 = self.cosa_block_x5(f1_5, f2_5, x5)
            g4 = F.interpolate(gate4, size=(H, W), mode='bilinear', align_corners=False)
            g5 = F.interpolate(gate5, size=(H, W), mode='bilinear', align_corners=False)
            attention_maps['attention_scale_8'] = g4.cpu().numpy()
            attention_maps['attention_scale_16'] = g5.cpu().numpy()
            attention_maps['correlation_scale_8'] = (1.0 - g4).cpu().numpy()
            attention_maps['correlation_scale_16'] = (1.0 - g5).cpu().numpy()
        else:
            x5, gate5 = self.cosa_block(f1_5, f2_5, x5)
            g5 = F.interpolate(gate5, size=(H, W), mode='bilinear', align_corners=False)
            attention_maps['attention_scale_16'] = g5.cpu().numpy()
            attention_maps['correlation_scale_16'] = (1.0 - g5).cpu().numpy()
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5_conv(x)
        logits = self.outc(x)
        return logits, attention_maps


class SiameseUNetEnhanced(nn.Module):
    """
    Enhanced Siamese U-Net with:
    - Multi-scale feature fusion
    - Channel-spatial attention at multiple levels
    - CoSA at bottleneck
    - Better feature refinement
    """
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, fusion='diff', topk=32):
        super().__init__()
        self.fusion = fusion
        self.encoder = SiameseEncoder(in_channels, base_channels)
        
        # CoSA block at bottleneck
        self.cosa_block = CoSABlockCD(base_channels * 16, kernel_size=3, topk=topk)
        
        # Channel-spatial attention at multiple decoder levels
        factor = 2
        self.attn_up1 = ChannelSpatialAttention(base_channels * 8 // factor)
        self.attn_up2 = ChannelSpatialAttention(base_channels * 4 // factor)
        self.attn_up3 = ChannelSpatialAttention(base_channels * 2 // factor)
        
        # Decoder
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8 // factor, bilinear=True)
        self.up2 = Up((base_channels * 8 // factor) + base_channels * 4, base_channels * 4 // factor, bilinear=True)
        self.up3 = Up((base_channels * 4 // factor) + base_channels * 2, base_channels * 2 // factor, bilinear=True)
        self.up4 = Up((base_channels * 2 // factor) + base_channels, base_channels // factor, bilinear=True)
        
        # Feature refinement before final output
        # After up4, channels are base_channels // factor
        refine_channels = base_channels // factor
        self.refine = nn.Sequential(
            DoubleConv(refine_channels, base_channels),
            ChannelSpatialAttention(base_channels)
        )
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, img1, img2):
        # Encode
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.encoder(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.encoder(img2)
        
        # Fuse (diff)
        x5 = torch.abs(f1_5 - f2_5)
        x4 = torch.abs(f1_4 - f2_4)
        x3 = torch.abs(f1_3 - f2_3)
        x2 = torch.abs(f1_2 - f2_2)
        x1 = torch.abs(f1_1 - f2_1)
        
        # Apply CoSA at bottleneck
        x5, change_gate = self.cosa_block(f1_5, f2_5, x5)
        
        # Decoder with attention
        x = self.up1(x5, x4)
        x = self.attn_up1(x)
        
        x = self.up2(x, x3)
        x = self.attn_up2(x)
        
        x = self.up3(x, x2)
        x = self.attn_up3(x)
        
        x = self.up4(x, x1)
        
        # Feature refinement before final output
        x = self.refine(x)
        logits = self.outc(x)
        
        return logits, change_gate


class SiameseUNetAligned(nn.Module):
    """
    Siamese U-Net with Alignment-First (B3).
    Aligns T2 features to T1 before differencing to reduce false positives from misalignment.
    Uses local correlation + soft-argmax + warping at x4 (1/8 resolution).
    """
    def __init__(self, in_channels=3, n_classes=1, base_channels=64, fusion='diff', 
                 radius=4, temperature=0.1, use_learnable_gate=True):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            n_classes: Output classes (1 for binary change)
            base_channels: Base channel width
            fusion: Fusion method (only 'diff' supported)
            radius: Search radius for alignment (r=4 → 81 offsets)
            temperature: Temperature for softmax in displacement estimation
            use_learnable_gate: If True, use learnable γ (init 0.0)
        """
        super().__init__()
        self.fusion = fusion
        self.encoder = SiameseEncoder(in_channels, base_channels)
        
        # Alignment module at x4 (1/8 resolution, 512 channels)
        self.alignment = FeatureAlignmentModule(
            radius=radius, 
            temperature=temperature,
            use_learnable_gate=use_learnable_gate
        )
        
        # Decoder (for diff fusion)
        factor = 2
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8 // factor, bilinear=True)
        self.up2 = Up((base_channels * 8 // factor) + base_channels * 4, base_channels * 4 // factor, bilinear=True)
        self.up3 = Up((base_channels * 4 // factor) + base_channels * 2, base_channels * 2 // factor, bilinear=True)
        self.up4 = Up((base_channels * 2 // factor) + base_channels, base_channels // factor, bilinear=True)
        self.up5_conv = DoubleConv(base_channels // factor, base_channels)
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, img1, img2):
        """
        Args:
            img1: [B, 3, H, W] - T1 image
            img2: [B, 3, H, W] - T2 image
        
        Returns:
            logits: [B, 1, H, W] - Change prediction
            flow: [B, 2, H, W] - Displacement field (for visualization)
        """
        # Encode both images
        f1_1, f1_2, f1_3, f1_4, f1_5 = self.encoder(img1)
        f2_1, f2_2, f2_3, f2_4, f2_5 = self.encoder(img2)
        
        # Fuse features (diff) at all scales except x4
        x5 = torch.abs(f1_5 - f2_5)
        x3 = torch.abs(f1_3 - f2_3)
        x2 = torch.abs(f1_2 - f2_2)
        x1 = torch.abs(f1_1 - f2_1)
        
        # Alignment-first at x4: align f2_4 to f1_4, then diff
        x4, flow = self.alignment(f1_4, f2_4)  # [B, C, H, W], [B, 2, H, W]
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final processing
        x = self.up5_conv(x)
        logits = self.outc(x)
        
        return logits, flow
