import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsDehazeLayer(nn.Module):
    """
    物理合成层 (支持 A 为全局向量或空间图)
    I_reconstructed = J * t + A * (1 - t)
    """

    def __init__(self):
        super().__init__()

    def forward(self, img_clean, transmission, atmosphere):
        # img_clean: (B,3,H,W)
        # transmission: (B,1,H,W)
        # atmosphere: (B,3,1,1) or (B,3,H,W) or (1,3,H,W) or (B,3)

        if atmosphere.dim() == 2:
            # (B,3) or (1,3)
            if atmosphere.shape[0] == img_clean.shape[0]:
                A = atmosphere.view(img_clean.shape[0], 3, 1, 1)
            else:
                A = atmosphere.view(1, 3, 1, 1).expand(img_clean.shape[0], -1, -1, -1)
        elif atmosphere.dim() == 4:
            A = atmosphere
            if A.shape[0] != img_clean.shape[0]:
                A = A.expand(img_clean.shape[0], -1, -1, -1)
            if A.shape[2:] != img_clean.shape[2:]:
                A = F.interpolate(A, size=img_clean.shape[2:], mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unsupported atmosphere dim: {atmosphere.dim()}")

        if transmission.shape[1] != 1:
            raise ValueError("transmission should be (B,1,H,W)")

        img_hazy_hat = img_clean * transmission + A * (1 - transmission)
        return img_hazy_hat


class UNetTransmission(nn.Module):
    """
    稍增强版轻量 U-Net，用于预测 transmission map
    输出范围: [T_MIN, 1.0]

    ✅ 论文可视化增强：
      forward(x, return_features=True) 会返回 (T, feats_dict)
      feats_dict: e1/e2/e3/e4/b/d4/d3/d2
    """

    def __init__(self, input_dim=3, t_min=0.1):
        super().__init__()
        self.t_min = float(t_min)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.btl = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x, return_features: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.btl(e4)

        u4 = self.up4(b)
        if u4.shape[2:] != e3.shape[2:]:
            u4 = F.interpolate(u4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([u4, e3], dim=1))

        u3 = self.up3(d4)
        if u3.shape[2:] != e2.shape[2:]:
            u3 = F.interpolate(u3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e2], dim=1))

        u2 = self.up2(d3)
        if u2.shape[2:] != e1.shape[2:]:
            u2 = F.interpolate(u2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        out = self.final(d2)
        t = torch.sigmoid(out) * (1.0 - self.t_min) + self.t_min

        if not return_features:
            return t

        feats = {
            "e1": e1, "e2": e2, "e3": e3, "e4": e4,
            "b": b,
            "d4": d4, "d3": d3, "d2": d2
        }
        return t, feats


class LossFactoryV7_1_NATURE:
    """
    Stage2 用的 loss 工厂
    修复重点：亮度保护（防止白云变黑）、灰度保持

    ✅ 论文可视化增强：
      compute_t_guide_map()：直接导出 t_guide（含亮度保护逻辑）
    """

    def __init__(self, device):
        self.device = device

    def get_dark_channel(self, img, patch_size=25):
        min_c, _ = torch.min(img, dim=1, keepdim=True)
        padding = patch_size // 2
        dark_channel = -F.max_pool2d(-min_c, kernel_size=patch_size, stride=1, padding=padding)
        return dark_channel

    def get_atmospheric_light(self, img, patch_size=35, top_percent=0.001, method="max"):
        b, c, h, w = img.shape
        dark_channel = self.get_dark_channel(img, patch_size)
        num_pixels = h * w
        num_top = int(max(num_pixels * float(top_percent), 1))

        dc_flat = dark_channel.view(b, -1)
        img_flat = img.view(b, c, -1)

        _, indices = torch.topk(dc_flat, k=num_top, dim=1)
        indices_expand = indices.unsqueeze(1).expand(-1, c, -1)
        top_pixels = torch.gather(img_flat, 2, indices_expand)  # (B,3,num_top)

        if method == "max":
            atmosphere, _ = torch.max(top_pixels, dim=2)
        else:
            atmosphere = torch.mean(top_pixels, dim=2)
        return atmosphere

    def get_bright_region_mask(self, img, threshold=0.7):
        """
        强力膨胀 Mask，保护云层/天空
        """
        max_v, _ = torch.max(img, dim=1, keepdim=True)
        min_v, _ = torch.min(img, dim=1, keepdim=True)
        saturation = (max_v - min_v) / (max_v + 1e-6)

        is_bright = (max_v > threshold).float()
        is_low_sat = (saturation < 0.3).float()
        mask = is_bright * is_low_sat

        kernel_size = 15
        padding = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
        mask = F.conv2d(mask, kernel, padding=padding)
        mask = torch.clamp(mask, 0, 1)
        return mask

    def get_saturation(self, img):
        max_v, _ = torch.max(img, dim=1, keepdim=True)
        min_v, _ = torch.min(img, dim=1, keepdim=True)
        saturation = (max_v - min_v) / (max_v + 1e-6)
        return saturation

    def loss_reconstruction(self, I_hat, I_gt):
        return F.l1_loss(I_hat, I_gt)

    def loss_dark_channel_hinge(self, img, target=0.02, mask=None, patch_size=25, protect_bright=True):
        dc = self.get_dark_channel(img, patch_size=patch_size)

        w = torch.ones_like(dc)
        if mask is not None:
            w = w * (1.0 - mask)

        if protect_bright:
            max_v, _ = torch.max(img, dim=1, keepdim=True)
            is_bright = (max_v > 0.85).float()
            w = w * (1.0 - 0.9 * is_bright)

        return torch.sum(F.relu(dc - target) * w) / (torch.sum(w) + 1e-6)

    def compute_t_guide_map(self, I, A=None, omega=0.95, patch_size=25, t_min=0.05):
        """
        ✅ 直接返回 t_guide（含亮度保护逻辑），用于论文可视化。
        """
        with torch.no_grad():
            if A is not None:
                if A.dim() == 2:
                    if A.shape[0] == I.shape[0]:
                        Ab = A.view(I.shape[0], 3, 1, 1)
                    else:
                        Ab = A.view(1, 3, 1, 1).expand(I.shape[0], -1, -1, -1)
                elif A.dim() == 4:
                    Ab = A
                    if Ab.shape[0] != I.shape[0]:
                        Ab = Ab.expand(I.shape[0], -1, -1, -1)
                    if Ab.shape[2:] != I.shape[2:]:
                        Ab = F.interpolate(Ab, size=I.shape[2:], mode="bilinear", align_corners=False)
                else:
                    raise ValueError(f"Unsupported A dim: {A.dim()}")

                I_norm = I / (Ab + 1e-6)
                I_norm = torch.clamp(I_norm, 0.0, 3.0)
            else:
                I_norm = I

            dc = self.get_dark_channel(I_norm, patch_size=patch_size)
            t_guide_raw = 1.0 - omega * dc

            max_I, _ = torch.max(I, dim=1, keepdim=True)
            bright_factor = torch.clamp((max_I - 0.85) / 0.15, 0.0, 1.0)

            t_guide = t_guide_raw * (1.0 - bright_factor) + 1.0 * bright_factor
            t_guide = torch.clamp(t_guide, t_min, 1.0)

        return t_guide

    def loss_transmission_guide(self, T_pred, I, A=None, omega=0.95, patch_size=25, t_min=0.05):
        t_guide = self.compute_t_guide_map(I, A=A, omega=omega, patch_size=patch_size, t_min=t_min)
        return F.mse_loss(T_pred, t_guide)

    def loss_gray_retention(self, J, I, mask=None):
        max_I, _ = torch.max(I, dim=1, keepdim=True)
        max_J, _ = torch.max(J, dim=1, keepdim=True)

        sat = self.get_saturation(I)
        is_gray = (sat < 0.3).float()
        is_mid_bright = ((max_I > 0.3) & (max_I < 0.9)).float()

        w = is_gray * is_mid_bright
        if mask is not None:
            w = w * (1.0 - mask)

        diff = F.relu((max_I * 0.7) - max_J)
        return torch.sum(diff * w) / (torch.sum(w) + 1e-6)

    def loss_tv(self, img):
        h_tv = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        w_tv = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return h_tv + w_tv

    def loss_edge_aware_tv(self, img, guide_rgb, sigma=10.0):
        gray = 0.2989 * guide_rgb[:, 0:1] + 0.5870 * guide_rgb[:, 1:2] + 0.1140 * guide_rgb[:, 2:3]
        dx_g = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        dy_g = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

        wx = torch.exp(-sigma * dx_g)
        wy = torch.exp(-sigma * dy_g)

        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])

        tv = (dx * wx).mean() + (dy * wy).mean()
        return tv

    def loss_saturation_suppress(self, J, mask=None, sat_sky_max=0.18, sat_global_max=0.85):
        sat = self.get_saturation(J)
        if mask is not None:
            loss_sky = torch.mean(F.relu(sat - sat_sky_max) * mask)
            loss_global = torch.mean(F.relu(sat - sat_global_max))
            return loss_sky * 2.0 + loss_global
        else:
            return torch.mean(F.relu(sat - sat_global_max))

    def loss_cosine_consistency(self, current_colors, original_colors):
        eps = 1e-8
        norm_curr = F.normalize(current_colors + eps, p=2, dim=1)
        norm_orig = F.normalize(original_colors + eps, p=2, dim=1)
        cos_sim = torch.sum(norm_curr * norm_orig, dim=1)
        loss = 1.0 - torch.mean(cos_sim)
        return loss

    def loss_color_uniformity(self, img, mask):
        w = mask
        w_sum = torch.sum(w) + 1e-6
        mean = torch.sum(img * w, dim=(0, 2, 3), keepdim=True) / w_sum
        var = torch.sum(((img - mean) ** 2) * w, dim=(0, 2, 3), keepdim=True) / w_sum
        return torch.mean(var)


class DehazeGaussianImageModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_net = UNetTransmission(input_dim=3, t_min=0.1)
        self.physics = PhysicsDehazeLayer()

    def estimate_transmission(self, img, return_features: bool = False):
        return self.t_net(img, return_features=return_features)

    def reconstruct(self, clean_img, t_map, atmosphere):
        return self.physics(clean_img, t_map, atmosphere)
