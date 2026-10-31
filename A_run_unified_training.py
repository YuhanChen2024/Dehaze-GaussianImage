import time
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from A_AE_G_V5_1 import DehazeGaussianImageModule, LossFactoryV7_1_NATURE
from gsplat2d.project_gaussians import project_gaussians
from gsplat2d.rasterize import rasterize_gaussians
from A_visualization_utils import save_tensor_as_image, save_heatmap, plot_and_save_loss_curves

TILE_SIZE = 16

# ==========================
# CONFIG（重点：Stage2 修复浅灰变黑和白云黑斑）
# ==========================
CONFIG = {
    "DATASET": 'dataset/1',

    # Stage 1 参数
    "NUM_POINTS": 100000,  # 初始点数增加
    "N_SCALES": 1,
    "ALLO_RATIO": 0.5,
    "RECONSTRUCT_ITERATIONS": 30000,
    "RECONSTRUCT_LR": 0.01,

    # ========== Stage 2 训练 schedule ==========
    "ITER_TA_WARMUP": 2000,
    "ITER_GAUSS_WARMUP": 1000,
    "ITER_JOINT": 6000,  # 延长时间以供精细拟合

    "FREEZE_TA_AFTER_WARMUP": True,

    # ========== 学习率 ==========
    "LR_T_NET": 6e-4,
    "LR_A_GRID": 2e-4,

    "LR_GAUSS_RGB": 5e-4,  # 加快颜色适应
    "LR_GAUSS_ALPHA": 2e-3,
    "LR_GAUSS_MEANS": 6e-4,  # 2e-4 -> 6e-4
    "LR_GAUSS_COV": 2e-4,

    # ========== Atmosphere Map ==========
    "A_GRID_HW": 16,
    "A_MIN": 0.25,
    "A_MAX": 0.99,

    # ========== Transmission (物理部分) ==========
    "T_FLOOR": 0.05,  # 保持通透
    "W_T_FLOOR": 1.5,
    "DCP_PATCH": 35,  # 大Patch平滑T，减少噪点
    "DCP_TARGET": 0.02,
    "W_DCP_JEST": 5.0,  # 强力去雾

    # ========== Loss 权重 (物理部分) ==========
    "W_RECON": 10.0,
    "W_T_GUIDE": 5.0,
    "W_TV_T": 0.4,  # 强平滑T
    "EDGE_TV_SIGMA": 10.0,

    "W_SAT_JEST": 3.5,
    "SAT_SKY_MAX": 0.25,
    "SAT_GLOBAL_MAX": 0.85,
    "W_SKY_UNIFORM": 1.0,

    "W_GRAY_RETAIN": 0.5,  # 弱灰度保持

    # ========== [补回遗漏的] A-Map 正则参数 ==========
    "W_A_REG": 0.6,
    "W_A_TV": 0.05,
    "W_AIRLIGHT": 1.0,
    "AIRLIGHT_T_TH": 0.35,

    # ========== 关键：2DGS 拟合权重 (2DGS TO 2DGS 核心) ==========
    "W_J_ALIGN": 1.8,  # 30 -> 20->5.0->2.0
    "ALIGN_T_POWER": 0.5,  # 关注全图细节

    # 关掉平滑，要细节
    "W_SMOOTH_J_UNI": 0.15,  # 0->0.15
    "W_SKY_UNIFORM_J": 0.2, # 0.1->0.2
    "W_TV_J": 0.05, # 0->0.5

    # 保持结构
    "W_SIM_START": 0.5,
    "W_SIM_END": 0.0,
    "W_COS": 1.0,

    # ========== 几何与动态控制 (为了锐度) ==========
    "LARGE_RADIUS_TH": 6.0,  # 4->6
    "W_BG_ALPHA": 0.5,
    "W_BG_TO_LOW": 0.5,
    "W_BG_SAT_G": 0.5,
    "BG_SAT_MAX": 0.35,

    "W_COV_ANISO": 0.1, # 0->0.1
    "ANISO_TH": 4.0, # 10->4
    "W_RADIUS_REG": 0.1,  # 0.2->0.1
    "RADIUS_MAX": 25.0,  # 15->25

    "SMOOTH_EDGE_SIGMA": 15.0,
    "LOWPASS_DS": 8,

    # ========== 几何漂移约束 ==========
    "ENABLE_GEOM_OPT": True,
    "W_MEAN_DRIFT": 0.005,  # 允许移动
    "W_COV_DRIFT": 0.005,
    "MOVE_ONLY_LARGE": False,  # 允许所有球移动

    # ========== 动态增删改 (激进模式) ==========
    "ENABLE_DYNAMIC": True,
    "DYNAMIC_START_IT": 0,  # 立即开始
    "DENSIFY_INTERVAL": 400,  # 高频加密
    "DENSIFY_TOPK": 1000,  # 每次加大量点
    "DENSIFY_COV_SIGMA": 1.5,

    "SPLIT_INTERVAL": 150,
    "SPLIT_TOPK": 2000,
    "SPLIT_FACTOR": 2,
    "SPLIT_SHRINK": 0.8,

    "PRUNE_INTERVAL": 200,
    "PRUNE_ALPHA_TH": 0.02,
    "MAX_POINTS": 350000,  # [极重要] 35万点上限

    "ENABLE_MERGE": True,
    "MERGE_INTERVAL": 1000,
    "MERGE_GRID": 4,
    "MERGE_MAX_GROUPS": 5000,

    # ========== 可视化 ==========
    "VISUALIZATION": {"ENABLED": True, "INTERVAL": 5000}
}

# ==========================
# Helper
# ==========================
def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x / (1 - x))


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W) -> (B,1,H,W)
    return 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight_1x1hw: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B,3,H,W)
    weight: (B,1,H,W) in [0,1]
    """
    w = weight_1x1hw
    if w.shape[1] == 1 and pred.shape[1] == 3:
        w = w.repeat(1, 3, 1, 1)
    return torch.sum(torch.abs(pred - target) * w) / (torch.sum(w) + 1e-6)


def compute_smooth_mask(guide_rgb: torch.Tensor, sigma: float = 15.0) -> torch.Tensor:
    """
    平滑区域 mask：边缘处趋近 0，平坦区域趋近 1
    """
    gray = rgb_to_gray(guide_rgb)
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))

    grad = torch.sqrt(dx * dx + dy * dy + 1e-12)
    smooth = torch.exp(-sigma * grad)
    smooth = F.avg_pool2d(smooth, kernel_size=3, stride=1, padding=1)
    return torch.clamp(smooth, 0.0, 1.0)


def lowpass(img: torch.Tensor, ds: int = 8) -> torch.Tensor:
    """
    快速低频：下采样 avg_pool -> 上采样
    """
    if ds <= 1:
        return img
    B, C, H, W = img.shape
    img_ds = F.avg_pool2d(img, kernel_size=ds, stride=ds)
    img_up = F.interpolate(img_ds, size=(H, W), mode="bilinear", align_corners=False)
    return img_up


def render_2dgs(H, W, means, cov2d, colors, device):
    """
    colors: (N,3) in [0,1]
    return: (1,3,H,W)
    """
    H_pad = (TILE_SIZE - H % TILE_SIZE) % TILE_SIZE
    W_pad = (TILE_SIZE - W % TILE_SIZE) % TILE_SIZE
    H_pad2, W_pad2 = H + H_pad, W + W_pad

    (xys, radii, conics, num_tiles_hit) = project_gaussians(cov2d, means, H_pad2, W_pad2, TILE_SIZE)
    colors_for_render = torch.clamp(colors, 0.0, 1.0)

    out_img_padded = rasterize_gaussians(
        xys, radii, conics, num_tiles_hit,
        colors_for_render, H_pad2, W_pad2, TILE_SIZE
    )[..., :3]

    out_img_padded = out_img_padded.view(1, H_pad2, W_pad2, 3).permute(0, 3, 1, 2).contiguous()
    out_img = out_img_padded[:, :, :H, :W]
    return out_img


def crop_to_orig(x, H, W):
    return x[:, :, :H, :W]


def cov_radius_3sigma(cov2d: torch.Tensor) -> torch.Tensor:
    """
    cov2d: (N,3) representing [[a,b],[b,c]]
    return approx 3-sigma radius in pixels: 3*sqrt(max_eig)
    """
    a = cov2d[:, 0]
    b = cov2d[:, 1]
    c = cov2d[:, 2]
    tr = a + c
    det = a * c - b * b
    delta = torch.clamp(tr * tr / 4.0 - det, min=1e-8)
    s = torch.sqrt(delta)
    eig1 = tr / 2.0 + s
    eig2 = tr / 2.0 - s
    max_eig = torch.clamp(torch.maximum(eig1, eig2), min=1e-8)
    radius = 3.0 * torch.sqrt(max_eig)
    return radius


def cov_anisotropy(cov2d: torch.Tensor) -> torch.Tensor:
    """
    返回 condition number (eig1/eig2)，越大越“长条椭球”
    """
    a = cov2d[:, 0]
    b = cov2d[:, 1]
    c = cov2d[:, 2]
    tr = a + c
    det = a * c - b * b
    delta = torch.clamp(tr * tr / 4.0 - det, min=1e-8)
    s = torch.sqrt(delta)
    eig1 = tr / 2.0 + s
    eig2 = tr / 2.0 - s
    cond = eig1 / (eig2 + 1e-8)
    return cond


def gaussian_color_saturation(colors: torch.Tensor) -> torch.Tensor:
    """
    colors: (N,3) in [0,1]
    sat = (max-min)/(max+eps)
    """
    max_v, _ = torch.max(colors, dim=1, keepdim=True)
    min_v, _ = torch.min(colors, dim=1, keepdim=True)
    sat = (max_v - min_v) / (max_v + 1e-6)
    return sat.squeeze(1)


def sample_mask_nearest(mask_1x1hw: torch.Tensor, means_xy: torch.Tensor) -> torch.Tensor:
    """
    mask: (1,1,H,W) in [0,1]
    means: (N,2) with x,y in pixel coords
    return: (N,) sampled mask value (nearest)
    """
    H = mask_1x1hw.shape[2]
    W = mask_1x1hw.shape[3]
    x = torch.round(means_xy[:, 0]).long().clamp(0, W - 1)
    y = torch.round(means_xy[:, 1]).long().clamp(0, H - 1)
    return mask_1x1hw[0, 0, y, x]


def sample_rgb_nearest(img_1x3hw: torch.Tensor, means_xy: torch.Tensor) -> torch.Tensor:
    """
    img: (1,3,H,W)
    return: (N,3)
    """
    H = img_1x3hw.shape[2]
    W = img_1x3hw.shape[3]
    x = torch.round(means_xy[:, 0]).long().clamp(0, W - 1)
    y = torch.round(means_xy[:, 1]).long().clamp(0, H - 1)
    return img_1x3hw[0, :, y, x].permute(1, 0).contiguous()


def topk_coords_from_error(err_1x1hw: torch.Tensor, k: int, ignore_mask_1x1hw=None, downsample: int = 4):
    """
    从误差图挑 top-k 坐标，用于 densify
    """
    err = err_1x1hw
    if downsample > 1:
        err = F.avg_pool2d(err, kernel_size=downsample, stride=downsample)
        if ignore_mask_1x1hw is not None:
            m = F.avg_pool2d(ignore_mask_1x1hw, kernel_size=downsample, stride=downsample)
            err = err * (1.0 - torch.clamp(m, 0, 1))

    _, _, Hs, Ws = err.shape
    flat = err.view(-1)
    k = min(k, flat.numel())
    vals, idx = torch.topk(flat, k=k, largest=True, sorted=False)
    ys = idx // Ws
    xs = idx % Ws
    # map back
    xs = (xs.float() + 0.5) * downsample
    ys = (ys.float() + 0.5) * downsample
    coords = torch.stack([xs, ys], dim=1)
    return coords


# ==========================
# Gaussian Field (Stage2 only)
# ==========================
class GaussianField2D(nn.Module):
    def __init__(self, means_init, cov2d_init, rgb_logits_init, device, alpha_init=0.99):
        super().__init__()
        self.device = device

        self.means = nn.Parameter(means_init.clone().to(device))
        self.cov2d = nn.Parameter(cov2d_init.clone().to(device))
        self.rgb_logits = nn.Parameter(rgb_logits_init.clone().to(device))

        alpha0 = torch.full((means_init.shape[0], 1), float(alpha_init), device=device)
        self.alpha_logits = nn.Parameter(inverse_sigmoid(alpha0))

        # reference for regularization
        self.ref_rgb = torch.sigmoid(rgb_logits_init.clone()).to(device)
        self.ref_means = means_init.clone().to(device)
        self.ref_cov2d = cov2d_init.clone().to(device)
        self.is_original = torch.ones(means_init.shape[0], dtype=torch.bool, device=device)

    def num_points(self):
        return self.means.shape[0]

    def colors(self):
        return torch.sigmoid(self.rgb_logits)

    def alphas(self):
        return torch.sigmoid(self.alpha_logits)

    def render(self, H, W):
        # opacity folded into color amplitude (no need to modify rasterizer)
        colors = self.colors()
        alphas = self.alphas()
        colors_eff = colors * alphas
        return render_2dgs(H, W, self.means, self.cov2d, colors_eff, self.device)

    @torch.no_grad()
    def sanitize_(self, H, W):
        # clamp means
        self.means.data[:, 0] = torch.clamp(self.means.data[:, 0], 0.0, float(W - 1))
        self.means.data[:, 1] = torch.clamp(self.means.data[:, 1], 0.0, float(H - 1))

        # make cov "roughly" valid
        a = self.cov2d.data[:, 0]
        b = self.cov2d.data[:, 1]
        c = self.cov2d.data[:, 2]

        a = torch.clamp(a, min=1e-4, max=1e4)
        c = torch.clamp(c, min=1e-4, max=1e4)
        max_b = torch.sqrt(a * c) * 0.99
        b = torch.clamp(b, -max_b, max_b)

        self.cov2d.data[:, 0] = a
        self.cov2d.data[:, 1] = b
        self.cov2d.data[:, 2] = c

        self.alpha_logits.data = torch.clamp(self.alpha_logits.data, -10.0, 10.0)

    def radius(self):
        return cov_radius_3sigma(self.cov2d)

    @torch.no_grad()
    def prune_(self, keep_mask: torch.Tensor):
        self.means = nn.Parameter(self.means.data[keep_mask])
        self.cov2d = nn.Parameter(self.cov2d.data[keep_mask])
        self.rgb_logits = nn.Parameter(self.rgb_logits.data[keep_mask])
        self.alpha_logits = nn.Parameter(self.alpha_logits.data[keep_mask])

        self.ref_rgb = self.ref_rgb[keep_mask]
        self.ref_means = self.ref_means[keep_mask]
        self.ref_cov2d = self.ref_cov2d[keep_mask]
        self.is_original = self.is_original[keep_mask]

    @torch.no_grad()
    def append_(self, means_new, cov2d_new, rgb_logits_new, alpha_logits_new, ref_rgb_new=None):
        self.means = nn.Parameter(torch.cat([self.means.data, means_new], dim=0))
        self.cov2d = nn.Parameter(torch.cat([self.cov2d.data, cov2d_new], dim=0))
        self.rgb_logits = nn.Parameter(torch.cat([self.rgb_logits.data, rgb_logits_new], dim=0))
        self.alpha_logits = nn.Parameter(torch.cat([self.alpha_logits.data, alpha_logits_new], dim=0))

        if ref_rgb_new is None:
            ref_rgb_new = torch.sigmoid(rgb_logits_new)
        self.ref_rgb = torch.cat([self.ref_rgb, ref_rgb_new], dim=0)
        self.ref_means = torch.cat([self.ref_means, means_new], dim=0)
        self.ref_cov2d = torch.cat([self.ref_cov2d, cov2d_new], dim=0)
        self.is_original = torch.cat(
            [self.is_original, torch.zeros(means_new.shape[0], dtype=torch.bool, device=self.device)], dim=0
        )

    @torch.no_grad()
    def split_(self, idx: torch.Tensor, split_factor=4, shrink=0.25):
        if idx.numel() == 0:
            return
        idx = idx.long()

        N = self.num_points()
        keep = torch.ones(N, dtype=torch.bool, device=self.device)
        keep[idx] = False

        means_p = self.means.data[idx]
        cov_p = self.cov2d.data[idx]
        rgb_p = self.rgb_logits.data[idx]
        alp_p = self.alpha_logits.data[idx]
        ref_rgb_p = self.ref_rgb[idx]

        r = cov_radius_3sigma(cov_p).unsqueeze(1)  # (M,1)
        jitter = 0.15 * r
        base = means_p.repeat_interleave(split_factor, dim=0)

        offsets = torch.tensor(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            dtype=torch.float32, device=self.device
        )
        offsets = offsets[:split_factor].repeat(means_p.shape[0], 1)
        offsets = offsets * jitter.repeat_interleave(split_factor, dim=0)

        means_child = base + offsets
        cov_child = cov_p.repeat_interleave(split_factor, dim=0) * shrink
        rgb_child = rgb_p.repeat_interleave(split_factor, dim=0)
        alp_child = alp_p.repeat_interleave(split_factor, dim=0)
        ref_rgb_child = ref_rgb_p.repeat_interleave(split_factor, dim=0)

        self.prune_(keep)
        self.append_(means_child, cov_child, rgb_child, alp_child, ref_rgb_new=ref_rgb_child)

    @torch.no_grad()
    def merge_grid_(self, H, W, grid=6, max_groups=2000):
        if self.num_points() < 1000:
            return

        x = torch.clamp(self.means.data[:, 0], 0, W - 1)
        y = torch.clamp(self.means.data[:, 1], 0, H - 1)
        gx = torch.floor(x / grid).long()
        gy = torch.floor(y / grid).long()
        gW = int(math.ceil(W / grid)) + 1
        key = gy * gW + gx

        key_sorted, idx_sorted = torch.sort(key)
        dif = torch.ones_like(key_sorted, dtype=torch.bool)
        dif[1:] = key_sorted[1:] != key_sorted[:-1]
        group_starts = torch.nonzero(dif).squeeze(1)

        if group_starts.numel() > max_groups:
            group_starts = group_starts[:max_groups]

        keep_mask = torch.ones(self.num_points(), dtype=torch.bool, device=self.device)

        for gs in group_starts:
            k0 = key_sorted[gs]
            ge = gs + 1
            while ge < key_sorted.numel() and key_sorted[ge] == k0:
                ge += 1
            group_idx = idx_sorted[gs:ge]
            if group_idx.numel() <= 1:
                continue

            al = torch.sigmoid(self.alpha_logits.data[group_idx]).squeeze(1)
            rep_local = torch.argmax(al)
            rep = group_idx[rep_local]
            others = torch.cat([group_idx[:rep_local], group_idx[rep_local + 1:]], dim=0)
            if others.numel() == 0:
                continue

            w_rep = torch.sigmoid(self.alpha_logits.data[rep]).clamp(1e-4, 1.0)
            w_oth = torch.sigmoid(self.alpha_logits.data[others]).clamp(1e-4, 1.0)
            w_sum = w_rep + torch.sum(w_oth)

            m_new = (self.means.data[rep] * w_rep + torch.sum(self.means.data[others] * w_oth, dim=0)) / w_sum
            self.means.data[rep] = m_new

            c_rep = torch.sigmoid(self.rgb_logits.data[rep])
            c_oth = torch.sigmoid(self.rgb_logits.data[others])
            c_new = (c_rep * w_rep + torch.sum(c_oth * w_oth, dim=0)) / w_sum
            self.rgb_logits.data[rep] = inverse_sigmoid(c_new)

            a_new = torch.clamp((w_sum).squeeze(), 0.0, 1.0)
            self.alpha_logits.data[rep] = inverse_sigmoid(a_new)

            keep_mask[others] = False

        self.prune_(keep_mask)


def create_gaussian_optimizer(gfield: GaussianField2D, cfg: dict):
    return torch.optim.Adam([
        {"params": [gfield.rgb_logits], "lr": cfg["LR_GAUSS_RGB"]},
        {"params": [gfield.alpha_logits], "lr": cfg["LR_GAUSS_ALPHA"]},
        {"params": [gfield.means], "lr": cfg["LR_GAUSS_MEANS"]},
        {"params": [gfield.cov2d], "lr": cfg["LR_GAUSS_COV"]},
    ])


# ==========================
# Stage2 training
# ==========================
def run_v7_0_training(image_path: Path, log_dir: Path, config: dict, viz_dir: Path):
    print(f"--- [阶段2] Zero-Shot 2DGS 物理场解耦去雾 (Fix Black Blob/Sky): {image_path.name} ---")

    if config["VISUALIZATION"]["ENABLED"]:
        viz_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== 1) load input (orig size) ==========
    pil = Image.open(image_path).convert("RGB")
    I_orig = transforms.ToTensor()(pil).unsqueeze(0).to(device)
    H_orig, W_orig = I_orig.shape[2], I_orig.shape[3]

    pad_H = (TILE_SIZE - H_orig % TILE_SIZE) % TILE_SIZE
    pad_W = (TILE_SIZE - W_orig % TILE_SIZE) % TILE_SIZE
    H_pad, W_pad = H_orig + pad_H, W_orig + pad_W

    # ========== 2) load Stage1 checkpoint ==========
    stage1_model_path = log_dir / "stage1_reconstruct_model.pth.tar"
    if not stage1_model_path.exists():
        print("错误：未找到阶段1模型。")
        return
    checkpoint = torch.load(stage1_model_path, map_location=device)

    all_means, all_cov2d, all_rgbs = [], [], []
    for state_dict in checkpoint["state_dicts"]:
        all_means.append(state_dict["means"])
        all_cov2d.append(state_dict["cov2d"])
        all_rgbs.append(state_dict["rgbs"])

    means_init = torch.cat(all_means, dim=0).detach().to(device)
    cov2d_init = torch.cat(all_cov2d, dim=0).detach().to(device)
    rgbs_logits_init = torch.cat(all_rgbs, dim=0).detach().to(device)

    colors_frozen = torch.sigmoid(rgbs_logits_init)

    # ========== 3) render hazy input from Stage1 ==========
    with torch.no_grad():
        I_hazy_pad = render_2dgs(H_pad, W_pad, means_init, cov2d_init, colors_frozen, device)
    I_hazy = crop_to_orig(I_hazy_pad, H_orig, W_orig)

    if config["VISUALIZATION"]["ENABLED"]:
        save_tensor_as_image(I_hazy, viz_dir / "00_input_hazy.png")

    # ========== 4) init modules ==========
    model = DehazeGaussianImageModule().to(device)
    loss_factory = LossFactoryV7_1_NATURE(device)

    # init A from hazy (vector)
    with torch.no_grad():
        A_init_vec = loss_factory.get_atmospheric_light(I_hazy, method="max")
        A_init_vec = torch.clamp(A_init_vec, min=config["A_MIN"], max=config["A_MAX"])
        print(f"--- 初始大气光 A_init: {A_init_vec.view(-1).cpu().numpy()} ---")

        sky_mask = loss_factory.get_bright_region_mask(I_hazy)
        if config["VISUALIZATION"]["ENABLED"]:
            save_heatmap(sky_mask, viz_dir / "00_sky_mask_dilated.png")

    # A logits grid
    A_hw = int(config["A_GRID_HW"])
    A_init_grid = A_init_vec.view(1, 3, 1, 1).expand(1, 3, A_hw, A_hw).contiguous()
    A_logits = nn.Parameter(inverse_sigmoid(torch.clamp(A_init_grid, 1e-4, 1 - 1e-4))).to(device)

    # gaussian field with alpha
    gfield = GaussianField2D(means_init, cov2d_init, rgbs_logits_init, device=device, alpha_init=0.99).to(device)

    # optimizers
    opt_ta = torch.optim.Adam([
        {"params": model.t_net.parameters(), "lr": config["LR_T_NET"]},
        {"params": [A_logits], "lr": config["LR_A_GRID"]},
    ])
    opt_g = create_gaussian_optimizer(gfield, config)

    # loss history
    loss_history = {
        "ta_tguide": [],
        "ta_dcp": [],
        "ta_gray": [],
        "ta_air": [],
        "ta_Areg": [],
        "ta_tvT": [],
        "recon": [],
        "j_align": [],
        "tv_j": [],
        "sim": [],
        "cos": [],
        "blob": [],
        "aniso": [],
        "rad": [],
        "j_uni_smooth": [],
        "j_uni_sky": [],
    }

    it_ta = int(config["ITER_TA_WARMUP"])
    it_g = int(config["ITER_GAUSS_WARMUP"])
    it_joint = int(config["ITER_JOINT"])
    total_iters = it_ta + it_g + it_joint

    pbar = tqdm(range(total_iters), desc="Stage2 Dehaze")

    # cache: freeze TA after warmup?
    freeze_ta = bool(config.get("FREEZE_TA_AFTER_WARMUP", True))

    for it in pbar:
        if it < it_ta:
            phase = "TA_WARMUP"
        elif it < it_ta + it_g:
            phase = "GAUSS_WARMUP"
        else:
            phase = "JOINT"
        joint_it = it - (it_ta + it_g)

        opt_ta.zero_grad(set_to_none=True)
        opt_g.zero_grad(set_to_none=True)

        # ---- A_map build ----
        A_grid = torch.sigmoid(A_logits)
        A_grid = torch.clamp(A_grid, config["A_MIN"], config["A_MAX"])
        A_map_pad = F.interpolate(A_grid, size=(H_pad, W_pad), mode="bilinear", align_corners=False)
        A_map = crop_to_orig(A_map_pad, H_orig, W_orig)

        # ---- smooth mask (for blob suppression / uniformity) ----
        with torch.no_grad():
            smooth_mask = compute_smooth_mask(I_hazy, sigma=float(config["SMOOTH_EDGE_SIGMA"]))

        # ---- T prediction ----
        if (phase != "TA_WARMUP") and freeze_ta:
            with torch.no_grad():
                T_pred = model.estimate_transmission(I_hazy)
            T_pred = torch.clamp(T_pred, float(config["T_FLOOR"]), 1.0).detach()
            A_map_for_use = A_map.detach()
        else:
            T_pred = model.estimate_transmission(I_hazy)
            T_pred = torch.clamp(T_pred, float(config["T_FLOOR"]), 1.0)
            A_map_for_use = A_map

        # ---- render J_clean ----
        if phase == "TA_WARMUP":
            with torch.no_grad():
                J_clean_pad = gfield.render(H_pad, W_pad)
        else:
            J_clean_pad = gfield.render(H_pad, W_pad)
        J_clean = crop_to_orig(J_clean_pad, H_orig, W_orig)

        # ---- reconstruct hazy ----
        I_hat = model.reconstruct(J_clean, T_pred, A_map_for_use)

        # ---- physics inverse ----
        # 使用安全的 T，防止除零
        T_safe = torch.clamp(T_pred, min=float(config["T_FLOOR"]))
        J_est = (I_hazy - A_map_for_use * (1.0 - T_pred)) / (T_safe + 1e-6)
        J_est = torch.clamp(J_est, 0.0, 1.0)

        # ==========================
        # Loss
        # ==========================
        loss = torch.tensor(0.0, device=device)

        # --- TA losses ---
        if phase == "TA_WARMUP":
            # 1) t guide (updated: with brightness protection)
            loss_t_guide = config["W_T_GUIDE"] * loss_factory.loss_transmission_guide(
                T_pred, I_hazy, A=A_map_for_use.detach(), omega=0.95, patch_size=config["DCP_PATCH"],
                t_min=float(config["T_FLOOR"])
            )

            # 2) edge-aware TV for T
            loss_tv_t = config["W_TV_T"] * loss_factory.loss_edge_aware_tv(
                T_pred, I_hazy, sigma=float(config["EDGE_TV_SIGMA"])
            )

            # 3) DCP hinge on J_est (updated: with brightness protection)
            loss_dcp = config["W_DCP_JEST"] * loss_factory.loss_dark_channel_hinge(
                J_est, target=float(config["DCP_TARGET"]), mask=sky_mask, patch_size=config["DCP_PATCH"],
                protect_bright=True
            )

            # 4) NEW: Gray Retention (防止浅灰变黑)
            loss_gray = config["W_GRAY_RETAIN"] * loss_factory.loss_gray_retention(J_est, I_hazy, mask=sky_mask)

            # 5) saturation suppress
            loss_sat = config["W_SAT_JEST"] * loss_factory.loss_saturation_suppress(
                J_est, mask=sky_mask, sat_sky_max=float(config["SAT_SKY_MAX"]),
                sat_global_max=float(config["SAT_GLOBAL_MAX"])
            )

            # 6) sky uniformity
            loss_sky_uni = config["W_SKY_UNIFORM"] * loss_factory.loss_color_uniformity(J_est, sky_mask)

            # 7) A regularization
            loss_A_reg = config["W_A_REG"] * F.mse_loss(A_grid, A_init_grid)
            loss_A_tv = config["W_A_TV"] * loss_factory.loss_tv(A_grid)

            # 8) airlight consistency
            with torch.no_grad():
                m_air = (T_pred < float(config["AIRLIGHT_T_TH"])).float()
                m_air = torch.clamp(m_air + sky_mask, 0, 1)
            loss_air = config["W_AIRLIGHT"] * weighted_l1(I_hazy, A_map_for_use, m_air)

            # 9) T floor safety
            with torch.no_grad():
                dark = (rgb_to_gray(I_hazy) < 0.25).float()
                w_floor = (1.0 - sky_mask) * (0.5 + 0.5 * dark)
            loss_t_floor = config["W_T_FLOOR"] * torch.mean(F.relu(float(config["T_FLOOR"]) - T_pred) * w_floor)

            loss = (
                    loss_t_guide + loss_tv_t + loss_dcp + loss_gray + loss_sat + loss_sky_uni +
                    loss_A_reg + loss_A_tv + loss_air + loss_t_floor
            )

            loss.backward()
            opt_ta.step()

            # log
            loss_history["ta_tguide"].append(loss_t_guide.item())
            loss_history["ta_dcp"].append(loss_dcp.item())
            loss_history["ta_gray"].append(loss_gray.item())
            loss_history["ta_air"].append(loss_air.item())
            loss_history["ta_Areg"].append((loss_A_reg + loss_A_tv).item())
            loss_history["ta_tvT"].append(loss_tv_t.item())

        else:
            # --- Gauss/JOIN losses ---
            if phase == "JOINT":
                w_sim = config["W_SIM_START"] + (config["W_SIM_END"] - config["W_SIM_START"]) * (
                        joint_it / max(it_joint - 1, 1)
                )
            else:
                w_sim = config["W_SIM_START"]

            loss_recon = config["W_RECON"] * loss_factory.loss_reconstruction(I_hat, I_hazy)

            with torch.no_grad():
                conf = torch.clamp(T_pred, float(config["T_FLOOR"]), 1.0) ** float(config["ALIGN_T_POWER"])
                conf = conf * (1.0 - sky_mask)
            loss_j_align = config["W_J_ALIGN"] * weighted_l1(J_clean, J_est.detach(), conf)

            loss_tv_j = config["W_TV_J"] * loss_factory.loss_edge_aware_tv(
                J_clean, I_hazy, sigma=float(config["EDGE_TV_SIGMA"])
            )

            loss_j_uni_smooth = config["W_SMOOTH_J_UNI"] * loss_factory.loss_color_uniformity(J_clean, smooth_mask)
            loss_j_uni_sky = config["W_SKY_UNIFORM_J"] * loss_factory.loss_color_uniformity(J_clean, sky_mask)

            colors_now = gfield.colors()
            colors_ref = gfield.ref_rgb
            orig_mask = gfield.is_original
            if torch.any(orig_mask):
                loss_sim = w_sim * F.mse_loss(colors_now[orig_mask], colors_ref[orig_mask])
                loss_cos = config["W_COS"] * loss_factory.loss_cosine_consistency(colors_now[orig_mask],
                                                                                  colors_ref[orig_mask])
            else:
                loss_sim = torch.tensor(0.0, device=device)
                loss_cos = torch.tensor(0.0, device=device)

            rad = gfield.radius()
            large_w = torch.sigmoid((rad - float(config["LARGE_RADIUS_TH"])) / 3.0)

            with torch.no_grad():
                smooth_w_g = sample_mask_nearest(smooth_mask, gfield.means)
            blob_w = large_w * torch.clamp(smooth_w_g, 0, 1)

            alpha_g = gfield.alphas().squeeze(1)
            col_g = gfield.colors()

            with torch.no_grad():
                J_low = lowpass(J_est.detach(), ds=int(config["LOWPASS_DS"]))
                J_low_g = sample_rgb_nearest(J_low, gfield.means)

            loss_blob_alpha = torch.mean(alpha_g * blob_w) * config["W_BG_ALPHA"]
            loss_blob_to_low = torch.mean(torch.abs(col_g - J_low_g) * blob_w.unsqueeze(1)) * config["W_BG_TO_LOW"]
            sat_g = gaussian_color_saturation(col_g)
            loss_blob_sat = torch.mean(F.relu(sat_g - float(config["BG_SAT_MAX"])) * blob_w) * config["W_BG_SAT_G"]
            loss_blob = loss_blob_alpha + loss_blob_to_low + loss_blob_sat

            cond = cov_anisotropy(gfield.cov2d)
            loss_aniso = config["W_COV_ANISO"] * torch.mean(F.relu(cond - float(config["ANISO_TH"])) * large_w)

            loss_rad = config["W_RADIUS_REG"] * torch.mean(F.relu(rad - float(config["RADIUS_MAX"])) ** 2)

            if config["ENABLE_GEOM_OPT"]:
                if torch.any(orig_mask):
                    loss_mean_drift = config["W_MEAN_DRIFT"] * F.mse_loss(gfield.means[orig_mask],
                                                                          gfield.ref_means[orig_mask])
                    loss_cov_drift = config["W_COV_DRIFT"] * F.mse_loss(gfield.cov2d[orig_mask],
                                                                        gfield.ref_cov2d[orig_mask])
                else:
                    loss_mean_drift = torch.tensor(0.0, device=device)
                    loss_cov_drift = torch.tensor(0.0, device=device)
            else:
                loss_mean_drift = torch.tensor(0.0, device=device)
                loss_cov_drift = torch.tensor(0.0, device=device)

            loss = (
                    loss_recon +
                    loss_j_align +
                    loss_tv_j +
                    loss_j_uni_smooth +
                    loss_j_uni_sky +
                    loss_sim +
                    loss_cos +
                    loss_blob +
                    loss_aniso +
                    loss_rad +
                    loss_mean_drift +
                    loss_cov_drift
            )

            loss.backward()

            if config["ENABLE_GEOM_OPT"] and config["MOVE_ONLY_LARGE"]:
                with torch.no_grad():
                    allow = (large_w > 0.5).float().unsqueeze(1)
                    if gfield.means.grad is not None:
                        gfield.means.grad.mul_(allow)
                    if gfield.cov2d.grad is not None:
                        gfield.cov2d.grad.mul_(allow)

            if gfield.rgb_logits.grad is not None:
                torch.nn.utils.clip_grad_norm_([gfield.rgb_logits], max_norm=1.0)
            if gfield.alpha_logits.grad is not None:
                torch.nn.utils.clip_grad_norm_([gfield.alpha_logits], max_norm=2.0)

            opt_g.step()
            gfield.sanitize_(H_orig, W_orig)

            # ====== dynamic ops ======
            if config["ENABLE_DYNAMIC"] and phase in ["JOINT", "GAUSS_WARMUP"]:
                if (phase == "JOINT") and (joint_it < int(config["DYNAMIC_START_IT"])):
                    pass
                else:
                    if ((phase == "JOINT") and (joint_it % config["PRUNE_INTERVAL"] == 0)) or (
                            (phase == "GAUSS_WARMUP") and ((it - it_ta) % config["PRUNE_INTERVAL"] == 0)
                    ):
                        with torch.no_grad():
                            alpha = gfield.alphas().squeeze(1)
                            rad_now = gfield.radius()
                            keep = torch.ones(gfield.num_points(), dtype=torch.bool, device=device)
                            keep &= (alpha > config["PRUNE_ALPHA_TH"])
                            keep &= (rad_now < 2000.0)
                            keep &= (gfield.means.data[:, 0] >= 0) & (gfield.means.data[:, 0] <= W_orig - 1)
                            keep &= (gfield.means.data[:, 1] >= 0) & (gfield.means.data[:, 1] <= H_orig - 1)

                            if gfield.num_points() > config["MAX_POINTS"]:
                                with torch.no_grad():
                                    smooth_w_g2 = sample_mask_nearest(smooth_mask, gfield.means)
                                score = torch.sigmoid((rad_now - float(config["LARGE_RADIUS_TH"])) / 3.0) * smooth_w_g2
                                drop_n = int(gfield.num_points() - config["MAX_POINTS"])
                                if drop_n > 0:
                                    _, drop_idx = torch.topk(score, k=min(drop_n, score.numel()), largest=True)
                                    keep[drop_idx] = False

                            gfield.prune_(keep)
                            opt_g = create_gaussian_optimizer(gfield, config)

                    if (phase == "JOINT") and (joint_it % config["DENSIFY_INTERVAL"] == 0):
                        with torch.no_grad():
                            err = torch.mean(torch.abs(J_clean - J_est.detach()), dim=1, keepdim=True)
                            ignore = torch.clamp(sky_mask + (T_pred < 0.05).float(), 0, 1)
                            coords = topk_coords_from_error(err, config["DENSIFY_TOPK"], ignore_mask_1x1hw=ignore,
                                                            downsample=4)

                            x = torch.round(coords[:, 0]).long().clamp(0, W_orig - 1)
                            y = torch.round(coords[:, 1]).long().clamp(0, H_orig - 1)
                            c_new = J_est.detach()[0, :, y, x].permute(1, 0)
                            c_new = torch.clamp(c_new, 0.0, 1.0)
                            rgb_logits_new = inverse_sigmoid(c_new)

                            a_new = torch.full((coords.shape[0], 1), 0.35, device=device)
                            alpha_logits_new = inverse_sigmoid(a_new)

                            means_new = coords.to(device)

                            sigma = float(config["DENSIFY_COV_SIGMA"])
                            aa = torch.full((coords.shape[0], 1), sigma * sigma, device=device)
                            bb = torch.zeros((coords.shape[0], 1), device=device)
                            cc = torch.full((coords.shape[0], 1), sigma * sigma, device=device)
                            cov_new = torch.cat([aa, bb, cc], dim=1)

                            gfield.append_(means_new, cov_new, rgb_logits_new, alpha_logits_new)
                            gfield.sanitize_(H_orig, W_orig)
                            opt_g = create_gaussian_optimizer(gfield, config)

                    if (phase == "JOINT") and (joint_it % config["SPLIT_INTERVAL"] == 0):
                        with torch.no_grad():
                            rad_now = gfield.radius()
                            smooth_w_g2 = sample_mask_nearest(smooth_mask, gfield.means)
                            cand = (rad_now > float(config["LARGE_RADIUS_TH"])) & (smooth_w_g2 > 0.6)
                            idx = torch.nonzero(cand).squeeze(1)
                            if idx.numel() > 0:
                                _, sel = torch.topk(rad_now[idx], k=min(config["SPLIT_TOPK"], idx.numel()),
                                                    largest=True)
                                idx_split = idx[sel]
                                gfield.split_(idx_split, split_factor=config["SPLIT_FACTOR"],
                                              shrink=config["SPLIT_SHRINK"])
                                gfield.sanitize_(H_orig, W_orig)
                                opt_g = create_gaussian_optimizer(gfield, config)

                    if config["ENABLE_MERGE"] and (phase == "JOINT") and (
                            joint_it % config["MERGE_INTERVAL"] == 0) and (
                            gfield.num_points() > config["MAX_POINTS"]):
                        with torch.no_grad():
                            gfield.merge_grid_(H_orig, W_orig, grid=config["MERGE_GRID"],
                                               max_groups=config["MERGE_MAX_GROUPS"])
                            gfield.sanitize_(H_orig, W_orig)
                            opt_g = create_gaussian_optimizer(gfield, config)

            loss_history["recon"].append(loss_recon.item())
            loss_history["j_align"].append(loss_j_align.item())
            loss_history["tv_j"].append(loss_tv_j.item())
            loss_history["sim"].append(loss_sim.item())
            loss_history["cos"].append(loss_cos.item())
            loss_history["blob"].append(loss_blob.item())
            loss_history["aniso"].append(loss_aniso.item())
            loss_history["rad"].append(loss_rad.item())
            loss_history["j_uni_smooth"].append(loss_j_uni_smooth.item())
            loss_history["j_uni_sky"].append(loss_j_uni_sky.item())

        if it % 50 == 0:
            if phase == "TA_WARMUP":
                pbar.set_postfix({
                    "phase": phase,
                    "Amean": f"{A_map.mean().item():.3f}",
                    "Tmean": f"{T_pred.mean().item():.3f}",
                    "GrayL": f"{(loss_history['ta_gray'][-1] if loss_history['ta_gray'] else 0):.3f}",
                })
            else:
                pbar.set_postfix({
                    "phase": phase,
                    "N": gfield.num_points(),
                    "Rec": f"{loss_history['recon'][-1]:.3f}",
                    "Align": f"{loss_history['j_align'][-1]:.3f}",
                    "Blob": f"{loss_history['blob'][-1]:.3f}",
                })

        if config["VISUALIZATION"]["ENABLED"] and (
                it % config["VISUALIZATION"]["INTERVAL"] == 0 or it == total_iters - 1):
            with torch.no_grad():
                T_vis = T_pred.repeat(1, 3, 1, 1)
                J_vis = torch.clamp(J_clean, 0, 1)
                I_vis = torch.clamp(I_hazy, 0, 1)
                I_hat_vis = torch.clamp(I_hat, 0, 1)
                J_est_vis = torch.clamp(J_est, 0, 1)

                combo = torch.cat([I_vis, T_vis, J_vis, J_est_vis, I_hat_vis], dim=3)
                save_tensor_as_image(combo, viz_dir / f"iter_{it:05d}_overview.png")

                save_tensor_as_image(A_map, viz_dir / f"iter_{it:05d}_Amap.png")
                save_heatmap(smooth_mask, viz_dir / f"iter_{it:05d}_smooth_mask.png")

    print("--- 阶段2训练完成，保存结果 ---")

    if config["VISUALIZATION"]["ENABLED"]:
        plot_and_save_loss_curves(loss_history, "Stage2 Loss Curves", viz_dir / "loss_curves.png")

    torch.save({
        "gaussians": {
            "means": gfield.means.detach().cpu(),
            "cov2d": gfield.cov2d.detach().cpu(),
            "rgbs_logits": gfield.rgb_logits.detach().cpu(),
            "alpha_logits": gfield.alpha_logits.detach().cpu(),
            "ref_rgb": gfield.ref_rgb.detach().cpu(),
            "is_original": gfield.is_original.detach().cpu()
        },
        "t_net": model.t_net.state_dict(),
        "A_init_vec": A_init_vec.detach().cpu(),
        "A_logits": A_logits.detach().cpu(),
        "config_stage1": checkpoint["config"]
    }, log_dir / "stage2_dehazed.pth.tar")

    with torch.no_grad():
        final_pad = gfield.render(H_pad, W_pad)
        final = crop_to_orig(final_pad, H_orig, W_orig)
        save_tensor_as_image(final, log_dir / "final_dehazed_result.png")
