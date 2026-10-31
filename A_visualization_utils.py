from pathlib import Path
import numpy as np
import torch
import torchvision.utils as tv_utils
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2

def _to_numpy_rgb(img_1x3hw: torch.Tensor) -> np.ndarray:
    img = img_1x3hw.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return img


def save_tensor_as_image(tensor: torch.Tensor, path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(tensor, path, **kwargs)


def save_heatmap(tensor: torch.Tensor, path: Path, colormap=cv2.COLORMAP_JET):
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_tensor = tensor.detach().squeeze().cpu().numpy()
    normalized_tensor = (normalized_tensor - normalized_tensor.min()) / (
        normalized_tensor.max() - normalized_tensor.min() + 1e-8
    )
    heatmap_uint8 = np.uint8(255 * normalized_tensor)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    cv2.imwrite(str(path), heatmap_color)


def plot_and_save_loss_curves(history: dict, title: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 8))
    for label, values in history.items():
        plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_histogram(values, title: str, xlabel: str, save_path: Path, bins: int = 60):
    """
    values: 1D numpy or torch
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values).reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_feature_maps_grid(feat: torch.Tensor, save_path: Path, max_channels: int = 16, normalize: bool = True):
    """
    feat: (B,C,H,W) -> 保存前 max_channels 个通道的网格图（论文常用）
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = feat.detach().cpu()
    if x.dim() != 4:
        raise ValueError("feat must be BCHW")
    x = x[0]  # take batch 0 -> (C,H,W)
    C = x.shape[0]
    k = min(int(max_channels), int(C))
    x = x[:k].unsqueeze(1)  # (k,1,H,W)
    grid = tv_utils.make_grid(x, nrow=int(np.ceil(np.sqrt(k))), normalize=normalize, scale_each=True, padding=2)
    tv_utils.save_image(grid, save_path)


def overlay_points_on_image(img_1x3hw: torch.Tensor, points_xy: np.ndarray, save_path: Path, s: int = 8):
    """
    在图上叠加散点（例如 densify top-k / error top-k）
    points_xy: (N,2) in pixel coords, x,y
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img = _to_numpy_rgb(img_1x3hw)

    H, W = img.shape[0], img.shape[1]
    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(img)
    plt.axis("off")

    if points_xy is not None and len(points_xy) > 0:
        pts = np.asarray(points_xy)
        xs = np.clip(pts[:, 0], 0, W - 1)
        ys = np.clip(pts[:, 1], 0, H - 1)
        plt.scatter(xs, ys, s=s)

    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def _eig_2x2_from_abc(a, b, c):
    """
    cov = [[a,b],[b,c]]
    return eigvals (2,), eigvecs (2,2)
    """
    cov = np.array([[a, b], [b, c]], dtype=np.float64)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        vals = np.array([1e-8, 1e-8], dtype=np.float64)
        vecs = np.eye(2, dtype=np.float64)
    vals = np.maximum(vals, 1e-8)
    return vals, vecs


def visualize_gaussians(means: torch.Tensor, cov2d: torch.Tensor, rgbs: torch.Tensor, H: int, W: int, save_path: Path,
                        bg_color='white', max_points=2000):
    """
    兼容你原先的可视化函数（保留）
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_points = means.shape[0]

    means_np = means.detach().cpu().numpy()
    cov2d_np = cov2d.detach().cpu().numpy()

    if torch.max(rgbs) > 1.0 or torch.min(rgbs) < 0.0:
        colors = torch.sigmoid(rgbs).detach().cpu().numpy()
    else:
        colors = rgbs.detach().cpu().numpy()

    if num_points > max_points:
        indices = np.random.choice(num_points, max_points, replace=False)
        means_np = means_np[indices]
        cov2d_np = cov2d_np[indices]
        colors = colors[indices]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    for i in range(means_np.shape[0]):
        cov_matrix = np.array([[cov2d_np[i, 0], cov2d_np[i, 1]],
                               [cov2d_np[i, 1], cov2d_np[i, 2]]])

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        except np.linalg.LinAlgError:
            continue

        eigenvalues = np.maximum(eigenvalues, 1e-8)

        width = 2 * np.sqrt(eigenvalues[0]) * 3
        height = 2 * np.sqrt(eigenvalues[1]) * 3
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        ellipse = Ellipse(xy=(means_np[i, 0], means_np[i, 1]),
                          width=width,
                          height=height,
                          angle=angle,
                          facecolor=colors[i],
                          edgecolor='none',
                          alpha=0.7)
        ax.add_patch(ellipse)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize_gaussian_ellipses(
    means_xy: torch.Tensor,
    cov2d_abc: torch.Tensor,
    rgb_0to1: torch.Tensor,
    alpha_0to1: torch.Tensor | None,
    H: int,
    W: int,
    save_path: Path,
    background_img_1x3hw: torch.Tensor | None = None,
    max_points: int = 8000,
    seed: int = 0,
    ellipse_sigma: float = 3.0,
):
    """
    ✅ 论文级 2DGS 椭球可视化（支持叠加背景像素图）

    means_xy: (N,2)
    cov2d_abc: (N,3) -> [a,b,c] for [[a,b],[b,c]]
    rgb_0to1: (N,3)
    alpha_0to1: (N,1) or (N,) or None
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    means_np = means_xy.detach().cpu().numpy()
    cov_np = cov2d_abc.detach().cpu().numpy()
    col_np = rgb_0to1.detach().clamp(0, 1).cpu().numpy()

    if alpha_0to1 is None:
        alp_np = None
    else:
        alp_np = alpha_0to1.detach().clamp(0, 1)
        if alp_np.dim() == 2:
            alp_np = alp_np.squeeze(1)
        alp_np = alp_np.cpu().numpy()

    N = means_np.shape[0]
    if N > max_points:
        if alp_np is not None:
            p = alp_np + 1e-6
            p = p / p.sum()
            idx = rng.choice(N, size=max_points, replace=False, p=p)
        else:
            idx = rng.choice(N, size=max_points, replace=False)
        means_np = means_np[idx]
        cov_np = cov_np[idx]
        col_np = col_np[idx]
        alp_np = alp_np[idx] if alp_np is not None else None

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    if background_img_1x3hw is not None:
        bg = _to_numpy_rgb(background_img_1x3hw)
        ax.imshow(bg, extent=(0, W, H, 0))

    for i in range(means_np.shape[0]):
        a, b, c = cov_np[i, 0], cov_np[i, 1], cov_np[i, 2]
        vals, vecs = _eig_2x2_from_abc(a, b, c)

        width = 2 * np.sqrt(vals[0]) * ellipse_sigma
        height = 2 * np.sqrt(vals[1]) * ellipse_sigma
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        face = col_np[i]
        a_draw = 0.7 if alp_np is None else float(np.clip(alp_np[i], 0.05, 0.95))

        e = Ellipse(
            xy=(means_np[i, 0], means_np[i, 1]),
            width=width,
            height=height,
            angle=angle,
            facecolor=face,
            edgecolor='none',
            alpha=a_draw
        )
        ax.add_patch(e)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
