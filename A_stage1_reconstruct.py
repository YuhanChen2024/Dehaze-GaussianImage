import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import math
from pytorch_msssim import ms_ssim
from gsplat2d.project_gaussians import project_gaussians
from gsplat2d.rasterize import rasterize_gaussians
# --- VIZ IMPORTS ---
from A_visualization_utils import save_tensor_as_image, plot_and_save_loss_curves, visualize_gaussians


# 损失函数
def loss_fn(pred, target, loss_type='L2', lambda_value=0.2):
    if loss_type == 'L2':
        return F.mse_loss(pred, target)
    elif loss_type == 'L1':
        return F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        return 1 - ms_ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'L1_SSIM':
        l1_loss = F.l1_loss(pred, target)
        ssim_loss = 1 - ms_ssim(pred, target, data_range=1, size_average=True)
        return (1 - lambda_value) * l1_loss + lambda_value * ssim_loss


class Gaussian2D(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_type = kwargs.get("loss_type", "L1_SSIM")
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.B_SIZE = 16
        self.device = kwargs["device"]

        w_init = torch.rand(self.init_num_points, 1, device=self.device) * self.W
        h_init = torch.rand(self.init_num_points, 1, device=self.device) * self.H
        self.means = nn.Parameter(torch.cat((w_init, h_init), dim=1))
        self.cov2d = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        self.rgbs = nn.Parameter(torch.zeros(self.init_num_points, 3, device=self.device))

        self.optimizer = torch.optim.Adam([self.rgbs, self.means, self.cov2d], lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7000, gamma=0.9)

    def forward(self):
        (xys, radii, conics, num_tiles_hit) = project_gaussians(self.cov2d, self.means, self.H, self.W, self.B_SIZE)
        colors_for_render = torch.sigmoid(self.rgbs)
        out_img = rasterize_gaussians(xys, radii, conics, num_tiles_hit, colors_for_render, self.H, self.W,
                                      self.B_SIZE)[..., :3]
        out_img = out_img.view(1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": torch.clamp(out_img, 0, 1), "colors": colors_for_render}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item()) if mse_loss.item() > 0 else 100.0
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss.item(), psnr, render_pkg


class LIG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_scales = kwargs["n_scales"]
        self.allo_ratio = kwargs.get("allo_ratio", 0.5)
        self.level_models = nn.ModuleList()
        self.store_min = []
        self.store_max = []

        total_points = kwargs["num_points"]
        if self.n_scales > 1:
            points_dist = [int(total_points * (1 - self.allo_ratio) ** (self.n_scales - 1 - s) * self.allo_ratio) for s
                           in range(self.n_scales - 1)]
            points_dist.append(total_points - sum(points_dist))
        else:
            points_dist = [total_points]

        for s in range(self.n_scales):
            H = int(kwargs["H"] * pow(2.0, -self.n_scales + s + 1))
            W = int(kwargs["W"] * pow(2.0, -self.n_scales + s + 1))
            num_points = points_dist[s]
            self.level_models.append(
                Gaussian2D(num_points=num_points, H=H, W=W, device=kwargs['device'], lr=kwargs['lr']))


def run_stage1(image_path: Path, log_dir: Path, config: dict, viz_dir: Path):
    print(f"--- [阶段1] 开始重建: {image_path.name} ---")
    viz_config = config.get("VISUALIZATION", {"ENABLED": False})
    if viz_config["ENABLED"]:
        print(f"--- [阶段1] 可视化已开启，将每隔 {viz_config['INTERVAL']} 次迭代保存一次。---")
        viz_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt_image = transforms.ToTensor()(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    H, W = gt_image.shape[2], gt_image.shape[3]

    model = LIG(
        num_points=config["NUM_POINTS"],
        n_scales=config["N_SCALES"],
        allo_ratio=config["ALLO_RATIO"],
        H=H, W=W,
        device=device,
        lr=config["RECONSTRUCT_LR"]
    ).to(device)

    start_time = time.time()
    im_estim = torch.zeros_like(gt_image)

    for scale_idx in range(model.n_scales):
        # --- VIZ SETUP ---
        scale_viz_dir = viz_dir / f"scale_{scale_idx}"
        loss_history = {"loss": [], "psnr": []}

        model.level_models[scale_idx].to(device)
        current_H = int(H * pow(2.0, -model.n_scales + scale_idx + 1))
        current_W = int(W * pow(2.0, -model.n_scales + scale_idx + 1))
        img_target_resized = F.interpolate(gt_image, size=(current_H, current_W), mode='area')

        if scale_idx == 0:
            img_target = img_target_resized
        else:
            im_estim_prev_resized = F.interpolate(im_estim, size=(current_H, current_W), mode='bilinear',
                                                  align_corners=False)
            residual = img_target_resized - im_estim_prev_resized
            store_min, store_max = torch.min(residual), torch.max(residual)
            if store_max - store_min > 1e-6:
                img_target = (residual - store_min) / (store_max - store_min)
            else:
                img_target = torch.zeros_like(residual)
            model.store_min.append(store_min)
            model.store_max.append(store_max)

        progress_bar = tqdm(range(config["RECONSTRUCT_ITERATIONS"]), desc=f"重建尺度 {scale_idx + 1}/{model.n_scales}")
        model.level_models[scale_idx].train()

        for iter_idx in progress_bar:
            loss, psnr, render_pkg = model.level_models[scale_idx].train_iter(img_target)
            progress_bar.set_postfix({"Loss": f"{loss:.6f}", "PSNR": f"{psnr:.2f}"})

            # --- VIZ LOGIC ---
            loss_history["loss"].append(loss)
            loss_history["psnr"].append(psnr)

            if viz_config["ENABLED"] and (
                    iter_idx % viz_config["INTERVAL"] == 0 or iter_idx == config["RECONSTRUCT_ITERATIONS"] - 1):
                rendered_image = render_pkg["render"]
                save_tensor_as_image(rendered_image, scale_viz_dir / f"iter_{iter_idx}_render.png")

                # 可视化高斯椭球
                current_model = model.level_models[scale_idx]
                visualize_gaussians(
                    current_model.means, current_model.cov2d, render_pkg["colors"],
                    current_H, current_W,
                    scale_viz_dir / f"iter_{iter_idx}_gaussians.png"
                )

        # --- VIZ: SAVE LOSS CURVE FOR THE SCALE ---
        if viz_config["ENABLED"]:
            plot_and_save_loss_curves(loss_history, f"阶段1 - 尺度 {scale_idx} 训练曲线",
                                      scale_viz_dir / "loss_curve.png")

        with torch.no_grad():
            model.level_models[scale_idx].eval()
            current_render = model.level_models[scale_idx]()["render"]
            if scale_idx == 0:
                im_estim = F.interpolate(current_render, size=(H, W), mode='bilinear', align_corners=False)
            else:
                store_min, store_max = model.store_min[-1], model.store_max[-1]
                residual_render = current_render * (store_max - store_min) + store_min
                residual_render_upsampled = F.interpolate(residual_render, size=(H, W), mode='bilinear',
                                                          align_corners=False)
                im_estim = im_estim + residual_render_upsampled
            im_estim = torch.clamp(im_estim, 0, 1).detach()
            model.level_models[scale_idx].to("cpu")
            torch.cuda.empty_cache()

    end_time = time.time()
    print(f"--- [阶段1] 重建完成, 耗时: {end_time - start_time:.2f}s ---")

    model_save_path = log_dir / "stage1_reconstruct_model.pth.tar"
    torch.save({
        'state_dicts': [m.state_dict() for m in model.level_models],
        'store_max': model.store_max,
        'store_min': model.store_min,
        'config': {
            'H': H, 'W': W,
            'n_scales': model.n_scales,
            'allo_ratio': model.allo_ratio,
            'num_points_total': config["NUM_POINTS"]
        }
    }, model_save_path)
    print(f"--- [阶段1] 模型已保存至: {model_save_path} ---")

    transform = transforms.ToPILImage()
    img = transform(im_estim.squeeze(0).cpu())
    img.save(log_dir / "stage1_reconstructed_image.png")
