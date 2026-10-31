from pathlib import Path
import torch

from A_stage1_reconstruct import run_stage1
from A_run_unified_training import run_v7_0_training, CONFIG


def main():
    # Single-image zero-shot dehazing
    image_path = Path("dataset/1/1.png")

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # Disable intermediate visualizations
    config = dict(CONFIG)
    config["VISUALIZATION"] = {
        "ENABLED": False,
        "INTERVAL": 5000
    }

    # Output directory
    output_dir = Path("checkpoints") / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = output_dir / "visualizations"

    print("=" * 60)
    print("Dehaze-GaussianImage")
    print("Input:", image_path)
    print("=" * 60)

    # Stage I: Explicit Hazy Scene Representation
    stage1_checkpoint = (
        output_dir / "stage1_reconstruct_model.pth.tar"
    )

    if not stage1_checkpoint.exists():
        print("\n[Stage I] Gaussian Reconstruction")
        run_stage1(
            image_path, output_dir, config, viz_dir
        )
    else:
        print("[Stage I] Existing checkpoint found.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage II: Physics-Disentangled Evolution
    print("\n[Stage II] Zero-Shot Dehazing")

    run_v7_0_training(
        image_path, output_dir, config, viz_dir
    )

    # Final output
    result_path = output_dir / "final_dehazed_result.png"

    if not result_path.is_file():
        raise RuntimeError(
            f"Final result was not generated: {result_path}"
        )

    print("\n" + "=" * 60)
    print("Dehazing completed successfully!")
    print("Final result:", result_path)
    print("=" * 60)


if __name__ == "__main__":
    main()