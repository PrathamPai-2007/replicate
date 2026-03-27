import json
from pathlib import Path

import torch

from dataset import LandslideTileDataset
from model import LandslideSSMoEModel


def compute_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> float:
    pred_mask = pred_mask.float()
    target_mask = target_mask.float()
    intersection = (pred_mask * target_mask).sum()
    union = ((pred_mask + target_mask) > 0).float().sum()
    return float((intersection + eps) / (union + eps))


def main() -> None:
    checkpoint_path = r"checkpoints\landslide_seg_dense_w8.pt"
    data_root = r"data_dense\val"
    threshold = 0.6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    dataset = LandslideTileDataset(
        task_type="segmentation",
        data_root=data_root,
    )

    model = LandslideSSMoEModel(
        in_channels=model_config["in_channels"],
        task_type=model_config["task_type"],
        dim=model_config["dim"],
        patch_size=model_config["patch_size"],
        specific_experts=model_config["specific_experts"],
        shared_experts=model_config["shared_experts"],
        top_k=model_config["top_k"],
        expert_hidden_dim=model_config["expert_hidden_dim"],
        dropout=model_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []
    ious = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target"].float()

            output = model(image)
            probability = torch.sigmoid(output.logits).cpu().squeeze(0)
            prediction = (probability >= threshold).float()

            if target.ndim == 2:
                target = target.unsqueeze(0)

            iou = compute_iou(prediction, target)
            ious.append(iou)

            results.append(
                {
                    "id": sample["id"],
                    "iou": iou,
                    "predicted_positive_pixels": int(prediction.sum().item()),
                    "target_positive_pixels": int(target.sum().item()),
                    "max_probability": float(probability.max().item()),
                }
            )

            print(f"{sample['id']}: IoU={iou:.4f}")

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    print(f"\nTotal samples: {len(results)}")
    print(f"Mean IoU: {mean_iou:.4f}")

    output_path = Path("predictions") / "all_samples_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_path,
                "data_root": data_root,
                "threshold": threshold,
                "mean_iou": mean_iou,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
