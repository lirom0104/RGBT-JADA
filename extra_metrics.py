"""Compute thermal boundary F-score and hot-region IoU from rendered images."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


BOUNDARY_FSCORE_KEY = "thermal_Boundary_Fscore"
HOT_REGION_IOU_KEY = "hot_region_IoU"


def read_gray_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (
        0.299 * array[..., 0]
        + 0.587 * array[..., 1]
        + 0.114 * array[..., 2]
    )


def sobel_magnitude(gray):
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gx = (
        padded[:-2, 2:]
        + 2.0 * padded[1:-1, 2:]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[1:-1, :-2]
        - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
        - padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def percentile_mask(values, percentile, positive_only=False):
    source = values[values > 1e-8] if positive_only else values
    if source.size == 0:
        return np.zeros_like(values, dtype=bool)
    threshold = np.percentile(source, percentile)
    return values >= threshold


def dilate(mask, radius):
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for row in range(size):
        for col in range(size):
            out |= padded[
                row : row + mask.shape[0],
                col : col + mask.shape[1],
            ]
    return out


def boundary_fscore(render_gray, gt_gray, edge_percentile, tolerance):
    render_edges = percentile_mask(
        sobel_magnitude(render_gray), edge_percentile, positive_only=True
    )
    gt_edges = percentile_mask(
        sobel_magnitude(gt_gray), edge_percentile, positive_only=True
    )

    render_count = int(render_edges.sum())
    gt_count = int(gt_edges.sum())
    if render_count == 0 and gt_count == 0:
        return 1.0
    if render_count == 0 or gt_count == 0:
        return 0.0

    render_hits = np.logical_and(render_edges, dilate(gt_edges, tolerance)).sum()
    gt_hits = np.logical_and(gt_edges, dilate(render_edges, tolerance)).sum()
    precision = render_hits / render_count
    recall = gt_hits / gt_count
    denominator = precision + recall
    return float(
        0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
    )


def hot_region_iou(render_gray, gt_gray, hot_percentile, threshold_mode):
    if threshold_mode == "gt":
        threshold = np.percentile(gt_gray, hot_percentile)
        render_hot = render_gray >= threshold
        gt_hot = gt_gray >= threshold
    else:
        render_hot = percentile_mask(render_gray, hot_percentile)
        gt_hot = percentile_mask(gt_gray, hot_percentile)

    intersection = np.logical_and(render_hot, gt_hot).sum()
    union = np.logical_or(render_hot, gt_hot).sum()
    return float(1.0 if union == 0 else intersection / union)


def evaluate_thermal_images(
    method_dir,
    edge_percentile,
    edge_tolerance,
    hot_percentile,
    hot_threshold_mode,
):
    renders_dir = method_dir / "renders_thermal"
    gt_dir = method_dir / "gt_thermal"
    image_names = sorted(
        fname
        for fname in os.listdir(renders_dir)
        if (renders_dir / fname).is_file() and (gt_dir / fname).is_file()
    )
    edge_scores = []
    hot_scores = []

    for fname in tqdm(
        image_names,
        desc=f"{method_dir.parent.name}/{method_dir.name} extra metrics",
    ):
        render_gray = read_gray_image(renders_dir / fname)
        gt_gray = read_gray_image(gt_dir / fname)
        edge_scores.append(
            boundary_fscore(
                render_gray,
                gt_gray,
                edge_percentile,
                edge_tolerance,
            )
        )
        hot_scores.append(
            hot_region_iou(
                render_gray,
                gt_gray,
                hot_percentile,
                hot_threshold_mode,
            )
        )

    mean_metrics = {
        BOUNDARY_FSCORE_KEY: float(np.mean(edge_scores)) if edge_scores else 0.0,
        HOT_REGION_IOU_KEY: float(np.mean(hot_scores)) if hot_scores else 0.0,
    }
    per_view_metrics = {
        BOUNDARY_FSCORE_KEY: dict(zip(image_names, edge_scores)),
        HOT_REGION_IOU_KEY: dict(zip(image_names, hot_scores)),
    }
    return mean_metrics, per_view_metrics


def merge_json(path, scene_data):
    existing = {}
    if path.exists():
        with open(path, "r") as fp:
            existing = json.load(fp)
    for method, metrics in scene_data.items():
        existing.setdefault(method, {}).update(metrics)
    with open(path, "w") as fp:
        json.dump(existing, fp, indent=True)


def evaluate(
    model_paths,
    split,
    method_filter,
    edge_percentile,
    edge_tolerance,
    hot_percentile,
    hot_threshold_mode,
    write_json,
):
    for model_path in model_paths:
        model_dir = Path(model_path)
        split_dir = model_dir / split
        if not split_dir.exists():
            print(f"[Skip] Missing split directory: {split_dir}")
            continue

        methods = (
            [method_filter]
            if method_filter
            else sorted(os.listdir(split_dir))
        )
        scene_results = {}
        scene_per_view = {}
        print(f"\nScene: {model_dir}")

        for method in methods:
            method_dir = split_dir / method
            if not method_dir.is_dir():
                continue

            image_metrics, per_view_metrics = evaluate_thermal_images(
                method_dir,
                edge_percentile,
                edge_tolerance,
                hot_percentile,
                hot_threshold_mode,
            )
            scene_results[method] = image_metrics
            scene_per_view[method] = per_view_metrics

            print(f"Method: {method}")
            print(
                "  thermal_Boundary_Fscore: {:>12.7f}".format(
                    image_metrics[BOUNDARY_FSCORE_KEY]
                )
            )
            print(
                "  hot_region_IoU         : {:>12.7f}".format(
                    image_metrics[HOT_REGION_IOU_KEY]
                )
            )

        if write_json:
            merge_json(model_dir / "results.json", scene_results)
            merge_json(
                model_dir / "per_view_extra_metrics.json",
                scene_per_view,
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Thermal boundary F-score and hot-region IoU metrics"
    )
    parser.add_argument(
        "--model_paths",
        "-m",
        required=True,
        nargs="+",
        type=str,
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--method", default=None)
    parser.add_argument("--edge_percentile", default=90.0, type=float)
    parser.add_argument("--edge_tolerance", default=1, type=int)
    parser.add_argument("--hot_percentile", default=90.0, type=float)
    parser.add_argument(
        "--hot_threshold_mode",
        default="self",
        choices=["self", "gt"],
    )
    parser.add_argument("--no_write_json", action="store_true")
    args = parser.parse_args()

    evaluate(
        args.model_paths,
        args.split,
        args.method,
        args.edge_percentile,
        args.edge_tolerance,
        args.hot_percentile,
        args.hot_threshold_mode,
        not args.no_write_json,
    )
