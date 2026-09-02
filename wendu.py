import os
import json
import cv2
import numpy as np
import pandas as pd
import argparse
try:
    from skimage.filters import threshold_otsu
except ModuleNotFoundError:
    def threshold_otsu(image, nbins=256):
        image = np.asarray(image, dtype=np.float64).ravel()
        image = image[np.isfinite(image)]
        if image.size == 0:
            raise ValueError("Empty image")
        imin = float(np.min(image))
        imax = float(np.max(image))
        if imin == imax:
            return imin
        hist, bin_edges = np.histogram(image, bins=nbins, range=(imin, imax))
        hist = hist.astype(np.float64, copy=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        eps = np.finfo(np.float64).eps
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + eps)
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + eps))[::-1]
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        idx = int(np.argmax(variance12))
        return float(bin_centers[idx])
from pathlib import Path
import glob

def load_bounds(json_file):
    """Load the absolute temperature range for a scene."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Temperature range file not found: {json_file}")
    with open(json_file, 'r') as f:
        bounds = json.load(f)
    return bounds['absolute_min_temperature'], bounds['absolute_max_temperature']

def _read_ir_uint8(img_path: str):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            return img[:, :, 3]
        if img.shape[2] == 3:
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            if np.array_equal(b, g) and np.array_equal(b, r):
                return b
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None

def convert_images_to_csv(input_dir, output_dir, t_min, t_max):
    """Convert normalized thermal PNG images into temperature CSV matrices."""
    if not os.path.exists(input_dir):
        return False

    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

    if not image_files:
        return False

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        ir_img = _read_ir_uint8(img_path)

        if ir_img is None:
            continue

        ir_float = ir_img.astype(np.float32)
        temp_matrix = (ir_float / 255.0) * (t_max - t_min) + t_min

        csv_filename = os.path.splitext(img_file)[0] + ".csv"
        csv_path = os.path.join(output_dir, csv_filename)
        np.savetxt(csv_path, temp_matrix, delimiter=",", fmt="%.2f")

    return True

def _count_png_files(dir_path: str) -> int:
    try:
        return sum(1 for f in os.listdir(dir_path) if f.lower().endswith('.png'))
    except Exception:
        return 0

def load_csv_as_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy(dtype=np.float32)

def calculate_metrics(pred_dir, gt_dir, clip_min=-20.0):
    """Calculate image-averaged MAE and Otsu-foreground MAE."""
    if not os.path.exists(pred_dir):
        return None

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.csv')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.csv')])

    common_files = sorted(list(set(pred_files) & set(gt_files)))

    if not common_files:
        return None

    mae_list = []
    mae_roi_list = []

    for csv_file in common_files:
        try:
            p_path = os.path.join(pred_dir, csv_file)
            g_path = os.path.join(gt_dir, csv_file)

            pred = load_csv_as_matrix(p_path)
            gt = load_csv_as_matrix(g_path)

            pred = np.maximum(pred, clip_min)
            gt = np.maximum(gt, clip_min)

            abs_err = np.abs(pred - gt)
            mae_global = np.mean(abs_err)

            try:
                otsu_thresh = threshold_otsu(gt)
                roi_mask = gt > otsu_thresh
                if roi_mask.sum() > 0:
                    mae_roi = np.mean(abs_err[roi_mask])
                else:
                    mae_roi = mae_global
            except:
                mae_roi = mae_global

            mae_list.append(mae_global)
            mae_roi_list.append(mae_roi)

        except Exception as e:
            print(f"  Failed to process {csv_file}: {e}")

    return {
        "mean_mae": np.mean(mae_list),
        "mean_mae_roi": np.mean(mae_roi_list),
        "count": len(mae_list)
    }

def find_matching_data_folder(scene_name, data_root):
    candidates = os.listdir(data_root)
    if scene_name in candidates:
        return os.path.join(data_root, scene_name)

    best_match = None
    max_len = 0
    for cand in candidates:
        if cand in scene_name and len(cand) > max_len:
            best_match = cand
            max_len = len(cand)

    if best_match:
        return os.path.join(data_root, best_match)
    return None

def process_all(data_root, output_root):
    if not os.path.exists(output_root):
        print(f"Output root does not exist: {output_root}")
        return
    if not os.path.exists(data_root):
        print(f"Data root does not exist: {data_root}")
        return
    output_scenes = sorted([d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))])

    print(f"=== Batch evaluation (test split only) ===")
    print(f"Data Root: {data_root}")
    print(f"Output Root: {output_root}")
    print(f"Found {len(output_scenes)} output scenes.\n")

    results_summary = []

    for scene_name in output_scenes:
        scene_out_path = os.path.join(output_root, scene_name)

        scene_data_path = find_matching_data_folder(scene_name, data_root)

        if not scene_data_path:
            continue

        json_file = os.path.join(scene_data_path, 'temperature_bounds.json')
        if not os.path.exists(json_file):
            continue

        try:
            t_min, t_max = load_bounds(json_file)
        except Exception:
            continue

        print(f"[{scene_name}] Processing (temperature range: {t_min} to {t_max})")

        found_test_in_scene = False

        for root, dirs, files in os.walk(scene_out_path):
            if 'gt_ir' in dirs or 'gt' in dirs or 'gt_thermal' in dirs:
                # Require an explicit test-split directory component.
                path_parts = root.replace('\\', '/').split('/')

                if 'test' not in path_parts and 'thermal_test' not in path_parts:
                    continue

                experiment_base = root
                found_test_in_scene = True

                gt_ir_dir = os.path.join(experiment_base, 'gt_ir')
                gt_dir = os.path.join(experiment_base, 'gt')
                gt_thermal_dir = os.path.join(experiment_base, 'gt_thermal')
                gt_csv_dir = os.path.join(experiment_base, 'gt_csv')
                pred_csv_dir = os.path.join(experiment_base, 'renders_csv')
                renders_ir_dir = os.path.join(experiment_base, 'renders_ir')
                renders_dir = os.path.join(experiment_base, 'renders')
                renders_thermal_dir = os.path.join(experiment_base, 'renders_thermal')

                print(f"  -> Test experiment: {os.path.relpath(experiment_base, output_root)}")

                if os.path.exists(gt_ir_dir):
                    gt_img_dir = gt_ir_dir
                elif os.path.exists(gt_thermal_dir):
                    gt_img_dir = gt_thermal_dir
                else:
                    gt_img_dir = gt_dir

                gt_csv_ready = False
                if os.path.exists(gt_csv_dir):
                    gt_csv_ready = len(glob.glob(os.path.join(gt_csv_dir, "*.csv"))) > 0
                if not gt_csv_ready:
                    convert_images_to_csv(gt_img_dir, gt_csv_dir, t_min, t_max)
                    gt_csv_ready = len(glob.glob(os.path.join(gt_csv_dir, "*.csv"))) > 0
                if not gt_csv_ready:
                    if os.path.exists(gt_img_dir) and _count_png_files(gt_img_dir) == 0:
                        print(f"     [Notice] GT directory contains no PNG files: {os.path.relpath(gt_img_dir, output_root)}")
                    else:
                        print(f"     [Notice] No GT CSV files were generated: {os.path.relpath(gt_csv_dir, output_root)}")
                    continue

                pred_csv_ready = False
                if os.path.exists(pred_csv_dir):
                    pred_csv_ready = len(glob.glob(os.path.join(pred_csv_dir, "*.csv"))) > 0

                if not pred_csv_ready:
                    if os.path.exists(renders_ir_dir):
                        convert_images_to_csv(renders_ir_dir, pred_csv_dir, t_min, t_max)
                    elif os.path.exists(renders_thermal_dir):
                        convert_images_to_csv(renders_thermal_dir, pred_csv_dir, t_min, t_max)
                    elif os.path.exists(renders_dir):
                        convert_images_to_csv(renders_dir, pred_csv_dir, t_min, t_max)

                pred_csv_ready = len(glob.glob(os.path.join(pred_csv_dir, "*.csv"))) > 0 if os.path.exists(pred_csv_dir) else False
                if not pred_csv_ready:
                    cand_pred_dir = None
                    if os.path.exists(renders_ir_dir):
                        cand_pred_dir = renders_ir_dir
                    elif os.path.exists(renders_thermal_dir):
                        cand_pred_dir = renders_thermal_dir
                    elif os.path.exists(renders_dir):
                        cand_pred_dir = renders_dir
                    if cand_pred_dir is not None and _count_png_files(cand_pred_dir) == 0:
                        print(f"     [Notice] Prediction directory contains no PNG files: {os.path.relpath(cand_pred_dir, output_root)}")
                    else:
                        print(f"     [Notice] No prediction CSV files were generated: {os.path.relpath(pred_csv_dir, output_root)}")
                    continue

                metrics = calculate_metrics(pred_csv_dir, gt_csv_dir)

                if metrics:
                    print(f"     Result: MAE={metrics['mean_mae']:.4f} | ROI={metrics['mean_mae_roi']:.4f}")
                    results_summary.append({
                        "Scene": scene_name,
                        "Type": "test",
                        "Path": os.path.relpath(experiment_base, output_root),
                        "MAE": metrics['mean_mae'],
                        "MAE_roi": metrics['mean_mae_roi']
                    })

        if not found_test_in_scene:
            print(f"  [Notice] No test directory containing 'gt', 'gt_ir', or 'gt_thermal' was found in {scene_name}")

        print("-" * 40)

    if results_summary:
        print("\n=== Test evaluation summary (output order) ===")
        df_res = pd.DataFrame(results_summary)
        print(df_res[['Scene', 'MAE', 'MAE_roi']].to_string(index=False))

        save_path = os.path.join(output_root, "batch_test_evaluation_results.csv")
        df_res.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    else:
        print("No eligible test data were found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch thermal evaluation for Gaussian Splatting test results")
    project_root = Path(__file__).resolve().parent
    default_data_root = project_root / 'data'
    default_output_root = project_root / 'output'
    if not default_output_root.exists() and (project_root / 'out').exists():
        default_output_root = project_root / 'out'

    parser.add_argument('--data_root', type=str,
                        default=str(default_data_root),
                        help='Data root containing scene-level temperature_bounds.json files')

    parser.add_argument('--output_root', type=str,
                        default=str(default_output_root),
                        help='Output root containing experiment results')

    args = parser.parse_args()

    process_all(args.data_root, args.output_root)
