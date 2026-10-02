# JADA

## Installation

The provided environment uses Python 3.10, PyTorch 2.10.0, and CUDA 12.8. An NVIDIA GPU with a compatible driver is required.

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate JADA
```

The required CUDA extensions are included directly in this repository. Install them from the repository root after activating the Conda environment:

```bash
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
```

Verify the installation:

```bash
python -c "import torch; from simple_knn._C import distCUDA2; from diff_gaussian_rasterization import GaussianRasterizer; print('CUDA extensions: OK')"
python train.py --help
```

The first LPIPS use may download pretrained VGG and LPIPS weights to the PyTorch cache.

## Run the included example

The repository includes a ready-to-use RGB--thermal scene in `example_scene/`. Its training and test views have already been split, and the COLMAP reconstruction is provided in `example_scene/sparse/0/`.

### Training

Train the example scene for the default 30,000 iterations:

```bash
python train.py \
    -s example_scene \
    -m output/example_scene
```

The trained Gaussian model and configuration will be saved under `output/example_scene/`.

### Rendering

Render the test views from the saved 30,000-iteration model:

```bash
python render.py \
    -m output/example_scene \
    --iteration 30000 \
    --skip_train
```

Omit `--iteration 30000` to load the latest saved iteration automatically. Remove `--skip_train` to render both the training and test views.

### Evaluation

Compute PSNR, SSIM, and LPIPS for both RGB and thermal test images:

```bash
python metrics.py -m output/example_scene
```

Compute the thermal boundary F-score and hot-region IoU:

```bash
python extra_metrics.py -m output/example_scene
```

The aggregate metrics are written to `output/example_scene/results.json`. Per-view metrics are written to `per_view.json` and `per_view_extra_metrics.json`.

## Use your own scene

JADA expects a preprocessed COLMAP scene with paired RGB and thermal images:

```text
SceneName/
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
  rgb/
    train/
    test/
  thermal/
    train/
    test/
```

Text-form COLMAP files are also supported. RGB and thermal images within each split should have matching filenames. The loader first matches exact basenames, then normalized stems, and finally sorted indices as a fallback.

Train a custom scene with:

```bash
python train.py \
    -s /path/to/SceneName \
    -m output/SceneName
```

Camera calibration is enabled by default for COLMAP scenes. It uses the supplied
camera intrinsics for off-center principal points and lens distortion, and
supports PINHOLE, SIMPLE_PINHOLE, OPENCV, SIMPLE_RADIAL, and RADIAL models.
Use `--no-use_camera_calibration` to disable it. `render.py` reads the saved
setting from `cfg_args` and applies the same projection correction at evaluation
time. Scenes without COLMAP camera metadata keep their existing projection.

To run both code versions on both datasets concurrently, with one method per
GPU, use `./run_two_methods_two_gpus.sh`. GPU 0 runs this checkout and GPU 1
runs `Our_Project-New-2-source-before-cleanup-20260926_163758`; both use camera
calibration and CWGC is disabled. Results are written under each method's own
`output/` directory by default; use `NEW1_OUTPUT_ROOT` and `NEW2_OUTPUT_ROOT`
to choose custom locations, or pass `--dry-run` to inspect commands without
training.

Frequently used options include:

- `--iterations`: total training iterations.
- `--test_iterations`: iterations at which validation is run.
- `--save_iterations`: iterations at which Gaussian models are saved.
- `--checkpoint_iterations`: iterations at which resumable checkpoints are saved.
- `--start_checkpoint`: path to a checkpoint from which training is resumed.
- `--port`: network GUI port; the default is `6009`.

Render and evaluate the custom scene with:

```bash
python render.py -m output/SceneName --iteration 30000 --skip_train
python metrics.py -m output/SceneName
python extra_metrics.py -m output/SceneName
```

`render.py` reads the training configuration from `output/SceneName/cfg_args`. Multiple scenes can be evaluated in one command:

```bash
python metrics.py -m output/SceneA output/SceneB
python extra_metrics.py -m output/SceneA output/SceneB
```

## Temperature evaluation

For ThermoScenes outputs, compute temperature MAE and ROI MAE across all scenes with:

```bash
python wendu.py \
    --data_root /path/to/ThermoScenes \
    --output_root output/Thermoscenes
```

Each scene under `--data_root` must contain a `temperature_bounds.json` file with the absolute temperature range:

```json
{
  "absolute_min_temperature": -20.0,
  "absolute_max_temperature": 50.0
}
```

If `gt_csv` or `renders_csv` is absent, `wendu.py` linearly maps the corresponding 8-bit thermal PNG values to this temperature range and creates the CSV matrices. Existing CSV matrices are reused. Before evaluation, predicted and ground-truth temperatures below `-20.0` degrees Celsius are clipped to `-20.0`. `MAE` is the image-averaged global mean absolute error. `MAE_roi` is the image-averaged mean absolute error over pixels whose ground-truth temperature is above the per-image Otsu threshold.

The per-scene summary is written to:

```text
output/Thermoscenes/batch_test_evaluation_results.csv
```

## Output layout

A completed run has the following structure:

```text
output/SceneName/
  cfg_args
  cameras.json
  input.ply
  point_cloud/
    iteration_30000/
  train/
    ours_30000/
  test/
    ours_30000/
      renders_color/
      gt_color/
      renders_thermal/
      gt_thermal/
  results.json
  per_view.json
  per_view_extra_metrics.json
```

## Acknowledgements

This project builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Thermal Gaussian](https://github.com/chen-hangyu/Thermal-Gaussian-main). The vendored `simple-knn`, differentiable Gaussian rasterization, and GLM sources retain their original copyright and license notices.

## License

This repository inherits the non-commercial research and evaluation license of 3D Gaussian Splatting. See [LICENSE.md](LICENSE.md) and the license files included with the third-party components for details.
