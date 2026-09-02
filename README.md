# JADA

JADA is a joint RGB--thermal scene reconstruction system based on 3D Gaussian Splatting. This repository is developed from the OMMG branch of [Thermal Gaussian](https://github.com/chen-hangyu/Thermal-Gaussian-main) and includes the code required for training, rendering, and evaluating both RGB and thermal outputs.

## Installation

The released configuration has been tested with:

- Ubuntu 22.04
- NVIDIA GeForce RTX 5090
- GCC/G++ 11.4
- Python 3.10
- PyTorch 2.10.0
- CUDA 12.8

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

PyTorch normally detects the compute capability of the visible GPU automatically. If the extensions must be compiled without a visible GPU, set `TORCH_CUDA_ARCH_LIST` explicitly to the target GPU's compute capability.

If compilation selects an incompatible Conda compiler or inherits conflicting build flags, retry with the system compiler:

```bash
env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    python -m pip install --no-build-isolation ./submodules/simple-knn

env -u CFLAGS -u CXXFLAGS -u CPPFLAGS -u LDFLAGS \
    -u NVCC_PREPEND_FLAGS -u NVCC_APPEND_FLAGS \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
```

For the tested RTX 5090 configuration, `TORCH_CUDA_ARCH_LIST=12.0` can be added before `python` when an explicit architecture is required.

Verify the installation:

```bash
python -c "import torch; from simple_knn._C import distCUDA2; from diff_gaussian_rasterization import GaussianRasterizer; print('CUDA extensions: OK')"
python train.py --help
```

The first LPIPS use may download pretrained VGG and LPIPS weights to the PyTorch cache.

## Dataset layout

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

Datasets and pretrained models are not included in this repository.

## Training

Train one scene for the default 30,000 iterations:

```bash
python train.py \
    -s /path/to/SceneName \
    -m output/SceneName
```

Frequently used options include:

- `--iterations`: total training iterations.
- `--test_iterations`: iterations at which validation is run.
- `--save_iterations`: iterations at which Gaussian models are saved.
- `--checkpoint_iterations`: iterations at which resumable checkpoints are saved.
- `--start_checkpoint`: path to a checkpoint from which training is resumed.
- `--port`: network GUI port; the default is `6009`.

The main JADA components are enabled by default in `arguments/__init__.py`, including paired-view sampling, Gaussian binding, thermal residual geometry, render calibration, dual-modal refinement, adaptive branch weighting, joint anchor lifecycle management, and EMA export.

## Rendering

Render both the training and test views from the latest saved iteration:

```bash
python render.py -m output/SceneName
```

Render only the test split or select a specific iteration:

```bash
python render.py -m output/SceneName --skip_train
python render.py -m output/SceneName --iteration 30000 --skip_train
```

`render.py` reads the training configuration from `output/SceneName/cfg_args`.

## Evaluation

Compute PSNR, SSIM, and LPIPS for the rendered RGB and thermal test images:

```bash
python metrics.py -m output/SceneName
```

Multiple scenes can be evaluated together:

```bash
python metrics.py -m output/SceneA output/SceneB
```

The aggregate and per-view results are written to `results.json` and `per_view.json` in each model directory.

Compute the thermal boundary F-score and hot-region IoU from rendered and ground-truth thermal images:

```bash
python extra_metrics.py -m output/SceneName
```

Multiple scenes can be evaluated in one invocation:

```bash
python extra_metrics.py -m output/SceneA output/SceneB
```

`extra_metrics.py` appends `thermal_Boundary_Fscore` and `hot_region_IoU` to each scene's `results.json`. Per-view values are written to `per_view_extra_metrics.json`. By default, thermal edges and hot regions are selected at the 90th percentile. Boundary matching allows a one-pixel tolerance.

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

This project builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and the OMMG branch of [Thermal Gaussian](https://github.com/chen-hangyu/Thermal-Gaussian-main). The vendored `simple-knn`, differentiable Gaussian rasterization, and GLM sources retain their original copyright and license notices.

## License

This repository inherits the non-commercial research and evaluation license of 3D Gaussian Splatting. See [LICENSE.md](LICENSE.md) and the license files included with the third-party components for details.
