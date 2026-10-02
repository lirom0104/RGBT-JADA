#!/usr/bin/env bash
# Run the two camera-calibrated code paths concurrently:
#   GPU 0: this checkout (Our_Project-New-1)
#   GPU 1: the source checkout's camera-only configuration
# Each method processes both datasets sequentially.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_NEW1_ROOT="${METHOD_NEW1_ROOT:-$SCRIPT_DIR}"
METHOD_NEW2_ROOT="${METHOD_NEW2_ROOT:-/home/lf/code/Our_Project-New-2-source-before-cleanup-20260926_163758}"
PYTHON_BIN="${JADA_PYTHON:-/home/lf/miniconda3/envs/JADA/bin/python}"
RGBT_ROOT="${RGBT_ROOT:-/home/lf/data/thermal3dgs/RGBT-Scenes}"
THERMO_ROOT="${THERMO_ROOT:-/home/lf/data/ThermoScenes1_3dgs}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT_NEW1="${NEW1_OUTPUT_ROOT:-$METHOD_NEW1_ROOT/output/camera_only_${RUN_STAMP}}"
OUTPUT_ROOT_NEW2="${NEW2_OUTPUT_ROOT:-$METHOD_NEW2_ROOT/output/camera_only_${RUN_STAMP}}"
ITERATIONS="${ITERATIONS:-30000}"
GPU_NEW1="${GPU_NEW1:-0}"
GPU_NEW2="${GPU_NEW2:-1}"
PORT_NEW1="${PORT_NEW1:-6100}"
PORT_NEW2="${PORT_NEW2:-6200}"
DRY_RUN=0

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    echo "用法: $0 [--dry-run]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    DRY_RUN=1
fi

[[ -x "$PYTHON_BIN" ]] || { echo "找不到 Python: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$METHOD_NEW1_ROOT/train.py" ]] || { echo "New-1 缺少 train.py" >&2; exit 1; }
[[ -f "$METHOD_NEW2_ROOT/train.py" ]] || { echo "New-2 缺少 train.py" >&2; exit 1; }
[[ -d "$RGBT_ROOT" ]] || { echo "找不到数据集: $RGBT_ROOT" >&2; exit 1; }
[[ -d "$THERMO_ROOT" ]] || { echo "找不到数据集: $THERMO_ROOT" >&2; exit 1; }
if (( ! DRY_RUN )); then
    for output_root in "$OUTPUT_ROOT_NEW1" "$OUTPUT_ROOT_NEW2"; do
        if [[ -e "$output_root" ]]; then
            echo "输出目录已存在，为避免覆盖结果请指定新的 NEW1_OUTPUT_ROOT/NEW2_OUTPUT_ROOT: $output_root" >&2
            exit 1
        fi
    done
fi

check_dataset() {
    local root="$1" scene part
    local scenes=("$root"/*/)
    [[ -d "${scenes[0]}" ]] || { echo "数据集没有场景: $root" >&2; exit 1; }
    for scene in "${scenes[@]}"; do
        scene="${scene%/}"
        for part in sparse/0 rgb/train rgb/test thermal/train thermal/test; do
            [[ -d "$scene/$part" ]] || {
                echo "场景缺少目录: $scene/$part" >&2
                exit 1
            }
        done
    done
}

run_stage() {
    local root="$1" gpu="$2" port="$3" log="$4"
    shift 4
    if (( DRY_RUN )); then
        printf '[预览] (cd %q && CUDA_VISIBLE_DEVICES=%q %q' "$root" "$gpu" "$PYTHON_BIN"
        printf ' %q' "$@"
        printf ')\n'
        return
    fi
    mkdir -p "$(dirname -- "$log")"
    echo "[开始] $log"
    (
        cd "$root"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
            PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$@"
    ) 2>&1 | tee "$log"
    echo "[完成] $log"
}

run_method() {
    local label="$1" root="$2" gpu="$3" port="$4" mode="$5" output="$6"
    local dataset_label data_root scene name model_dir log
    local -a scenes train_args

    echo "=== $label：GPU $gpu；输出 $output ==="
    for dataset_label in RGBT-Scenes ThermoScenes1_3dgs; do
        if [[ "$dataset_label" == RGBT-Scenes ]]; then
            data_root="$RGBT_ROOT"
        else
            data_root="$THERMO_ROOT"
        fi
        mapfile -t scenes < <(find "$data_root" -mindepth 1 -maxdepth 1 -type d -print | sort)
        echo "[$label] $dataset_label：${#scenes[@]} 个场景"
        for scene in "${scenes[@]}"; do
            name="${scene##*/}"
            model_dir="$output/$dataset_label/$name"
            if [[ "$mode" == new1 ]]; then
                train_args=(
                    train.py -s "$scene" -m "$model_dir"
                    --port "$port" --iterations "$ITERATIONS"
                    --test_iterations 7000 15000 "$ITERATIONS"
                    --save_iterations "$ITERATIONS"
                    --checkpoint_iterations "$ITERATIONS"
                    --use_camera_calibration
                )
            else
                # Match New-2's camera-only experiment: keep the original
                # model, disable CWGC and the unrelated late-stage ablations.
                train_args=(
                    train.py -s "$scene" -m "$model_dir"
                    --port "$port" --iterations "$ITERATIONS"
                    --test_iterations 7000 15000 "$ITERATIONS"
                    --save_iterations "$ITERATIONS"
                    --checkpoint_iterations "$ITERATIONS"
                    --use_camera_calibration --no-use_cwgc
                    --late_rmse_weight 0 --late_lr_final_factor 1
                    --near_camera_prune_ratio 0
                    --no-cmo_use_thermal_densification
                    --no-cmo_preserve_rgb_densification
                )
            fi
            run_stage "$root" "$gpu" "$port" "$model_dir/.pipeline/train.log" "${train_args[@]}"
            run_stage "$root" "$gpu" "$port" "$model_dir/.pipeline/render.log" \
                render.py -m "$model_dir" --iteration "$ITERATIONS" --skip_train
            run_stage "$root" "$gpu" "$port" "$model_dir/.pipeline/metrics.log" \
                metrics.py -m "$model_dir"
            run_stage "$root" "$gpu" "$port" "$model_dir/.pipeline/extra_metrics.log" \
                extra_metrics.py -m "$model_dir"
        done
    done

    run_stage "$root" "$gpu" "$port" "$output/.pipeline/temperature.log" \
        wendu.py --data_root "$THERMO_ROOT" --output_root "$output/ThermoScenes1_3dgs"
    echo "=== $label 完成 ==="
}

check_dataset "$RGBT_ROOT"
check_dataset "$THERMO_ROOT"

if (( ! DRY_RUN )); then
    mkdir -p "$OUTPUT_ROOT_NEW1" "$OUTPUT_ROOT_NEW2"
fi
echo "GPU $GPU_NEW1：New-1；GPU $GPU_NEW2：New-2 camera-only；相机修正均开启。"
echo "New-1 输出：$OUTPUT_ROOT_NEW1"
echo "New-2 输出：$OUTPUT_ROOT_NEW2"

run_method "New-1-camera" "$METHOD_NEW1_ROOT" "$GPU_NEW1" "$PORT_NEW1" new1 \
    "$OUTPUT_ROOT_NEW1" &
PID_NEW1=$!
run_method "New-2-camera-only" "$METHOD_NEW2_ROOT" "$GPU_NEW2" "$PORT_NEW2" new2 \
    "$OUTPUT_ROOT_NEW2" &
PID_NEW2=$!

status=0
wait "$PID_NEW1" || status=1
wait "$PID_NEW2" || status=1
if (( status != 0 )); then
    echo "至少有一个方法运行失败，请查看各场景的 .pipeline/*.log。" >&2
    exit "$status"
fi
echo "两个方法、两个数据集全部完成。"
