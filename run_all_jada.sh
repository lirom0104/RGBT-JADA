#!/usr/bin/env bash
# Train, render, and evaluate every scene in RGBT-Scenes and ThermoScenes1_3dgs.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/lf/miniconda3/etc/profile.d/conda.sh"
RGBT_ROOT="/home/lf/data/thermal3dgs/RGBT-Scenes"
THERMO_ROOT="/home/lf/data/ThermoScenes1_3dgs"
OUTPUT_ROOT="${JADA_OUTPUT_ROOT:-$PROJECT_ROOT/output/JADA_batch}"
ITERATION=30000
DRY_RUN=0

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
    echo "用法: $0 [--dry-run]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    DRY_RUN=1
fi

[[ -f "$CONDA_SH" ]] || { echo "找不到 Conda: $CONDA_SH" >&2; exit 1; }
# Activate once so every Python command uses the JADA environment.
set +u  # Some Conda CUDA activation hooks read unset variables.
source "$CONDA_SH"
conda activate JADA
set -u
PYTHON="$(command -v python)"
[[ "$PYTHON" == /home/lf/miniconda3/envs/JADA/bin/python ]] || {
    echo "Python 不属于 JADA 环境: $PYTHON" >&2
    exit 1
}
export CUDA_VISIBLE_DEVICES=0
cd "$PROJECT_ROOT"

verify_step() {
    local stage="$1" model_dir="$2" method="ours_$ITERATION" folder
    case "$stage" in
        train)
            [[ -s "$model_dir/point_cloud/iteration_$ITERATION/point_cloud.ply" ]]
            ;;
        render)
            for folder in renders_color gt_color renders_thermal gt_thermal; do
                compgen -G "$model_dir/test/$method/$folder/*.png" > /dev/null || return 1
            done
            ;;
        metrics)
            [[ -s "$model_dir/results.json" && -s "$model_dir/per_view.json" ]]
            "$PYTHON" - "$model_dir" "$method" <<'PY'
import json
import sys
from pathlib import Path

root, method = Path(sys.argv[1]), sys.argv[2]
for name in ("results.json", "per_view.json"):
    data = json.loads((root / name).read_text())
    if method not in data or not data[method]:
        raise SystemExit(f"{name} 缺少 {method} 的评估结果")
PY
            ;;
        extra_metrics)
            [[ -s "$model_dir/per_view_extra_metrics.json" ]]
            "$PYTHON" - "$model_dir" "$method" <<'PY'
import json
import sys
from pathlib import Path

root, method = Path(sys.argv[1]), sys.argv[2]
data = json.loads((root / "per_view_extra_metrics.json").read_text())
if method not in data or not data[method]:
    raise SystemExit(f"缺少 {method} 的额外评估结果")
PY
            ;;
        temperature)
            [[ -s "$model_dir/batch_test_evaluation_results.csv" ]]
            "$PYTHON" - "$THERMO_ROOT" "$model_dir/batch_test_evaluation_results.csv" <<'PY'
import csv
import sys
from pathlib import Path

data_root, csv_path = map(Path, sys.argv[1:])
expected = {p.name for p in data_root.iterdir() if p.is_dir()}
with csv_path.open(newline="") as stream:
    actual = {row["Scene"] for row in csv.DictReader(stream)}
missing = expected - actual
if missing:
    raise SystemExit(f"温度评估缺少场景: {', '.join(sorted(missing))}")
PY
            ;;
    esac
}

run_step() {
    local stage="$1" model_dir="$2"; shift 2
    local state_dir="$model_dir/.pipeline"
    local marker="$state_dir/$stage.ok"
    local log="$state_dir/$stage.log"

    if [[ -f "$marker" ]]; then
        if verify_step "$stage" "$model_dir"; then
            echo "[跳过] $model_dir / $stage（已完成）"
            return
        fi
        echo "[重试] $model_dir / $stage（完成标记存在，但结果缺失）"
        if (( ! DRY_RUN )); then
            rm -f "$marker"
        fi
    fi
    if (( DRY_RUN )); then
        printf '[预览]'
        printf ' %q' "$@"
        printf '\n'
        return
    fi

    mkdir -p "$state_dir"
    echo "[开始] $model_dir / $stage；日志: $log"
    if "$@" 2>&1 | tee "$log"; then
        if ! verify_step "$stage" "$model_dir"; then
            echo "[失败] $model_dir / $stage（命令成功，但未找到预期结果）；请查看 $log" >&2
            exit 1
        fi
        touch "$marker"
        echo "[完成] $model_dir / $stage"
    else
        local status=$?
        echo "[失败] $model_dir / $stage（退出码 $status），请查看 $log" >&2
        exit "$status"
    fi
}

check_scene() {
    local scene="$1" part
    for part in sparse/0 rgb/train rgb/test thermal/train thermal/test; do
        [[ -d "$scene/$part" ]] || {
            echo "场景缺少目录: $scene/$part" >&2
            exit 1
        }
    done
}

run_dataset() {
    local label="$1" data_root="$2" scene name model_dir
    [[ -d "$data_root" ]] || { echo "找不到数据集: $data_root" >&2; exit 1; }
    local scenes=("$data_root"/*/)
    [[ -d "${scenes[0]}" ]] || { echo "数据集没有场景: $data_root" >&2; exit 1; }

    for scene in "${scenes[@]}"; do
        check_scene "${scene%/}"
    done
    echo "=== $label：${#scenes[@]} 个场景 ==="
    for scene in "${scenes[@]}"; do
        scene="${scene%/}"
        name="${scene##*/}"
        model_dir="$OUTPUT_ROOT/$label/$name"

        run_step train "$model_dir" "$PYTHON" train.py \
            -s "$scene" -m "$model_dir" --iterations "$ITERATION"
        run_step render "$model_dir" "$PYTHON" render.py \
            -m "$model_dir" --iteration "$ITERATION" --skip_train
        run_step metrics "$model_dir" "$PYTHON" metrics.py -m "$model_dir"
        run_step extra_metrics "$model_dir" "$PYTHON" extra_metrics.py -m "$model_dir"
    done
}

run_dataset RGBT-Scenes "$RGBT_ROOT"
run_dataset ThermoScenes1_3dgs "$THERMO_ROOT"

# ThermoScenes provides temperature_bounds.json for temperature MAE/ROI MAE.
run_step temperature "$OUTPUT_ROOT/ThermoScenes1_3dgs" "$PYTHON" wendu.py \
    --data_root "$THERMO_ROOT" \
    --output_root "$OUTPUT_ROOT/ThermoScenes1_3dgs"

echo "全部完成。结果目录: $OUTPUT_ROOT"
