#!/usr/bin/env bash
# Portable TRON2 training entrypoint for container and cloud platforms.

set -Eeuo pipefail

PROJECT_DIR="${OPENPI_PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${HF_LEROBOT_HOME:-/data}}"
WEIGHT_PATH="${WEIGHT_PATH:-/data/checkpoint/params}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"

REPO_ID=""
EXP_NAME=""
PROMPT=""
STEPS=20000
SAVE_INTERVAL=5000
BATCH_SIZE=32
ACTION_HORIZON=50
USE_DELTA=false
RTC_DELAY=""
PROMPT_FROM_TASK=false
MAX_FRAMES=""
SKIP_NORM=false
NAME_OVERRIDE=""
TASK_CONFIG=""
CROP_HIGH=""
CROP_LEFT_WRIST=""
CROP_RIGHT_WRIST=""
RESUME=false
OVERWRITE=false
USE_WANDB=false
DRY_RUN=false
GENERATED_TASK_CONFIG=false
ORIGINAL_ARGS=("$@")

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

log_error() {
  printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2
}

quote_args() {
  printf '%q ' "$@"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/cloud_train_entrypoint_portable.sh --repo-id ID --exp NAME [options]
  scripts/cloud_train_entrypoint_portable.sh --task-config FILE [--exp NAME] [options]

Default platform paths:
  dataset root:  --data-dir, DATA_DIR, or HF_LEROBOT_HOME; fallback /data
  model params:  /data/checkpoint/params
  output root:   /data/output

Options:
  --repo-id ID          LeRobot dataset ID below the dataset root
  --exp NAME            Experiment and checkpoint directory name
  --prompt TEXT         Task prompt; required unless --prompt-from-task is used
  --prompt-from-task    Use each LeRobot episode's task text as the prompt
  --steps N             Training steps (default: 20000)
  --save-interval N     Checkpoint interval (default: 5000)
  --batch-size N        Global batch size (default: 32)
  --action-horizon N    Action horizon (default: 50)
  --rtc-delay N         Enable training-time RTC with this simulated delay
  --delta               Train with delta joint actions
  --crop-high X,Y,W,H   Override high camera crop box before resize
  --crop-left-wrist X,Y,W,H  Override left wrist camera crop box before resize
  --crop-right-wrist X,Y,W,H Override right wrist camera crop box before resize
  --max-frames N        Maximum frames used for normalization statistics
  --skip-norm           Skip normalization-statistics computation
  --name NAME           Override the generated training config name
  --data-dir DIR        Override the LeRobot dataset root and HF_LEROBOT_HOME
  --weight PATH         Override the initialization params path
  --output-dir DIR      Override the output root
  --task-config FILE    Use an existing task YAML instead of generating one
  --resume              Resume an existing checkpoint; never overwrites it
  --overwrite           Replace an existing experiment checkpoint
  --wandb               Enable Weights & Biases (disabled by default)
  --dry-run             Validate inputs and print commands without training
  -h, --help            Show this help

Environment:
  OPENPI_AUTO_INSTALL_UV=0 disables the runtime uv install fallback.

Cloud/platform mount example:
  scripts/cloud_train_entrypoint_portable.sh \
    --repo-id input --exp example \
    --prompt "perform the configured task" \
    --steps 30000 --rtc-delay 10 --action-horizon 30 --max-frames 100000

Local custom paths example:
  scripts/cloud_train_entrypoint_portable.sh \
    --data-dir /path/to/datasets --repo-id my_dataset \
    --weight /path/to/checkpoint/params --output-dir /path/to/output \
    --exp example --prompt "perform the configured task" --max-frames 100000
EOF
}

die() {
  log_error "$*"
  exit 1
}

on_error() {
  local exit_code="$1"
  local line_no="$2"
  local command="$3"
  [ "$exit_code" -eq 0 ] && return
  log_error "command failed at line ${line_no} with exit code ${exit_code}: ${command}"
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [ -n "$value" ] || die "$option requires a value"
}

require_positive_integer() {
  local option="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$option must be a positive integer: $value"
}

require_nonnegative_integer() {
  local option="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$option must be a non-negative integer: $value"
}

print_command() {
  local label="$1"
  shift
  printf '%s' "$label"
  printf ' %q' "$@"
  printf '\n'
}

cleanup() {
  if [ "$GENERATED_TASK_CONFIG" = true ] && [ -n "$TASK_CONFIG" ]; then
    rm -f "$TASK_CONFIG"
  fi
}
trap cleanup EXIT
trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR

setup_runtime_path() {
  local home_dir="${HOME:-/root}"
  export PATH="/usr/local/bin:${home_dir}/.local/bin:/root/.local/bin:${PATH:-}"
}

log_runtime_context() {
  log "Invoked command: $0 $(quote_args "${ORIGINAL_ARGS[@]}")"
  log "Current user: $(id -un 2>/dev/null || printf unknown) uid=$(id -u 2>/dev/null || printf unknown) gid=$(id -g 2>/dev/null || printf unknown)"
  log "Initial working directory: $(pwd)"
  log "Runtime PATH: ${PATH}"
  log "Python candidate: $(command -v python3 2>/dev/null || command -v python 2>/dev/null || printf '<not found>')"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    local uv_path
    local uv_version
    uv_path="$(command -v uv)"
    uv_version="$(uv --version 2>&1)" || die "uv exists at ${uv_path}, but 'uv --version' failed: ${uv_version}"
    log "Found uv: ${uv_path}"
    log "uv version: ${uv_version}"
    return
  fi

  log "uv was not found on PATH; checking common install locations..."
  local candidate
  for candidate in \
    "/usr/local/bin/uv" \
    "${HOME:-/root}/.local/bin/uv" \
    "/root/.local/bin/uv"
  do
    if [ -x "$candidate" ]; then
      export PATH="$(dirname "$candidate"):$PATH"
      log "Found uv outside PATH: ${candidate}; updated PATH."
      local uv_version
      uv_version="$(uv --version 2>&1)" || die "uv exists at ${candidate}, but 'uv --version' failed: ${uv_version}"
      log "uv version: ${uv_version}"
      log "Runtime PATH after uv discovery: ${PATH}"
      return
    fi
  done

  if [ "${OPENPI_AUTO_INSTALL_UV:-1}" = "0" ]; then
    die "uv is not available and OPENPI_AUTO_INSTALL_UV=0; install uv into /usr/local/bin or add it to PATH before starting the job"
  fi

  command -v curl >/dev/null 2>&1 || die "uv is not available and curl is not installed; cannot run runtime uv install fallback"

  log "uv is not available; attempting runtime install to /usr/local/bin..."
  if curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh; then
    setup_runtime_path
  else
    log "Runtime install to /usr/local/bin failed; attempting user-local install under ${HOME:-/root}/.local/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv runtime install failed; rebuild the image with uv installed"
    setup_runtime_path
  fi

  if ! command -v uv >/dev/null 2>&1; then
    die "uv install fallback completed, but uv is still not on PATH: ${PATH}"
  fi

  local uv_path
  local uv_version
  uv_path="$(command -v uv)"
  uv_version="$(uv --version 2>&1)" || die "uv was installed at ${uv_path}, but 'uv --version' failed: ${uv_version}"
  log "Installed/found uv: ${uv_path}"
  log "uv version: ${uv_version}"
  log "Runtime PATH after uv setup: ${PATH}"
}

log_project_environment() {
  if [ -d ".venv" ]; then
    log "Project virtualenv found: ${PROJECT_DIR}/.venv"
    if [ -x ".venv/bin/python" ]; then
      log "Project venv Python: $(.venv/bin/python --version 2>&1)"
    fi
  else
    log "Project virtualenv not found at ${PROJECT_DIR}/.venv; 'uv run' may create or sync it at runtime."
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-id) require_value "$1" "${2:-}"; REPO_ID="$2"; shift 2 ;;
    --exp) require_value "$1" "${2:-}"; EXP_NAME="$2"; shift 2 ;;
    --prompt) require_value "$1" "${2:-}"; PROMPT="$2"; shift 2 ;;
    --steps) require_value "$1" "${2:-}"; STEPS="$2"; shift 2 ;;
    --save-interval) require_value "$1" "${2:-}"; SAVE_INTERVAL="$2"; shift 2 ;;
    --batch-size) require_value "$1" "${2:-}"; BATCH_SIZE="$2"; shift 2 ;;
    --action-horizon) require_value "$1" "${2:-}"; ACTION_HORIZON="$2"; shift 2 ;;
    --rtc-delay) require_value "$1" "${2:-}"; RTC_DELAY="$2"; shift 2 ;;
    --delta) USE_DELTA=true; shift ;;
    --crop-high) require_value "$1" "${2:-}"; CROP_HIGH="$2"; shift 2 ;;
    --crop-left-wrist) require_value "$1" "${2:-}"; CROP_LEFT_WRIST="$2"; shift 2 ;;
    --crop-right-wrist) require_value "$1" "${2:-}"; CROP_RIGHT_WRIST="$2"; shift 2 ;;
    --prompt-from-task) PROMPT_FROM_TASK=true; shift ;;
    --max-frames) require_value "$1" "${2:-}"; MAX_FRAMES="$2"; shift 2 ;;
    --skip-norm) SKIP_NORM=true; shift ;;
    --name) require_value "$1" "${2:-}"; NAME_OVERRIDE="$2"; shift 2 ;;
    --data-dir) require_value "$1" "${2:-}"; DATA_DIR="$2"; shift 2 ;;
    --weight) require_value "$1" "${2:-}"; WEIGHT_PATH="$2"; shift 2 ;;
    --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
    --task-config) require_value "$1" "${2:-}"; TASK_CONFIG="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    --overwrite) OVERWRITE=true; shift ;;
    --wandb) USE_WANDB=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

require_positive_integer --steps "$STEPS"
require_positive_integer --save-interval "$SAVE_INTERVAL"
require_positive_integer --batch-size "$BATCH_SIZE"
require_positive_integer --action-horizon "$ACTION_HORIZON"
[ -z "$MAX_FRAMES" ] || require_positive_integer --max-frames "$MAX_FRAMES"
[ -z "$RTC_DELAY" ] || require_nonnegative_integer --rtc-delay "$RTC_DELAY"

parse_crop_box() {
  local option="$1"
  local value="$2"
  local name="$3"
  local -a parts=()
  IFS=',' read -r -a parts <<< "$value"
  [ "${#parts[@]}" -eq 4 ] || die "$option must be in the form X,Y,W,H"
  for part in "${parts[@]}"; do
    [[ "$part" =~ ^[0-9]+$ ]] || die "$option must contain non-negative integers: $value"
  done
  printf '%s' "$name"
}

for crop_arg in "--crop-high:$CROP_HIGH" "--crop-left-wrist:$CROP_LEFT_WRIST" "--crop-right-wrist:$CROP_RIGHT_WRIST"; do
  option="${crop_arg%%:*}"
  value="${crop_arg#*:}"
  if [ -n "$value" ]; then
    case "$option" in
      --crop-high) parse_crop_box "$option" "$value" "cam_high" >/dev/null ;;
      --crop-left-wrist) parse_crop_box "$option" "$value" "cam_left_wrist" >/dev/null ;;
      --crop-right-wrist) parse_crop_box "$option" "$value" "cam_right_wrist" >/dev/null ;;
    esac
  fi
done

[ "$RESUME" = false ] || [ "$OVERWRITE" = false ] || die "--resume and --overwrite are mutually exclusive"

cd "$PROJECT_DIR"
setup_runtime_path
log_runtime_context
log "Project directory: ${PROJECT_DIR}"
log "Dataset root: ${DATA_DIR}"
log "Model params: ${WEIGHT_PATH}"
log "Output root: ${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR/logs" || die "cannot create output directory: $OUTPUT_DIR"

if [ -n "$TASK_CONFIG" ]; then
  [ -r "$TASK_CONFIG" ] || die "task config is not readable: $TASK_CONFIG"
  EXP_NAME="${EXP_NAME:-$(basename "$TASK_CONFIG")}"
  EXP_NAME="${EXP_NAME%.yaml}"
  EXP_NAME="${EXP_NAME%.yml}"
else
  [ -n "$REPO_ID" ] || die "--repo-id is required unless --task-config is used"
  [ -n "$EXP_NAME" ] || die "--exp is required unless --task-config is used"
  [ "$PROMPT_FROM_TASK" = false ] || [ -z "$PROMPT" ] || die "--prompt cannot be used with --prompt-from-task"
  [ "$PROMPT_FROM_TASK" = true ] || [ -n "$PROMPT" ] || die "--prompt is required unless --prompt-from-task or --task-config is used"
  [[ "$REPO_ID" != /* ]] || die "--repo-id must be relative to --data-dir"
  case "/$REPO_ID/" in
    */../*|*/./*) die "--repo-id must not contain '.' or '..' path segments" ;;
  esac

  DATASET_ROOT="$DATA_DIR/$REPO_ID"
  log "Checking dataset directory: ${DATASET_ROOT}"
  [ -d "$DATASET_ROOT/data" ] || die "dataset data directory is missing: $DATASET_ROOT/data"
  [ -d "$DATASET_ROOT/meta" ] || die "dataset metadata directory is missing: $DATASET_ROOT/meta"
  log "Checking model params: ${WEIGHT_PATH}"
  [ -r "$WEIGHT_PATH" ] || die "model params are not readable: $WEIGHT_PATH"

  NAME="${NAME_OVERRIDE:-pi05_tron2_${EXP_NAME%%_*}}"
  [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "generated config name contains unsupported characters: $NAME"

  TASK_CONFIG="$(mktemp "${TMPDIR:-/tmp}/tron2_train_task.XXXXXX.yaml")"
  GENERATED_TASK_CONFIG=true

  if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_COMMAND=("$PYTHON_BIN")
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_COMMAND=(python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_COMMAND=(python)
  else
    die "python3 or python is required to generate the task config"
  fi
  log "Using Python for task config generation: ${PYTHON_COMMAND[*]}"

  "${PYTHON_COMMAND[@]}" - \
    "$TASK_CONFIG" "$NAME" "$REPO_ID" "$PROMPT" "$WEIGHT_PATH" \
    "$STEPS" "$SAVE_INTERVAL" "$BATCH_SIZE" "$ACTION_HORIZON" \
    "$USE_DELTA" "$PROMPT_FROM_TASK" "$RTC_DELAY" "$OUTPUT_DIR" "$CROP_HIGH" "$CROP_LEFT_WRIST" "$CROP_RIGHT_WRIST" <<'PY'
import json
import pathlib
import sys

(
    output_path,
    name,
    repo_id,
    prompt,
    weight_path,
    steps,
    save_interval,
    batch_size,
    action_horizon,
    use_delta,
    prompt_from_task,
    rtc_delay,
    output_dir,
    crop_high,
    crop_left_wrist,
    crop_right_wrist,
) = sys.argv[1:]


def parse_crop_box(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"crop box must be in the form X,Y,W,H: {value}")
    x, y, w, h = (int(part) for part in parts)
    return {"x": x, "y": y, "w": w, "h": h}


config = {
    "name": name,
    "repo_id": repo_id,
    "prompt_from_task": prompt_from_task == "true",
    "weight_loader": weight_path,
    "num_train_steps": int(steps),
    "save_interval": int(save_interval),
    "batch_size": int(batch_size),
    "action_horizon": int(action_horizon),
    "use_delta_joint_actions": use_delta == "true",
    "checkpoint_base_dir": output_dir,
    "assets_base_dir": str(pathlib.Path(output_dir) / "assets"),
}
if prompt:
    config["prompt"] = prompt
if rtc_delay:
    config["rtc_training_simulated_delay"] = int(rtc_delay)

crop_config = {}
for key, value in (
    ("observation.images.cam_high", crop_high),
    ("observation.images.cam_left_wrist", crop_left_wrist),
    ("observation.images.cam_right_wrist", crop_right_wrist),
):
    crop_box = parse_crop_box(value)
    if crop_box is not None:
        crop_config[key] = crop_box
if crop_config:
    config["crop_config"] = crop_config

with pathlib.Path(output_path).open("w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)
    file.write("\n")
PY
fi

[[ "$EXP_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "experiment name contains unsupported characters: $EXP_NAME"

export HF_LEROBOT_HOME="$DATA_DIR"
if [ "$USE_WANDB" = false ]; then
  export WANDB_MODE=disabled
  log "Weights & Biases disabled: WANDB_MODE=disabled"
else
  log "Weights & Biases enabled by --wandb"
fi

ensure_uv
log_project_environment

NORM_COMMAND=(uv run scripts/compute_norm_stats.py --task-config "$TASK_CONFIG")
[ -z "$MAX_FRAMES" ] || NORM_COMMAND+=(--max-frames "$MAX_FRAMES")

TRAIN_COMMAND=(uv run scripts/train_tron2_task.py --task-config "$TASK_CONFIG" --exp-name "$EXP_NAME")
if [ "$RESUME" = true ]; then
  TRAIN_COMMAND+=(--resume)
elif [ "$OVERWRITE" = true ]; then
  TRAIN_COMMAND+=(--overwrite)
fi
[ "$USE_WANDB" = true ] || TRAIN_COMMAND+=(--no-wandb-enabled)

printf 'Project directory: %s\n' "$PROJECT_DIR"
printf 'Dataset root:     %s\n' "$DATA_DIR"
printf 'Model params:     %s\n' "$WEIGHT_PATH"
printf 'Output root:      %s\n' "$OUTPUT_DIR"
printf 'Task config:      %s\n' "$TASK_CONFIG"
cat "$TASK_CONFIG"
print_command 'Norm command:' "${NORM_COMMAND[@]}"
print_command 'Train command:' "${TRAIN_COMMAND[@]}"

if [ "$DRY_RUN" = true ]; then
  log "Dry run complete; training was not started."
  exit 0
fi

LOG_FILE="$OUTPUT_DIR/logs/training_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
log "Training log file: ${LOG_FILE}"
if [ "$SKIP_NORM" = false ]; then
  log "Computing normalization statistics..."
  print_command 'Running norm command:' "${NORM_COMMAND[@]}" | tee -a "$LOG_FILE"
  "${NORM_COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
  log "Normalization statistics finished."
else
  log "Skipping normalization statistics because --skip-norm was provided."
fi

log "Starting training..."
print_command 'Running train command:' "${TRAIN_COMMAND[@]}" | tee -a "$LOG_FILE"
XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"

log "Training complete. Checkpoints: ${OUTPUT_DIR}/<config-name>/${EXP_NAME}/<step>/"
