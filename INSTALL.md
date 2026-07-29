# TRON2 OpenPI Installation

[中文安装指南](INSTALL_CN.md) | [README](README.md)

This guide covers environment requirements and installation for `tron2_openpi`
and the sibling `tron2_env` runtime package.

## Requirements

- Ubuntu 22.04 is the primary deployment target.
- Python 3.11 or newer.
- `uv` for Python dependency management.
- NVIDIA GPU and CUDA-compatible JAX environment for policy inference.
- A reachable TRON2 WebSocket robot controller.
- TRON2 Bridge access when using `client.observation_source: bridge`.
- Intel RealSense cameras and local camera access when using
  `client.observation_source: legacy`.

## Install uv

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your shell cannot find `uv` after installation, update `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Clone Repositories

Clone this repository and the independent TRON2 runtime repository as siblings:

```bash
git clone https://github.com/limx-tron2/tron2_openpi.git
git clone https://github.com/limx-tron2/tron2_env.git

cd tron2_openpi
```

Keep the two repositories side by side:

```text
parent-directory/
├── tron2_openpi/
└── tron2_env/
```

## Optional FFmpeg 7 Dependencies

If installation fails while building or importing `av`, install FFmpeg 7
development libraries before syncing the Python environment:

```bash
sudo apt install pkg-config
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7 -y
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev -y
```

`av` is used by LeRobot video dataset loading. Policy serving without LeRobot
dataset loading usually does not use it directly, but the default environment
installs it as part of the training stack.

## Install Python Environment

Install the locked dependencies for `tron2_openpi`:

```bash
uv venv
uv sync --no-dev
```
## cuDNN symlinks

```bash
  CUDNN_LIB=.venv/lib/python3.11/site-packages/nvidia/cudnn/lib
  for so in $CUDNN_LIB/libcudnn*.so.9; do
      base=$(basename "$so" .so.9)
      ln -sf "$(basename "$so")" "$CUDNN_LIB/${base}.so.9.24.0"
      ln -sf "$(basename "$so")" "$CUDNN_LIB/${base}.so.9.24"
  done
```

Then install the sibling `tron2_env` package:

```bash
source .venv/bin/activate
uv pip install -e "../tron2_env[bridge,openpi]"
```

## Verify Installation

Verify that the sibling runtime can be imported:

```bash
PYTHONPATH="$(cd ../tron2_env/src && pwd)" uv run python -c "import tron2_env; print(tron2_env.__file__)"
```

The printed path should point to `../tron2_env/src/tron2_env/__init__.py`.

For training environments, also verify the LeRobot/PyAV import path:

```bash
uv run python - <<'PY'
import av
print("av:", av.__version__)

import lerobot.common.datasets.lerobot_dataset
print("lerobot dataset import ok")
PY
```

After installation, continue with deployment configuration in [README.md](README.md).
