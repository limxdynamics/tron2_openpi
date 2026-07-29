# TRON2 OpenPI 安装指南

[English](INSTALL.md) | [README_CN](README_CN.md)

本文说明 `tron2_openpi` 以及同级 `tron2_env` 运行时包的环境要求和安装步骤。

## 环境要求

- 主要部署环境为 Ubuntu 22.04。
- Python 3.11 或更新版本。
- 使用 `uv` 管理 Python 依赖。
- 策略推理需要 NVIDIA GPU 和兼容 CUDA 的 JAX 环境。
- 客户端机器需要能访问 TRON2 WebSocket 机器人控制器。
- 使用 `client.observation_source: bridge` 时需要访问 TRON2 Bridge。
- 使用 `client.observation_source: legacy` 时需要 Intel RealSense 相机和本机相机访问权限。

## 安装 uv

如果系统中还没有 `uv`，可以安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果安装后 shell 仍找不到 `uv`，先更新 `PATH`：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## 克隆仓库

将本仓库和独立的 TRON2 运行时仓库 clone 到同级目录：

```bash
git clone https://github.com/limx-tron2/tron2_openpi.git
git clone https://github.com/limx-tron2/tron2_env.git

cd tron2_openpi
git checkout blackwell-pro5000
```

两个仓库需要保持同级：

```text
同级目录/
├── tron2_openpi/
└── tron2_env/
```

## 可选 FFmpeg 7 依赖

如果安装 `av` 依赖时失败，或者导入 `av` 时报 FFmpeg 动态库相关错误，先安装
FFmpeg 7 开发库：

```bash
sudo apt install pkg-config
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7 -y
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev -y
```

`av` 主要用于 LeRobot 视频数据集读取。只启动 policy server 做纯推理时通常不会
直接用到它，但默认环境会把它作为训练栈的一部分安装。

## 安装 Python 环境

先安装 `tron2_openpi` 的锁定依赖：

```bash
uv venv
uv sync --no-dev
```

## cuDNN 软链接

```bash
  CUDNN_LIB=.venv/lib/python3.11/site-packages/nvidia/cudnn/lib
  for so in $CUDNN_LIB/libcudnn*.so.9; do
      base=$(basename "$so" .so.9)
      ln -sf "$(basename "$so")" "$CUDNN_LIB/${base}.so.9.24.0"
      ln -sf "$(basename "$so")" "$CUDNN_LIB/${base}.so.9.24"
  done
```

然后安装同级 `tron2_env` 包：

```bash
source .venv/bin/activate
uv pip install -e "../tron2_env[bridge,openpi]"
```

## 验证安装

验证同级运行时能被导入：

```bash
PYTHONPATH="$(cd ../tron2_env/src && pwd)" uv run  --no-sync python -c "import tron2_env; print(tron2_env.__file__)"
```

输出路径应该指向 `../tron2_env/src/tron2_env/__init__.py`。

如果需要训练，也建议验证 LeRobot/PyAV 导入链路：

```bash
uv run --no-sync python - <<'PY'
import av
print("av:", av.__version__)

import lerobot.common.datasets.lerobot_dataset
print("lerobot dataset import ok")
PY
```

安装完成后，回到 [README_CN.md](README_CN.md) 查看部署配置和训练说明。
