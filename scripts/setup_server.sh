#!/bin/bash
# ============================================================
# DitHub + NCRN 服务器一键环境部署脚本
# 目标平台: Ubuntu 22.04 + CUDA 11.8 + RTX 4090
# ============================================================
set -e

echo "🚀 DitHub + NCRN 环境部署开始..."

# ---- 1. 创建 conda 环境 ----
echo "📦 [1/6] 创建 conda 环境 dithub (Python 3.9)..."
conda create -n dithub python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate dithub

# ---- 2. 安装 PyTorch (CUDA 11.8) ----
echo "🔥 [2/6] 安装 PyTorch + CUDA 11.8..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ---- 3. 安装项目依赖 ----
echo "📚 [3/6] 安装项目依赖..."
pip install transformers==4.55.2 \
    addict==2.4.0 \
    yapf==0.40.2 \
    timm==1.0.8 \
    numpy==1.26.3 \
    opencv-python==4.10.0.84 \
    supervision==0.3.2 \
    pillow==9.5.0 \
    scipy==1.13.1

# ---- 4. 安装 detectron2 ----
echo "🏗️ [4/6] 安装 detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2'

# ---- 5. 编译 Deformable-DETR CUDA 算子 ----
echo "⚙️ [5/6] 编译 Deformable-DETR CUDA 算子..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -d "${PROJECT_DIR}/Deformable-DETR" ]; then
    cd "$PROJECT_DIR"
    git clone https://github.com/fundamentalvision/Deformable-DETR.git
fi

cd "${PROJECT_DIR}/Deformable-DETR/models/ops"
sh ./make.sh
cd "$PROJECT_DIR"

# ---- 6. 下载预训练权重 ----
echo "📥 [6/6] 下载预训练权重..."
if [ ! -f "${PROJECT_DIR}/groundingdino_swint_ogc.pth" ]; then
    wget -q --show-progress -O "${PROJECT_DIR}/groundingdino_swint_ogc.pth" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
fi

# ---- 验证安装 ----
echo ""
echo "✅ 环境部署完成！验证中..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
import transformers, timm, detectron2
print(f'transformers: {transformers.__version__}')
print(f'timm: {timm.__version__}')
print(f'detectron2: {detectron2.__version__}')
print('🎉 All dependencies OK!')
"

echo ""
echo "============================================"
echo "  部署完成！下一步："
echo "  1. bash scripts/download_data.sh"
echo "  2. bash train_ncrn.sh"
echo "============================================"
