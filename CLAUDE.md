# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DitHub is a modular framework for **Incremental Open-Vocabulary Object Detection** (NeurIPS 2025). It extends GroundingDINO with class-specific LoRA adaptation modules that can be incrementally trained and merged, inspired by version control systems.

## Key Commands

### Environment Setup
```bash
conda create -n dithub python=3.9 pip
conda activate dithub
pip install -r requirements.txt

# Compile Deformable-DETR CUDA operators
git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops && sh ./make.sh && python test.py
```

### Download Pre-trained Weights
```bash
# GroundingDINO base model (required for training)
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# DitHub trained model (for evaluation only)
wget -q https://github.com/chiara-cap/DitHub/releases/download/v.0.1/model_final.pth
```

### Training & Evaluation
```bash
# Training on ODinW-13
sh train_dithub.sh

# Evaluation
sh eval_dithub.sh
```

### Command-line Arguments (main.py)
| Argument | Description |
|----------|-------------|
| `--config-file` | Path to dataset config (e.g., `test/test_odinw13`) |
| `--model-config-file` | Model architecture config (e.g., `groundingdino/config/GroundingDINO_SwinT_OGC_dt_dithub.py`) |
| `--model-checkpoint-path` | Pre-trained weights path |
| `--output-dir` | Output directory for logs/checkpoints |
| `--dithub` | Enable DitHub LoRA adaptation |
| `--eval-only` | Run evaluation only |
| `--shuffle-tasks` | Randomize task order |
| `--lora-r`, `--lora-alpha`, `--lora-lr` | LoRA hyperparameters |

## Architecture Overview

### Core Components

```
groundingdino/
├── models/GroundingDINO/
│   ├── groundingdino_dt.py      # Main model class (GroundingDINO)
│   ├── transformer_for_adapter.py # Transformer with LoRA hooks
│   ├── modules/
│   │   ├── lora.py              # Base LoRA layer
│   │   ├── lora_pool.py         # LinearPool: per-class LoRA adapters
│   │   └── multi_head_attention.py # Custom attention for LoRA injection
│   └── criterion/               # Loss functions (SetCriterion)
├── util/
│   ├── lora_utils.py            # LoRA application & checkpointing
│   └── task_memory.py           # Singleton for managing task-specific params
└── config/
    ├── GroundingDINO_SwinT_OGC_dt_dithub.py  # Model config
    └── configs/common/data/odinw/            # Dataset configs
```

### Key Classes

1. **`GroundingDINO`** (`groundingdino_dt.py`): Main detection model extending GroundingDINO with:
   - `load_custom_attention()`: Replaces nn.MultiheadAttention with custom module for LoRA
   - `prompt_memory_pool`: nn.ParameterDict storing learned class embeddings
   - Two-phase training: warmup (task-agnostic) → specialization (per-class)

2. **`LinearPool`** (`lora_pool.py`): Per-class LoRA adapter layer:
   - `warmup_lora_a`: Shared warmup LoRA-A matrix
   - `per_class_lora_A`: nn.ParameterDict of class-specific LoRA-A matrices
   - `shared_lora_b`: Shared LoRA-B matrix across all classes

3. **`TaskMemory`** (`task_memory.py`): Singleton managing incremental learning:
   - `task_mapping`: Maps dataset names to class indices
   - `_modules`: Stores per-class LoRA parameters across tasks
   - `enable_per_class()`: Switches from warmup to specialization phase
   - `merge_B()`: Merges LoRA-B matrices with exponential moving average

### Training Flow

```
main.py
├── do_train()
│   ├── load_model() → Apply LoRA to selected layers
│   ├── Trainer.train()
│   │   ├── Phase 1 (warmup): Shared LoRA updates
│   │   └── Phase 2 (specialization): At iter=max_iter/2, enable_per_class()
│   └── TaskMemory.end_task() → Save per-class params
└── do_test() → Evaluate on each dataset, compute average AP
```

### LoRA Injection Points

LoRA is applied to linear layers in the transformer encoder/decoder (excluding backbone, bert, and feat_map). See `lora_utils.py:get_lora_modules()` for exclusion list.

## Dataset Structure

- **ODinW-13**: 13 datasets for incremental training (downloaded via `tools/download_odinw.py`)
- **ODinW-O**: Overlapping class variant (generated via `tools/overlapped_classes_dataset.py`)
- **COCO**: Zero-shot evaluation benchmark
- Place datasets in `datasets/` directory

## Config System

Uses detectron2's LazyConfig. Dataset configs in `test/test_odinw13/for_train/` define:
- `dataloader`: Dataset paths, batch size, augmentation
- `optimizer`: AdamW with layer-wise LR scaling
- `train`: Output dir, max iterations, checkpointing

## Checkpointing

- **During training**: Saves `lora_final.pth` per task in `output_dir/<task_name>/`
- **Final model**: `last_lora.pth` → merged into `model_final.pth`
- **Evaluation**: `DetectionLoraCheckpointer` handles LoRA-only state dicts