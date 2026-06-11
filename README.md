# Sapiens Adapter for Dataloop

Dataloop adapter for Facebook's Sapiens human segmentation models (0.3B, 0.6B, 1B). GPU-accelerated pixel-level segmentation of human body parts and clothing.

## Features

- GPU-accelerated inference (NVIDIA T4)
- 28 segmentation classes (body parts, clothing, facial features)
- TorchScript models for fast inference
- Native Dataloop BaseModelAdapter
- RabbitMQ-based autoscaling

## Segmentation Classes (28)

**Body Parts:** face_neck, hair, torso, left_foot, left_hand, left_lower_arm, left_lower_leg, left_upper_arm, left_upper_leg, right_foot, right_hand, right_lower_arm, right_lower_leg, right_upper_arm, right_upper_leg

**Clothing:** apparel, lower_clothing, upper_clothing, left_shoe, right_shoe, left_sock, right_sock

**Facial Features:** lower_lip, upper_lip, lower_teeth, upper_teeth, tongue

## Installation

**Prerequisites:** Dataloop account with GPU access, NVIDIA GPU (CUDA 11.8), Python 3.10+, Docker

**Build Docker image:**
```bash
docker build --no-cache -t sapiens-adapter:latest -f Dockerfile .
```

**Dependencies:** torch, opencv-python, numpy, Pillow, dtlpy

## Configuration

Configured via `adapters/seg/dataloop.json`:

**Model:** input_height=1024, input_width=768, batch_size=1
**Compute:** GPU T4, concurrency=1, autoscaling (0-1 replicas, queue threshold=8)

## Usage

```python
import dtlpy as dl

adapter = dl.AdapterAdapter.from_json('adapters/seg/dataloop.json')
adapter.deploy()
```

**Preprocessing:** BGR conversion, normalization (mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]), resize to 768x1024, CHW format

## Architecture

```
sapiens-adapter/
├── adapters/seg/
│   ├── model_adapter.py    # Main adapter with labels
│   └── dataloop.json       # Dataloop manifest
├── Dockerfile
└── README.md
```

## Available Models

| Model | Parameters | mIoU | Epoch |
|-------|-----------|------|-------|
| 0.3B  | 0.3B      | 76.73% | 194 |
| 0.6B  | 0.6B      | 77.77% | 178 |
| 1B    | 1B        | 79.94% | 151 |

**Source:** Facebook Research (TorchScript format, 768x1024 input)

**License:** Refer to Facebook's Sapiens model license for usage terms.
