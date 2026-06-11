# Sapiens Adapter for Dataloop

A Dataloop adapter for Facebook's Sapiens human segmentation model. This adapter enables high-quality human body part segmentation within the Dataloop platform using GPU-accelerated inference.

## Overview

The Sapiens adapter integrates Facebook's Sapiens segmentation model (0.3B parameters) with Dataloop's ML platform. It performs pixel-level segmentation of human body parts and clothing items, producing detailed annotation masks for each detected class.

## Features

- **GPU-accelerated inference** - Optimized for NVIDIA T4 GPUs with CUDA support
- **28 segmentation classes** - Covers body parts, clothing, and facial features
- **TorchScript model** - Uses optimized TorchScript format for fast inference
- **Dataloop integration** - Native Dataloop BaseModelAdapter implementation
- **Auto-scaling** - Configurable RabbitMQ-based autoscaling for efficient resource utilization
- **Batch processing** - Supports batch inference on multiple items

## Segmentation Labels

The model segments the following 28 classes (excluding background):

- **Body Parts:** face_neck, hair, torso, left_foot, left_hand, left_lower_arm, left_lower_leg, left_upper_arm, left_upper_leg, right_foot, right_hand, right_lower_arm, right_lower_leg, right_upper_arm, right_upper_leg
- **Clothing:** apparel, lower_clothing, upper_clothing, left_shoe, right_shoe, left_sock, right_sock
- **Facial Features:** lower_lip, upper_lip, lower_teeth, upper_teeth, tongue

## Installation

### Prerequisites

- Dataloop account with GPU access
- NVIDIA GPU with CUDA 11.8 support
- Python 3.10+
- Docker

### Setup

1. Clone this repository
2. Ensure you have the required dependencies (see `requirements.txt`)
3. Build the Docker image:

```bash
docker build --no-cache -t sapiens-adapter:0.1.7 -f Dockerfile .
```

4. Push to your container registry:

```bash
docker push gcr.io/viewo-g/piper/agent/runner/gpu/sapiens:0.1.7
```

## Configuration

The adapter is configured via the `dataloop.json` manifest file:

### Model Configuration

- **weights_filename:** `sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2`
- **input_height:** 1024
- **input_width:** 768
- **batch_size:** 1
- **device:** cuda
- **person_label:** person

### Compute Configuration

- **GPU Type:** NVIDIA T4 (gpu-t4)
- **Concurrency:** 1
- **Autoscaling:** RabbitMQ-based (0-1 replicas, queue length threshold: 8)

## Usage

### As a Dataloop Service

Deploy the adapter as a Dataloop service using the `dataloop.json` manifest:

```python
import dtlpy as dl

# Load and deploy the adapter
adapter = dl.AdapterAdapter.from_json('adapters/seg/dataloop.json')
adapter.deploy()
```

### Prediction Function

The adapter exposes a `predict_items` function that:

1. Accepts an array of Dataloop items
2. Downloads and preprocesses images (BGR conversion, resizing, normalization)
3. Runs inference using the Sapiens model
4. Resizes segmentation masks back to original dimensions
5. Creates Dataloop segmentation annotations for each detected class

### Model Preprocessing

The adapter uses the exact preprocessing required by Sapiens:
- BGR color space conversion
- Normalization with mean `[103.53, 116.28, 123.675]` and std `[57.375, 57.12, 58.395]`
- Resizing to 768x1024 input dimensions
- Channel-first tensor format (CHW)

## Architecture

```
sapiens-adapter/
├── adapters/
│   └── seg/
│       ├── model_adapter.py    # Main adapter implementation
│       ├── labels.py           # Segmentation label definitions
│       └── dataloop.json       # Dataloop manifest
├── Dockerfile                  # GPU-enabled Docker image
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Model Information

- **Model:** Sapiens Segmentation 0.3B (Goliath variant)
- **Source:** Facebook Research
- **Format:** TorchScript (.pt2)
- **Input Size:** 768x1024
- **Parameters:** 0.3B
- **mIoU:** 76.73% (epoch 194)

## Dependencies

- PyTorch with CUDA 11.8 support
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Datalopy SDK

## License

This adapter uses Facebook's Sapiens model. Please refer to the original model license for usage terms.

## Contributing

Contributions are welcome. Please ensure any changes maintain compatibility with the Dataloop platform and the Sapiens model's preprocessing requirements.
