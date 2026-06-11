# Sapiens Adapter for Dataloop 🚀

Welcome to the Sapiens Adapter repository! 🎉 This adapter integrates Facebook's Sapiens human segmentation models (0.3B, 0.6B, 1B) with the [Dataloop](https://dataloop.ai/) platform. It enables high-quality GPU-accelerated pixel-level segmentation of human body parts and clothing items. 📚

## Introduction

This repo is a model integration between [Facebook's Sapiens](https://github.com/facebookresearch/sapiens) human segmentation models and [Dataloop](https://dataloop.ai/). The Sapiens models are designed for high-performance human body part segmentation, achieving state-of-the-art results in pixel-level annotation tasks. This integration provides a seamless workflow for utilizing Sapiens models within the Dataloop platform with GPU-accelerated inference.

## Requirements

* An account in the [Dataloop platform](https://console.dataloop.ai/)
* dtlpy
* torch
* opencv-python
* numpy
* Pillow

## Installation

To install the package and create the Sapiens model adapter, you will need a [project](https://developers.dataloop.ai/tutorials/getting_started/sdk_overview/chapter/#to-create-a-new-project) and a [dataset](https://developers.dataloop.ai/tutorials/data_management/manage_datasets/chapter/#create-dataset) in the Dataloop platform.

**Build Docker image:**
```bash
docker build --no-cache -t sapiens-adapter:latest -f Dockerfile .
```

## Cloning

For instruction how to clone the pretrained model for prediction, click [here](https://developers.dataloop.ai/tutorials/model_management/ai_library/chapter/#predicting)

## Segmentation Classes

The model segments the following 28 classes (excluding background):

**Body Parts:** face_neck, hair, torso, left_foot, left_hand, left_lower_arm, left_lower_leg, left_upper_arm, left_upper_leg, right_foot, right_hand, right_lower_arm, right_lower_leg, right_upper_arm, right_upper_leg

**Clothing:** apparel, lower_clothing, upper_clothing, left_shoe, right_shoe, left_sock, right_sock

**Facial Features:** lower_lip, upper_lip, lower_teeth, upper_teeth, tongue

## Configuration

The adapter is configured via `adapters/seg/dataloop.json`:

* **input_height:** 1024
* **input_width:** 768
* **batch_size:** 1
* **GPU Type:** NVIDIA T4 (gpu-t4)
* **Concurrency:** 1
* **Autoscaling:** RabbitMQ-based (0-1 replicas, queue length threshold: 8)

## Available Models

| Model | Parameters | mIoU | Epoch |
|-------|-----------|------|-------|
| 0.3B  | 0.3B      | 76.73% | 194 |
| 0.6B  | 0.6B      | 77.77% | 178 |
| 1B    | 1B        | 79.94% | 151 |

**Source:** Facebook Research (TorchScript format, 768x1024 input)

## Preprocessing

The adapter uses the exact preprocessing required by Sapiens:

* BGR color space conversion
* Normalization with mean `[103.53, 116.28, 123.675]` and std `[57.375, 57.12, 58.395]`
* Resizing to 768x1024 input dimensions
* Channel-first tensor format (CHW)

## Architecture

```
sapiens-adapter/
├── adapters/seg/
│   ├── model_adapter.py    # Main adapter with labels
│   └── dataloop.json       # Dataloop manifest
├── Dockerfile
└── README.md
```

## License

This adapter uses Facebook's Sapiens model. Please refer to the original model license for usage terms.
