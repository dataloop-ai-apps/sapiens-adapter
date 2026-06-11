# Sapiens Model Adapter

## Introduction

This repository provides an integration between the Sapiens segmentation models and the [Dataloop](https://dataloop.ai/) platform.

Sapiens models are high-quality semantic segmentation models designed for detailed human parsing. They provide fine-grained predictions for body parts, clothing, and facial components.

This adapter connects Sapiens models to the Dataloop ecosystem, enabling streamlined inference and deployment workflows for segmentation tasks.

## Why Sapiens?

Sapiens models provide strong capabilities for human-centric segmentation tasks:

- **Fine-grained segmentation**: Detailed labels for body parts, clothing, and facial features  
- **Multiple model sizes**: Choose between performance and accuracy (0.3B / 0.6B / 1B)  
- **TorchScript support**: Optimized for efficient deployment  
- **Scalable inference**: Runs on GPU infrastructure within Dataloop  

## Requirements

- dtlpy  
- torch  
- torchvision  
- numpy  
- An account on the [Dataloop platform](https://console.dataloop.ai/)

## Installation

To use this adapter, make sure you have a project and a dataset in your Dataloop account.

### Docker Build

```bash
# Build the Docker image
podman build -t sapiens-adapter:0.0.7 .
```