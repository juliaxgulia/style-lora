# Style LoRA for Off Label

This repo contains code and configs for training LoRA adapters on SDXL to support Off Label’s personalized virtual try-on and styling engine.

## Structure
- `train_lora_sdxl.py`: LoRA training script
- `config/`: LoRA hyperparameters
- `data/`: Sample image-caption dataset
- `scripts/`: Helper tools (e.g., upload to S3)
- `sagemaker/`: Launch SageMaker training jobs

## Setup
```bash
pip install -r requirements.txt
```
