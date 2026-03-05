#!/bin/bash

CODE_PATH=/home/rdesc/scratch/SafeVLA
DATA_PATH=/home/rdesc/scratch/SafeVLA/data
WANDB_PROJECT=safety_chores
WANDB_ENTITY=rdesc1-milaquebec
OUTPUT_DIR=/home/rdesc/scratch/SafeVLA/data/models/
HF_HOME=/home/rdesc/scratch/huggingface

rm -f ~/.ai2thor/cuda-vulkan-mapping.json

module load apptainer/1.4.5

apptainer shell --nv --cleanenv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  --bind ${OUTPUT_DIR}:/root/output_dir \
  --env DISPLAY= \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH=/root/SafeVLA/allenact \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_ENTITY=${WANDB_ENTITY} \
  --env HF_HOME=/root/huggingface \
  --env OBJAVERSE_HOUSES_DIR=/root/data/objaverse_houses \
  --env OBJAVERSE_DATA_DIR=/root/data/objaverse_assets \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCH_CUDNN_V8_API_ENABLED=1 \
  safevla.sif

