#!/bin/bash

CODE_PATH=/network/scratch/d/deschaer/SafeVLA/
DATA_PATH=/network/scratch/d/deschaer/SafeVLA/data
WANDB_PROJECT=safety_chores
WANDB_ENTITY=rdesc1-milaquebec
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
TORCH_HOME=/home/mila/d/deschaer/.cache/torch/hub
OUTPUT_DIR=/network/scratch/d/deschaer/SafeVLA/data/models/

rm ~/.ai2thor/cuda-vulkan-mapping.json # remove cache to fixe multi-gpu issues

module load singularity

singularity shell --nv --cleanenv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  --bind ${TORCH_HOME}:/root/torch_cache \
  --bind ${OUTPUT_DIR}:/root/output_dir \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_ENTITY=${WANDB_ENTITY} \
  --env VK_ICD_FILENAMES=${VK_ICD_FILENAMES} \
  --env TORCH_HOME=/root/torch_cache \
  --env HF_HOME=/root/huggingface \
  --env OBJAVERSE_HOUSES_DIR=/root/data/objaverse_houses \
  --env OBJAVERSE_DATA_DIR=/root/data/objaverse_assets \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCH_CUDNN_V8_API_ENABLED=1 \
  --env PYTHONNOUSERSITE=1 \
  /network/scratch/d/deschaer/SafeVLA/safevla.sif

