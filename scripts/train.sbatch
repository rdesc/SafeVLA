#!/bin/bash
#SBATCH --job-name=safevla-train
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable

set -euo pipefail

EXP_NAME=vanilla_grpo_objnav_gen_4
OUTPUT_DIR=/root/data/models/  # path inside the container
CODE_PATH=/network/scratch/d/deschaer/SafeVLA/
DATA_PATH=/network/scratch/d/deschaer/SafeVLA/data
WANDB_PROJECT=safety_chores
WANDB_ENTITY=rdesc1-milaquebec
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch/hub}
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

rm -f ~/.ai2thor/cuda-vulkan-mapping.json

module load singularity

singularity exec --nv --cleanenv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  --bind ${TORCH_HOME}:/root/torch_cache \
  --env TORCH_HOME=/root/torch_cache \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_ENTITY=${WANDB_ENTITY} \
  --env HF_HOME=/root/huggingface \
  --env VK_ICD_FILENAMES=${VK_ICD_FILENAMES} \
  --env OBJAVERSE_HOUSES_DIR=/root/data/objaverse_houses \
  --env OBJAVERSE_DATA_DIR=/root/data/objaverse_assets \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_LAUNCH_BLOCKING=1 \
  --env TORCH_CUDNN_V8_API_ENABLED=1 \
  ${CODE_PATH}/safevla.sif \
  /bin/bash -lc "cd /root/SafeVLA && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh \
    --task_type objectnav \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir ${OUTPUT_DIR} \
    --dataset_dir /root/data/training_data/astar/ \
    --num_train_processes 16 \
    --tag ${EXP_NAME} \
    --max_houses 500 \
    --max_stage_steps 500"
