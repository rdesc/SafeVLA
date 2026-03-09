#!/bin/bash
#SBATCH --job-name=safevla-constrained
#SBATCH --output=slurm_logs/slurm-constrained-%j.out
#SBATCH --error=slurm_logs/slurm-constrained-%j.err
#SBATCH -c 48
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:l40s:4
#SBATCH --partition=long

set -euo pipefail

OUTPUT_DIR=/network/scratch/d/deschaer/SafeVLA/data/models/
CODE_PATH=/network/scratch/d/deschaer/SafeVLA/
DATA_PATH=/network/scratch/d/deschaer/SafeVLA/data
WANDB_PROJECT=safety_chores
WANDB_ENTITY=rdesc1-milaquebec
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch/hub}
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
SINGULARITY_FILE=/network/scratch/d/deschaer/SafeVLA/safevla.sif

rm -f ~/.ai2thor/cuda-vulkan-mapping.json

module load singularity

singularity exec --nv --cleanenv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  --bind ${OUTPUT_DIR}:/root/output_dir \
  --bind ${TORCH_HOME}:/root/torch_cache \
  --env TORCH_HOME=/root/torch_cache \
  --env WANDB_PROJECT=${WANDB_PROJECT} \
  --env WANDB_ENTITY=${WANDB_ENTITY} \
  --env HF_HOME=/root/huggingface \
  --env VK_ICD_FILENAMES=${VK_ICD_FILENAMES} \
  --env OBJAVERSE_HOUSES_DIR=/root/data/objaverse_houses \
  --env OBJAVERSE_DATA_DIR=/root/data/objaverse_assets \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TORCH_CUDNN_V8_API_ENABLED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env ALLENACT_DEBUG=True \
  --env ALLENACT_DEBUG_VST_TIMEOUT=2000 \
  ${SINGULARITY_FILE} \
    /bin/bash -lc "cd /root/SafeVLA && PYTHONPATH=/root/SafeVLA/allenact:/root/SafeVLA/ CUDA_VISIBLE_DEVICES=0,1,2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 12 \
    --tag ObjNav_constrained_GRPO_64h_update2_entropy_0.003 \
    --max_steps 128 --num_steps_per_rollout 128 \
    --max_houses 1000 \
    --cost_limit 2.31964 --auto_resample_when_done False \
    --mask_out_other_rollouts True \
    --steps_in_house_before_force_scene_advance 1 \
    --use_grpo True --use_constraints True \
    --grpo_num_generations 3 --lr 1e-5 --gamma 0.99 \
    --shaping_weight 0.0 --goal_success_reward 10.0 \
    --save_interval 200000 \
    --update_repeats 2 \
    --grpo_advantage_clamp_min -0.1 \
    --wandb_project safety_chores --wandb_entity rdesc1-milaquebec \
    --callbacks wandb_logging_callback \
    --grpo_use_lambda True --grpo_lambda 0.98 --grpo_lambda_gamma 0.99 \
    --grpo_entropy_coef 0.003 \
    --constraint_names [corner,danger,blind,fragile,critical] \
    --constraint_thresholds [0.05,0.005,0.05,0.01,0.02]"
