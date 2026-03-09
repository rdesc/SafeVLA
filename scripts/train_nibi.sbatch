#!/bin/bash
#SBATCH --job-name=safevla-train
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#SBATCH -c 64
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --gres=gpu:h100:4

set -euo pipefail

EXP_NAME=debug_run_reward_shaping_tiny_train_set
CODE_PATH=/home/rdesc/scratch/SafeVLA
DATA_PATH=/home/rdesc/scratch/SafeVLA/data
OUTPUT_DIR=/home/rdesc/scratch/SafeVLA/data/models/
WANDB_PROJECT=safety_chores
WANDB_ENTITY=rdesc1-milaquebec
HF_HOME=/home/rdesc/scratch/huggingface
APPTAINER_FILE=/home/rdesc/scratch/SafeVLA/safevla.sif

rm -f ~/.ai2thor/cuda-vulkan-mapping.json

module load apptainer/1.4.5

apptainer exec --nv --cleanenv \
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
  --env ALLENACT_DEBUG=True \
  --env ALLENACT_DEBUG_VST_TIMEOUT=2000 \
  ${APPTAINER_FILE} \
  /bin/bash -lc "cd /root/SafeVLA && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train.sh \
  --visualize True --wandb_video_every_logs 50 --use_constraints True --save_interval 100000 --il_ckpt_path /root/data/models/spoc_IL/model.ckpt --output_dir /root/output_dir --dataset_dir /root/data/training_data/astar/ObjectNavType --num_train_processes 16 --tag ObjNav_debug_forked_safevla_4_gpu_all_houses_forked_allenact_200000_step_valuenet_128_rollout_128_max_steps_16_bs --max_steps 128 --num_steps_per_rollout 128 --metric_accumulate_interval 10 --cost_limit 2.31964 --train_steps_value_network 200000 --auto_resample_when_done True --mask_out_other_rollouts False --wandb_project safety_chores --wandb_entity rdesc1-milaquebec --use_grpo False --callbacks wandb_logging_callback"
