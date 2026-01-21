export CODE_PATH=/network/scratch/d/deschaer/SafeVLA/
export DATA_PATH=/network/scratch/d/deschaer/SafeVLA/data
export WANDB_PROJECT=safety_chores
export WANDB_ENTITY=rdesc1-milaquebec
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

rm ~/.ai2thor/cuda-vulkan-mapping.json # remove cache to fixe multi-gpu issues

module load singularity

singularity shell --nv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  safevla.sif


# debug 
# export PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH
