# SafeVLA setup on Mila cluster with singularity

## Installation
Clone repo
```bash
git clone https://github.com/rdesc/SafeVLA
cd SafeVLA
```

Copy singularity image
```bash
cp /network/scratch/d/deschaer/SafeVLA/data/models/safevla.sif .
```

Set env variables
```bash
export CODE_PATH=path_to_repo_SafeVLA/  # edit this!!
export DATA_PATH=/network/scratch/d/deschaer/SafeVLA/data
export WANDB_PROJECT=safety_chores
export WANDB_ENTITY=rdesc1-milaquebec # edit this!!
```

## Launching singularity environment
Launch singularity module
```bash
module load singularity
```

Start singularity container
```bash
singularity shell --nv \
  --bind ${CODE_PATH}:/root/SafeVLA \
  --bind ${DATA_PATH}:/root/data \
  --bind ${HF_HOME}:/root/huggingface \
  safevla.sif
```

Or, launch the startup script (you'll need to update the env variables)
```bash
./startup.sh
```

Go to safevla root inside repo
```bash
cd /root/SafeVLA 
```

## Checkpoints provided by SafeVLA
1. spoc_IL ([FLaRe](https://robot-flare.github.io/) IL checkpoint)
    - Checkpoint: FLaRe_IL_50000.ckpt
    - Training Method: Imitation Learning (IL) - trained for 50,000 steps
    - Scope: Generalist model trained on demonstrations across multiple tasks
    - Purpose: Serves as the initialization for Safe RL fine-tuning
    - Path: `data/models/spoc_IL/model.ckpt`

2. Task-Specific FLaRe Models (fetch, pickup, objectnav)
    - Checkpoints:
      - FLaRe_fetch_sparse_reward_000047079268.pt
      - FLaRe_pickup_sparse_reward_000044088446.pt
      - FLaRe_objectnav_sparse_reward_000021026752.pt
    - Training Method: Reinforcement Learning (RL) with sparse rewards - trained for millions of steps
    - Scope: Specialist models - each fine-tuned on a specific task
    - Purpose: Task-specific baselines for comparison
    - Path: `data/models/{task}/model.ckpt`

3. Task-Specific SafeVLA  Models (fetch, pickup, objectnav)
    - Checkpoints: https://huggingface.co/SafetyEmbodiedAI/safety-model  
    - Path: `data/models/safe_{task}.pt`

## Demo scripts
Test eval script
```bash
bash /root/SafeVLA/scripts/eval.sh --task_type pickup --ckpt_path /root/data/models/spoc_IL/model.ckpt
```
Test train script
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  bash /root/SafeVLA/scripts/train.sh --task_type fetch --il_ckpt_path /root/data/models/spoc_IL/model.ckpt --output_dir /root/data/models/  --dataset_dir /root/data/training_data/astar/ --num_train_processes 4
```

## FLaRe's Role in SafeVLA
SafeVLA builds upon FLaRe by:
1. Using FLaRe's pretrained models as baselines
2. Adding safety constraints via constrained RL

The relationship:
```
FLaRe (Base System)
├── IL Training → FLaRe_IL_50000.ckpt (spoc_IL)
└── RL Training → Task-specific baselines (fetch, pickup, objectnav)
         ↓
    [SafeVLA adds]
         ↓
Safe RL Fine-tuning with cost constraints
         ↓
   SafeVLA (Safety-aligned models)
```

## Debug
Log files
- ~/.config/unity3d/Allen\ Institute\ for\ Artificial\ Intelligence/AI2-THOR/Player-prev.log
- ~/.ai2thor/log/unity.log

Adding breakpoints in code in singularity + torch.distributed:
```python
  from remote_pdb import set_trace
  set_trace()
```
After breakpoint is reached:
```bash
Ctrl-Z
bg
telnet IP PORT
```
Debug in pdb-like environment


## Rebuilding singularity image
```bash
sudo singularity build safevla.sif safevla.def
```

## SafeVLA changes to AllenAct
https://github.com/rdesc/allenact/commit/724c6ff35d6a8000083a8e6ea4ea447bc003a5ce#diff-649be34b9c4feca89c74a7a1a8d612ef927e820f9f4e4780b3195fa2d2eb17f5
