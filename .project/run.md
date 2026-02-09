# For tracking commands used to run experiments


## Testing SafeVLA

What does training look like for the original SafeVLA (forked repo) with 1 example and 1 GPU?
```
PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True CUDA_VISIBLE_DEVICES=0 python3 training/online/dinov2_vits_tsfm_base.py train \
--il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
--output_dir /root/output_dir \
--dataset_dir /root/data/training_data/astar/ObjectNavType \
--num_train_processes 1 \
--tag ObjNav_debug_safevla_orig \
--max_steps 500 \
--num_steps_per_rollout 128 \
--max_houses 12 \
--auto_resample_when_done True \
--cost_limit 2.31964 \
--train_steps_value_network 1000 \
--mask_out_other_rollouts False \
--wandb_project $WANDB_PROJECT \
--wandb_entity $WANDB_ENTITY \
--use_grpo False \
--callbacks wandb_logging_callback 2>&1 | tee output.log
```

What happens if we increase the batch size?
```
PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=1,2 python3 training/online/dinov2_vits_tsfm_base.py train \
--il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
--output_dir /root/output_dir \
--dataset_dir /root/data/training_data/astar/ObjectNavType \
--num_train_processes 4 \
--tag ObjNav_debug_safevla_orig \
--max_steps 500 \
--num_steps_per_rollout 128 \
--max_houses 77 \
--auto_resample_when_done True \
--cost_limit 2.31964 \
--train_steps_value_network 1000 \
--mask_out_other_rollouts False \
--wandb_project $WANDB_PROJECT \
--wandb_entity $WANDB_ENTITY \
--use_grpo False \
--callbacks wandb_logging_callback 2>&1 | tee output2.log
```

Can we confirm that this looks good when we run the original repo


What happens if we increase the num steps per rollout?

Can we add the safety constraint in SafeVLA with a flag?



300 steps per rollout, 300 steps episode, masking, 4 houses, 1 GPU
```
PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True CUDA_VISIBLE_DEVICES=0 python3 training/online/dinov2_vits_tsfm_base.py train \
--il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
--output_dir /root/output_dir \
--dataset_dir /root/data/training_data/astar/ObjectNavType \
--num_train_processes 1 \
--tag ObjNav_debug_safevla_orig \
--max_steps 300 \
--num_steps_per_rollout 300 \
--max_houses 77 \
--steps_in_house_before_force_scene_advance 128 \
--auto_resample_when_done False \
--cost_limit 2.31964 \
--train_steps_value_network 10000 \
--mask_out_other_rollouts True \
--wandb_project $WANDB_PROJECT \
--wandb_entity $WANDB_ENTITY \
--callbacks wandb_logging_callback 2>&1 | tee output.log
```

300 steps per rollout, 300 steps episode, masking, 4 houses, 4 GPUs



300 steps per rollout, 300 steps episode, masking, 4 houses, 4 GPUs, constrained PPO



128 steps per rollout, 300 steps episode, 4 houses, 4 GPUs



128 steps per rollout, 300 steps episode, 4 houses, 4 GPUs, constrained PPO


## Testing Constrained GRPO