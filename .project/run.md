# For tracking commands used to run experiments



## Testing SafeVLA

1. What does training look like for the original SafeVLA (forked repo) with 1 example and 1 GPU?
    ```shell
    PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True CUDA_VISIBLE_DEVICES=0 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 1 \
    --tag ObjNav_debug_forked_safevla_1_gpu_1_house_forked_allenact_1000_step_valuenet_128_rollout \
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

2. What happens if we increase the batch size?
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=1,2 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_forked_safevla_2_gpu_4_house_forked_allenact_1000_step_valuenet_128_rollout \
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

3. How sensitive is training to the seed?
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=0,1 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_forked_safevla_2_gpu_4_house_forked_allenact_1000_step_valuenet_128_rollout_seed3 \
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

4. How does training look with 1 example when we run the original repo before I made all the changes
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 1 \
    --tag ObjNav_debug_orig_safevla_1_gpu_1_house_orig_allenact_1000_step_valuenet_128_rollout \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 12 \
    --cost_limit 2.31964 \
    --train_steps_value_network 1000 \
    --mask_out_other_rollouts False \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --callbacks wandb_logging_callback 2>&1 | tee output3.log
    ```

5. Try with larger batch size
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=0,1 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_1000_step_valuenet_128_rollout \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 77 \
    --cost_limit 2.31964 \
    --train_steps_value_network 1000 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --callbacks wandb_logging_callback 2>&1 | tee output4.log
    ```

6. Try with more training steps on value function
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_10000_step_valuenet_128_rollout \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 77 \
    --cost_limit 2.31964 \
    --train_steps_value_network 10000 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --callbacks wandb_logging_callback 2>&1 | tee output5.log
    ```

7. What happens if we increase the num steps per rollout (max_steps = 400 and num_steps_per_rollout = 400) keeping the batch size of 4?
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=0,1,2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_orig_safevla_4_gpu_4_house_orig_allenact_10000_step_valuenet_400_rollout \
    --max_steps 400 \
    --num_steps_per_rollout 400 \
    --max_houses 77 \
    --cost_limit 2.31964 \
    --train_steps_value_network 10000 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --callbacks wandb_logging_callback 2>&1 | tee output6.log
    ```

8. What happens if we resume training for the original repo with a checkpoint that was saved mid training
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=0,1 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_1000_step_valuenet_128_rollout \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 77 \
    --cost_limit 2.31964 \
    --train_steps_value_network 1000 \
    --checkpoint data/models/checkpoints/ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_1000_step_valuenet_128_rollout/2026-02-09_16-06-40/exp_ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_1000_step_valuenet_128_rollout__stage_01__steps_000000103424.pt \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --callbacks wandb_logging_callback 2>&1 | tee output4.log
    ```

9. Can we get learning to work with 1 example if we increase training steps for the value function?
    ```shell
    PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 CUDA_VISIBLE_DEVICES=2 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 1 \
    --tag ObjNav_debug_forked_safevla_1_gpu_1_house_forked_allenact_10000_step_valuenet_128_rollout \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 12 \
    --auto_resample_when_done True \
    --cost_limit 2.31964 \
    --train_steps_value_network 10000 \
    --mask_out_other_rollouts False \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --use_grpo False \
    --callbacks wandb_logging_callback 2>&1 | tee output.log
    ```

10. Is our main problem that we need a decent batch size for training to work? Here we're setting batch size to 2 per gpu, but we can increase to 4 per gpu
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=0,1,2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 8 \
    --tag ObjNav_debug_orig_safevla_4_gpu_8_house_orig_allenact_10000_step_valuenet_128_rollout_8_bs \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 136 \
    --cost_limit 2.31964 \
    --train_steps_value_network 10000 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --use_grpo False \
    --callbacks wandb_logging_callback
    ```

11. Increase batch size to 4 per gpu
    ```shell
    ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 PYTHONPATH=$PYTHONPATH:/root/SafeVLA CUDA_VISIBLE_DEVICES=0,1,2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 16 \
    --tag ObjNav_debug_orig_safevla_4_gpu_16_house_orig_allenact_10000_step_valuenet_128_rollout_16_bs \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 299 \
    --cost_limit 2.31964 \
    --train_steps_value_network 10000 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --use_grpo False \
    --callbacks wandb_logging_callback
    ```

12. Can we learn with autoresampling and masking set to true on the forked repo with constraint enabled?
    ```shell
    PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 12 \
    --tag ObjNav_debug_forked_safevla_4_gpu_16_house_forked_allenact_100000_step_valuenet_250_rollout_250_max_steps_12_bs_mask_out \
    --max_steps 250 \
    --num_steps_per_rollout 250 \
    --max_houses 299 \
    --cost_limit 2.31964 \
    --train_steps_value_network 100000 \
    --auto_resample_when_done False \
    --mask_out_other_rollouts True \
    --steps_in_house_before_force_scene_advance 1 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --use_grpo False \
    --callbacks wandb_logging_callback
    ```

13. Vanilla GRPO
    ```shell
    PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 CUDA_VISIBLE_DEVICES=0,1 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_vanilla_GRPO_4_gpu_16_house_250_rollout_250_max_steps_24_bs_6_groupsize \
    --max_steps 250 \
    --num_steps_per_rollout 250 \
    --max_houses 299 \
    --cost_limit 2.31964 \
    --auto_resample_when_done False \
    --mask_out_other_rollouts True \
    --steps_in_house_before_force_scene_advance 1 \
    --use_grpo True \
    --use_constraints False \
    --grpo_num_generations 2 \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --callbacks wandb_logging_callback
    ```


- test resume training



(TODO: NEXT)
10. How does training look if we do the full rollout with masking thing?
    ```shell
    PYTHONPATH=/root/SafeVLA/allenact:$PYTHONPATH:/root/SafeVLA ALLENACT_DEBUG=True ALLENACT_DEBUG_VST_TIMEOUT=2000 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 training/online/dinov2_vits_tsfm_base.py train \
    --il_ckpt_path /root/data/models/spoc_IL/model.ckpt \
    --output_dir /root/output_dir \
    --dataset_dir /root/data/training_data/astar/ObjectNavType \
    --num_train_processes 4 \
    --tag ObjNav_debug_forked_safevla_4_gpu_4_house_forked_allenact_10000_step_valuenet_300_rollout \
    --max_steps 500 \
    --num_steps_per_rollout 128 \
    --max_houses 77 \
    --auto_resample_when_done False \
    --cost_limit 2.31964 \
    --train_steps_value_network 10000 \
    --mask_out_other_rollouts True \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --use_grpo False \
    --callbacks wandb_logging_callback 2>&1 | tee output.log
    ```


(there should be no difference in training between rollout masking thing with batch size of 1 per gpu and without rollout masking)


Try with lagrange lambda set to 0


Try with the forked allenact


Does resume training work and with the dataset loading thing?


How does training look if we do the full rollout with masking thing?




Can we add the safety constraint in SafeVLA with a flag?



300 steps per rollout, 300 steps episode, masking, 4 houses, 4 GPUs



300 steps per rollout, 300 steps episode, masking, 4 houses, 4 GPUs, constrained PPO



128 steps per rollout, 300 steps episode, 4 houses, 4 GPUs



128 steps per rollout, 300 steps episode, 4 houses, 4 GPUs, constrained PPO


## Testing Constrained GRPO
## 2026-02-09
- [2026-02-09T23:21:54Z] Initialized tracked template files in .project (force=0, git=yes).
