

### [ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_10000_step_valuenet_128_rollout/2026-02-09_18-39-03](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/r3vgloon?nw=nwuserrdesc1)
- Training the value function for longer definitely seems to speed up learning of the actor 

### [ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_1000_step_valuenet_128_rollout/2026-02-09_18-12-51](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/0n5emeb6?nw=nwuserrdesc1)
- resume trainig seems broken for the original SafeVLA code
- lagrange multipliers are not saved as part of the checkpoint

### [ObjNav_debug_forked_safevla_1_gpu_1_house_forked_allenact_1000_step_valuenet_128_rollout](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/zx8odo5a?nw=nwuserrdesc1)
- can't seem to train with 1 house and batch size of 1 

### [ObjNav_debug_orig_safevla_4_gpu_4_house_orig_allenact_10000_step_valuenet_400_rollout/2026-02-09_22-34-53](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/u3x3cmz4?nw=nwuserrdesc1)
- Why is training wack when we increase the number of rollout steps?
- This is with the original codebase with basically no changes

###
[ObjNav_debug_orig_safevla_4_gpu_8_house_orig_allenact_10000_step_valuenet_128_rollout_8_bs/2026-02-11_12-28-48](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/5ksvsx8z?nw=nwuserrdesc1)
- What is going on ?? Why is it not learning?

###
[ObjNav_debug_forked_safevla_2_gpu_4_house_forked_allenact_1000_step_valuenet_128_rollout](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/z2m9anih?nw=nwuserrdesc1)
- [seed1](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/do2ppzqt?nw=nwuserrdesc1) and [seed2](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/ggjus7da?nw=nwuserrdesc1)
- seems sensitive to seeds
- my guess is its due to small batch size?

### [ObjNav_debug_orig_safevla_4_gpu_16_house_orig_allenact_10000_step_valuenet_128_rollout_16_bs/2026-02-11_14-38-38](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/q3gcscra?nw=nwuserrdesc1)
- Larger batch size definitely seems to help
- converges on the 16 houses after 200K steps after ~1h of training which gets a success rate of >0.8

### [ObjNav_debug_forked_safevla_4_gpu_16_house_forked_allenact_100000_step_valuenet_250_rollout_250_max_steps_12_bs_mask_out/2026-02-13_00-17-38](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/rnzi7ty0?nw=nwuserrdesc1)
- Much slower for success rate to converge
- Is the masking variable rollout hurting performance?

### [ObjNav_debug_forked_safevla_4_gpu_16_house_forked_allenact_100000_step_valuenet_250_rollout_250_max_steps_12_bs/2026-02-16_20-11-04](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/8m1jgz62?nw=nwuserrdesc1)
- Same run as above but without the rollout masking thing 
- [Resume training works](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/7fcrg8wt?nw=nwuserrdesc1) (except for multiplier loading)! It brings training up to score of ~0.95

### [ObjNav_debug_orig_safevla_4_gpu_8_house_orig_allenact_50000_step_valuenet_300_rollout_300_max_steps_8_bs/2026-02-11_19-45-31](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/kk9cich1?nw=nwuserrdesc1)
- Training works with rollout steps set to 300!
- This is still the original SafeVLA code

### [ObjNav_debug_vanilla_GRPO_4_gpu_4_house_128_rollout_128_max_steps_32_bs_8_groupsize_0.995_gamma_2_update_repeats_2e-5_lr/2026-02-18_22-26-11](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/x8zi56h5?nw=nwuserrdesc1)
- Finally some good learning!!! Here we have only 4 houses
- But why is fps much faster than the other runs?? Ok yea whyyyy is it faster???
- Why is it plateauing at 0.6 ????
- Also why does it start at average score of 0 instead of same average score as 16 houses

### [ObjNav_debug_GRPO_lamdba_4_gpu_4_house_128_rollout_128_max_steps_32_bs_8_groupsize_0.995_gamma_2_update_repeats_2e-5_lr/2026-02-19_01-30-24](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/nnerwe8w?nw=nwuserrdesc1)
- Not clear if GRPO lambda is helping yet
- Resume training also clearly works with GRPO setup (vanilla setup)
- We need to test on the 16 houses
- Also how does `steps_in_house_before_force_scene_advance` affect training? 
- Finally managed to [overfit](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/6ux15xvq?nw=nwuserrdesc1) to the scenarios !! Weird that the score does not plateau there though

### [ObjNav_debug_GRPO_lamdba_4_gpu_4_house_128_rollout_128_max_steps_16_bs_4_groupsize_0.995_gamma_2_update_repeats_2e-5_lr/2026-02-22_19-33-51](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/rhh03pw8?nw=nwuserrdesc1)
- Looks like group size 4 also works fine