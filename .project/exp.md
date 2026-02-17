

### [ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_10000_step_valuenet_128_rollout/2026-02-09_18-39-03](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/r3vgloon?nw=nwuserrdesc1)
- Training the value function for longer definitely seems to speed up learning of the actor 

### [ObjNav_debug_orig_safevla_2_gpu_4_house_orig_allenact_1000_step_valuenet_128_rollout/2026-02-09_18-12-51](https://wandb.ai/rdesc1-milaquebec/safety_chores/runs/0n5emeb6?nw=nwuserrdesc1)
- resume training seems broken for the original SafeVLA code
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

