from collections import OrderedDict
from typing import Dict, cast
from allenact.algorithms.onpolicy_sync.losses import PPO

import torch
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import Distr, CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput

from typing import Dict, Optional, Callable, cast, Tuple
from omnisafe.common.lagrange import Lagrange


class Imitation(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def __init__(
        self, uuid: str = "expert_pickupable", action_idx: int = 8, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.uuid = uuid
        self.action_idx = action_idx

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[Distr],
        *args,
        **kwargs,
    ):
        """Computes the imitation loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.batched_experience_generator`.
            Here `batch["observations"]` must contain `"expert_action"` observations
            or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
            a sensor producing such observations.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args. Ignored.
        kwargs : Extra kwargs. Ignored.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        losses = OrderedDict()

        should_report_loss = False
        has_observation_to_compute = False

        total_loss = 0
        if self.uuid in observations:
            should_report_loss = True
            has_observation_to_compute = True
            total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                actor_critic_output.distributions.logits[:, :, self.action_idx],
                observations[self.uuid],
            )

        if not has_observation_to_compute:
            raise NotImplementedError(
                "Imitation loss requires either `expert_action` or `expert_policy`"
                " sensor to be active."
            )
        return (
            total_loss,
            (
                {"expert_cross_entropy": total_loss.item(), **losses}
                if should_report_loss
                else {}
            ),
        )


class PPOValueStopGrad(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    """

    def __init__(
        self,
        clip_param: float,
        discrete_critics: bool,
        use_clipped_value_loss=True,
        clip_decay=None,
        *args,
        **kwargs,
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)
        self.discrete_critics = discrete_critics

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        if self.discrete_critics:
            logits = actor_critic_output.extras["stop_grad_logits"]
            loss_func = actor_critic_output.extras["loss_func"]
            reshaped_logits = logits.view(-1, logits.shape[-1])
            returns = cast(torch.FloatTensor, batch["returns"])
            reshaped_returns = returns.view(-1)
            value_loss = 0.5 * loss_func(reshaped_logits, reshaped_returns)
        else:
            values = actor_critic_output.extras["stop_grad_values"]
            clip_param = self.clip_param * self.clip_decay(step_count)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - batch["returns"]).pow(2)
                value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (
                    0.5
                    * (cast(torch.FloatTensor, batch["returns"]) - values).pow(2).mean()
                )

        bias_norm = actor_critic_output.extras["bias_norm"]
        weight_norm = actor_critic_output.extras["weight_norm"]
        try:
            weight_grad = actor_critic_output.extras["weight_grad_norm"]
        except:
            weight_grad = 0

        return (
            value_loss,
            {
                "value": value_loss.item(),
                "bias_norm": bias_norm,
                "weight_norm": weight_norm,
                "weight_grad": weight_grad,
            },
        )


class PPOLogGrad(PPO):
    def __init__(self, discrete_critics: bool, action_loss_schedule, *args, **kwargs):
        """
        Args:
            discrete_critics: whether the critic is discrete
            action_loss_schedule: a function that takes the step count and returns the weight for the action loss
        """
        super().__init__(*args, **kwargs)
        self.discrete_critics = discrete_critics
        self.action_loss_schedule = (
            action_loss_schedule
            if action_loss_schedule is not None
            else (lambda x: 1.0)
        )

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:  # TODO tuple output

        actions = cast(torch.LongTensor, batch["actions"])

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(
                t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * batch[self.adv_key]
        surr2 = clamped_ratio * batch[self.adv_key]

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        if self.discrete_critics:
            logits = actor_critic_output.extras["full_logits"]
            loss_func = actor_critic_output.extras["loss_func"]
            reshaped_logits = logits.view(-1, logits.shape[-1])
            returns = cast(torch.FloatTensor, batch["returns"])
            reshaped_returns = returns.view(-1)
            value_loss = 0.5 * loss_func(reshaped_logits, reshaped_returns)
        else:
            values = actor_critic_output.values
            clip_param = self.clip_param * self.clip_decay(step_count)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - batch["returns"]).pow(2)
                value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (
                    0.5
                    * (cast(torch.FloatTensor, batch["returns"]) - values).pow(2).mean()
                )

        bias_norm = actor_critic_output.extras["bias_norm"]
        weight_norm = actor_critic_output.extras["weight_norm"]
        try:
            weight_grad = actor_critic_output.extras["weight_grad_norm"]
        except:
            weight_grad = torch.tensor([0.0])

        action_weight = self.action_loss_schedule(step_count)

        # noinspection PyUnresolvedReferences
        return (
            {
                "value": (value_loss, self.value_loss_coef),
                "action": (action_loss, action_weight),
                "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
            },
            {
                "bias_norm": bias_norm,
                "weight_norm": weight_norm,
                "weight_grad": weight_grad,
                "action_weight": action_weight,
                # "ratio": ratio,
                # "ratio_clamped": clamped_ratio,
                # "ratio_used": torch.where(
                #     cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                # ),
            },
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        
        losses_per_step, ratio_info = self.loss_per_step(
            step_count=step_count,
            batch=batch,
            actor_critic_output=actor_critic_output,
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            }
            | ratio_info,
        )

        return result


class SafePPOLogGrad(PPO):
    def __init__(self, discrete_critics: bool, action_loss_schedule, mask_out_other_rollouts, *args, **kwargs):
        """
        Args:
            discrete_critics: whether the critic is discrete
            action_loss_schedule: a function that takes the step count and returns the weight for the action loss
        """
        super().__init__(*args, **kwargs)
        self.discrete_critics = discrete_critics
        self.action_loss_schedule = (
            action_loss_schedule
            if action_loss_schedule is not None
            else (lambda x: 1.0)
        )
        self.c_adv_key = "c_" + self.adv_key
        self._mask_out_other_rollouts = mask_out_other_rollouts
        print("mask_out_other_rollouts:", mask_out_other_rollouts)

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        lagrangian_multiplier: torch.Tensor,
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:  # TODO tuple output

        actions = cast(torch.LongTensor, batch["actions"])

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(
                t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        with torch.no_grad():
            penalty = torch.tensor(lagrangian_multiplier.item())
        surr1 = (
            ratio
            * (batch[self.adv_key] - penalty * batch[self.c_adv_key])
            / (1.0 + penalty)
        )
        surr2 = (
            clamped_ratio
            * (batch[self.adv_key] - penalty * batch[self.c_adv_key])
            / (1.0 + penalty)
        )

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)
        
        if self._mask_out_other_rollouts:
            masks = cast(torch.FloatTensor, batch["masks"])
            action_loss = (action_loss * masks).sum() / masks.sum().clamp(min=1)

        if self.discrete_critics:
            logits = actor_critic_output.extras["full_logits"]
            loss_func = actor_critic_output.extras["loss_func"]
            reshaped_logits = logits.view(-1, logits.shape[-1])
            returns = cast(torch.FloatTensor, batch["returns"])
            reshaped_returns = returns.view(-1)
            value_loss = 0.5 * loss_func(reshaped_logits, reshaped_returns)
        else:
            values = actor_critic_output.values
            clip_param = self.clip_param * self.clip_decay(step_count)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - batch["returns"]).pow(2)
                value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                if self._mask_out_other_rollouts:
                    value_loss = (
                    0.5
                    * ((cast(torch.FloatTensor, batch["returns"]) - values).pow(2) * masks).sum() / masks.sum().clamp(min=1)
                )
                else:
                    value_loss = (
                        0.5
                        * (cast(torch.FloatTensor, batch["returns"]) - values).pow(2).mean()
                          # TODO: make sure returns look ok, is it because index 0 is 0 in mask
                          # TODO: make sure returns look ok 
                    )

        bias_norm = actor_critic_output.extras["bias_norm"]
        weight_norm = actor_critic_output.extras["weight_norm"]
        try:
            weight_grad = actor_critic_output.extras["weight_grad_norm"]
        except:
            weight_grad = torch.tensor([0.0])

        action_weight = self.action_loss_schedule(step_count)
        
        # noinspection PyUnresolvedReferences
        return (
            {
                "value": (value_loss, self.value_loss_coef),
                "action": (action_loss, action_weight),
                "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
            },
            {
                "bias_norm": bias_norm,
                "weight_norm": weight_norm,
                "weight_grad": weight_grad,
                "action_weight": action_weight,
                # "ratio": ratio,
                # "ratio_clamped": clamped_ratio,
                # "ratio_used": torch.where(
                #     cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                # ),
            },
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        losses_per_step, ratio_info = self.loss_per_step(
            step_count=step_count,
            batch=batch,
            actor_critic_output=actor_critic_output,
            lagrangian_multiplier=kwargs["lagrangian_multiplier"],
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            }
            | ratio_info,
        )

        return result

class SafeGRPOLogGrad(AbstractActorCriticLoss):
    def __init__(
        self,
        clip_param: float,
        group_advantage_eps: float = 1e-5,
        per_step_advantage: bool = False,
    ):
        super().__init__()
        self.clip_param = clip_param
        self.group_advantage_eps = group_advantage_eps
        self.per_step_advantage = per_step_advantage

    def _compute_advantages(self,
                            returns: torch.Tensor,
                            final_time_steps: torch.Tensor,
                            reward_weight: list,
                            constraint_weights: list,
                            costs: list,
                            advantage_method: str,
                            ) -> torch.Tensor:
        episode_returns = returns[final_time_steps, torch.arange(returns.shape[1])]
        
        # TODO: this is the last part right??!
        
        # need t get the costs here 
        # if advantage_method == 'scalarize_rewards':
        #     scaled_returns = reward_weight * episode_returns
        #     for idx, c_weight in enumerate(constraint_weights):
        #         scaled_returns += c_weight * 
        #     episode_returns = scaled_returns
        
        mean_return = episode_returns.mean()
        std_return = episode_returns.std()
        
        if self.per_step_advantage:
            adv_reward = (returns.squeeze() - mean_return) / (std_return + self.group_advantage_eps)
            
            # TODO: this is wrong!!!
        else:
            adv_reward = ((episode_returns - mean_return) / (std_return + 1e-5)).view(1, -1)
            print("episode returns:", episode_returns.squeeze(), "adv_reward:", adv_reward.squeeze())
        
        return (
                adv_reward,
            {
                "group_return_mean": float(mean_return.item()),
                "group_return_std": float(std_return.item()),
                "group_return_max": float(episode_returns.max().item()),
                "group_return_min": float(episode_returns.min().item()),
            }
        )


    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        actions = cast(torch.LongTensor, batch["actions"])
        rewards = cast(torch.FloatTensor, batch["rewards"])
        reward_weight = kwargs.get("reward_weight")  # float
        constraint_weights = kwargs.get("constraint_weights")  # list of floats
        advantage_method = kwargs.get("advantage_method")
        costs = kwargs.get("costs")  # list of tensors
        
        print("tensor shape in computing loss:", actions.shape)
        print("reward and constraint weights", reward_weight, constraint_weights)
        
        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        returns = cast(torch.FloatTensor, batch["returns"])
        
        print("returns sum", returns.sum().item())
        
        masks = torch.clone(cast(torch.FloatTensor, batch["masks"]))
        masks[0] = 1.0  # ensure first step is valid 
        final_time_steps = masks.sum(dim=0).long().squeeze() - 1
        
        episode_returns = returns[0]
        
        if self.per_step_advantage:
            valid = masks > 0
            valid_rewards = rewards[valid]
            mu = valid_rewards.mean()
            sigma = valid_rewards.std(unbiased=False).clamp_min(self.group_advantage_eps)
            advs = ((rewards - mu) / sigma * masks).flip(0).cumsum(dim=0).squeeze().flip(0)  # TODO: this still seems broken
            
            # mu = 1 / (final_time_steps.sum()) * rewards.cumsum(dim=0)[final_time_steps, torch.arange(rewards.shape[1])].sum()
            # sigma = torch.sqrt(1 / (final_time_steps.sum()) * ((rewards - mu).pow(2)).cumsum(dim=0)[final_time_steps, torch.arange(rewards.shape[1])].sum())
            # advs = (((rewards - mu) / (sigma + self.group_advantage_eps)) * masks).flip(0).cumsum(dim=0).squeeze().flip(0)
            
        else:
            mu = episode_returns.mean()
            sigma = episode_returns.std(unbiased=False).clamp_min(self.group_advantage_eps)
            advs = ((episode_returns - mu) / (sigma)).view(1, -1)
        
        adv_stats = {
            "group_return_mean": float(mu.item()),
            "group_return_std": float(sigma.item()),
            "group_return_max": float(episode_returns.max().item()),
            "group_return_min": float(episode_returns.min().item()),
        }
        print("episode returns:", episode_returns.squeeze(), "advs:", advs.squeeze())
        print(adv_stats)
        
        # advs, adv_stats = self._compute_advantages(rewards,
        #                                            final_time_steps,
        #                                            reward_weight,
        #                                            constraint_weights,
        #                                            costs,
        #                                            advantage_method)

        clip_param = self.clip_param
        log_ratio = action_log_probs - batch["old_action_log_probs"]
        ratio = torch.exp(log_ratio)
        
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * advs
        surr2 = clamped_ratio * advs
        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        action_loss = ((action_loss * masks.squeeze()).sum(dim=0) / masks.sum(dim=0).squeeze()).mean()

        for idx in range(masks.shape[1]):
            assert masks[:, idx].sum() > 0, f"GRPO expected non-empty valid mask for step {idx}."

        result = (
            action_loss,
            {
                "action": float(action_loss.item()),
                "log_ratio_mean": float(log_ratio.mean().item()),
                "ratio_mean": float(ratio.mean().item()),
                "clamped_ratio_mean": float(clamped_ratio.mean().item()),
                "rollout_num_steps": actions.shape[0],
                "rollout_avg_num_steps": float(final_time_steps.sum().item() / len(final_time_steps)),
                "mean_reward_per_step": float((returns * masks).sum().item() / masks.sum().item()),
                **adv_stats
            },
        )

        return result
    
# TODO: what if we try ppo without the baseline?
# TODO: try GRPO without stdev
# TODO: why is performance crashing like crazy when we set gamma to 0.90
# TODO: try per step and with reward shaping!
# TODO: try GRPO with the fixed norm
    

class PPOStopGrad(PPO):
    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:

        actions = cast(torch.LongTensor, batch["actions"])
        # values = actor_critic_output.values
        values = actor_critic_output.extras["stop_grad_values"]

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(
                t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * batch[self.adv_key]
        surr2 = clamped_ratio * batch[self.adv_key]

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        if self.use_clipped_value_loss:
            value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                -clip_param, clip_param
            )
            value_losses = (values - batch["returns"]).pow(2)
            value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (cast(torch.FloatTensor, batch["returns"]) - values).pow(
                2
            )

        # noinspection PyUnresolvedReferences
        return (
            {
                "value": (value_loss, self.value_loss_coef),
                "action": (action_loss, None),
                "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
            },
            (
                {
                    "ratio": ratio,
                    "ratio_clamped": clamped_ratio,
                    "ratio_used": torch.where(
                        cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                    ),
                }
                if self.show_ratios
                else {}
            ),
        )
