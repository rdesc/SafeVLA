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
        num_generations: int = 1,
        use_grpo_lambda: bool = False,
        grpo_lambda: float = 0.95,
        grpo_gamma: float = 0.99,
        advantage_clamp_min: Optional[float] = None,
        advantage_method: str = "scalarize_advantages",
        entropy_coef: float = 0.0,
    ):
        super().__init__()
        assert advantage_method in ("scalarize_advantages", "scalarize_rewards")
        self.clip_param = clip_param
        self.group_advantage_eps = group_advantage_eps
        self.num_generations = num_generations
        self.use_grpo_lambda = use_grpo_lambda  # ε-weight variant from https://arxiv.org/abs/2510.00194
        self.grpo_lambda = grpo_lambda
        self.grpo_gamma = grpo_gamma
        self.advantage_clamp_min = advantage_clamp_min
        self.advantage_method = advantage_method
        self.entropy_coef = entropy_coef
        if self.num_generations <= 0:
            raise ValueError("`num_generations` must be >= 1 for GRPO.")
        if self.use_grpo_lambda:
            if not 0.0 <= self.grpo_lambda <= 1.0:
                raise ValueError(f"Expected grpo_lambda in [0, 1], got {self.grpo_lambda}.")
            if not 0.0 < self.grpo_gamma <= 1.0:
                raise ValueError(f"Expected grpo_gamma in (0, 1], got {self.grpo_gamma}.")

    @staticmethod
    def _squeeze_trailing_dim(t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 2 and t.shape[-1] == 1:
            return t.squeeze(-1)
        return t

    def _sampler_group_ids(  # enables having multiple groups on 1 device
        self, batch: ObservationType, num_samplers: int, device: torch.device
    ) -> torch.LongTensor:
        if "sampler_ids" in batch:
            sampler_ids = torch.as_tensor(
                batch["sampler_ids"], dtype=torch.long, device=device
            )
            if sampler_ids.numel() != num_samplers:
                raise ValueError(
                    f"Expected `sampler_ids` with {num_samplers} entries but found {sampler_ids.numel()}."
                )
        else:
            sampler_ids = torch.arange(num_samplers, device=device, dtype=torch.long)

        group_ids = torch.div(sampler_ids, self.num_generations, rounding_mode="floor")
        unique_group_ids = torch.unique(group_ids, sorted=True)
        for group_id in unique_group_ids.tolist():
            count = int((group_ids == group_id).sum().item())
            if count != self.num_generations:
                raise ValueError(
                    "GRPO mini-batch does not contain full groups. "
                    f"Expected {self.num_generations} samplers for group {group_id}, got {count}. "
                    "Use num_mini_batch=1 or ensure mini-batches align with groups."
                )
        return group_ids

    def _compute_epsilon_weights(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute per-step ε-weight trace: w_t = Σ_{l=0}^{t} (γλ)^l within each episode segment.

        Implements the ε-weight variant from https://arxiv.org/abs/2510.00194 Section 3.2.
        These weights are applied to the per-step loss, not to the log-ratio.
        """
        num_steps, num_samplers = masks.shape
        decay = self.grpo_gamma * self.grpo_lambda
        valid = masks > 0
        all_rows, all_cols, all_vals = [], [], []
        for sampler_idx in range(num_samplers):
            col_valid = valid[:, sampler_idx]
            t = 0
            while t < num_steps:
                while t < num_steps and not bool(col_valid[t].item()):
                    t += 1
                if t >= num_steps:
                    break

                start = t
                while t < num_steps and bool(col_valid[t].item()):
                    t += 1
                end = t

                seg_len = end - start
                if seg_len <= 0:
                    continue

                positions = torch.arange(seg_len, device=masks.device, dtype=masks.dtype)
                powers = decay ** positions  # [(γλ)^0, (γλ)^1, ..., (γλ)^{T-1}]
                weights = torch.cumsum(powers, dim=0)  # w_t = Σ_{l=0}^{t} (γλ)^l
                all_vals.append(weights)
                all_rows.append(torch.arange(start, end, device=masks.device))
                all_cols.append(torch.full((seg_len,), sampler_idx, device=masks.device, dtype=torch.long))

        if all_vals:
            return torch.zeros_like(masks).index_put(
                (torch.cat(all_rows), torch.cat(all_cols)), torch.cat(all_vals)
            )
        return torch.zeros_like(masks)

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        actions = cast(torch.LongTensor, batch["actions"])
        rewards = self._squeeze_trailing_dim(cast(torch.FloatTensor, batch["rewards"]))

        action_log_probs = self._squeeze_trailing_dim(
            actor_critic_output.distributions.log_prob(actions)
        )
        dist_entropy: torch.FloatTensor = actor_critic_output.distributions.entropy()
        returns = self._squeeze_trailing_dim(cast(torch.FloatTensor, batch["returns"]))

        masks = torch.clone(
            self._squeeze_trailing_dim(cast(torch.FloatTensor, batch["masks"]))
        )
        masks[0] = 1.0  # ensure first step is valid
        # final_time_steps = masks.sum(dim=0).long() - 1

        episode_returns = returns[0]  # [num_samplers]
        num_samplers = int(episode_returns.shape[0])
        group_ids = self._sampler_group_ids(
            batch=batch, num_samplers=num_samplers, device=episode_returns.device
        )
        unique_group_ids = torch.unique(group_ids, sorted=True)

        # --- Constraint kwargs ---
        reward_weight = float(kwargs.get("reward_weight", 1.0))
        constraint_weights = kwargs.get("constraint_weights", None)  # [num_constraints] or None
        constraint_names = kwargs.get("constraint_names", [])        # e.g. ["corner","danger",...]
        use_constraints = constraint_weights is not None

        if use_constraints:
            # Per-step cost signals: [T, S] for each constraint.
            per_step_costs = torch.stack([
                self._squeeze_trailing_dim(cast(torch.FloatTensor, batch[name]))
                for name in constraint_names
            ], dim=-1)  # [T, S, K]
            # Episode-level cost per sampler per constraint: sum over masked steps.
            masks_expanded = masks.unsqueeze(-1)  # [T, S, 1]
            episode_costs = (per_step_costs * masks_expanded).sum(dim=0)  # [S, K]

        # --- Advantage computation (episode-level) ---
        groupwise_means = []
        groupwise_stds = []
        cost_groupwise_means: list = []
        cost_groupwise_stds: list = []

        advs = torch.zeros_like(episode_returns)
        c_advs = torch.zeros(
            *episode_returns.shape, len(constraint_names),
            device=episode_returns.device, dtype=episode_returns.dtype,
        ) if use_constraints and self.advantage_method == "scalarize_advantages" else None

        for group_id in unique_group_ids.tolist():
            in_group = group_ids == group_id
            g_returns = episode_returns[in_group]
            mu_r = g_returns.mean()
            sigma_r = g_returns.std(unbiased=False).clamp_min(self.group_advantage_eps)
            groupwise_means.append(float(mu_r.item()))
            groupwise_stds.append(float(sigma_r.item()))

            if use_constraints and self.advantage_method == "scalarize_rewards":
                # Scalarize episode returns: w_r * R[s] + sum_k(w_k * cost_k[s]).
                g_ep_costs = episode_costs[in_group]  # [K_group, num_constraints]
                scalarized = (
                    reward_weight * g_returns
                    + (g_ep_costs * constraint_weights.unsqueeze(0)).sum(dim=-1)
                )
                mu_s = scalarized.mean()
                sigma_s = scalarized.std(unbiased=False).clamp_min(self.group_advantage_eps)
                advs[in_group] = (scalarized - mu_s) / sigma_s
                cost_groupwise_means.append(float(g_ep_costs.mean().item()))
                cost_groupwise_stds.append(float(g_ep_costs.std(unbiased=False).item()))

            elif use_constraints and self.advantage_method == "scalarize_advantages":
                # Normalize reward and each constraint's episode cost independently per group.
                advs[in_group] = (g_returns - mu_r) / sigma_r
                g_ep_costs = episode_costs[in_group]  # [K_group, num_constraints]
                for k in range(len(constraint_names)):
                    g_cost_k = g_ep_costs[:, k]
                    mu_c = g_cost_k.mean()
                    sigma_c = g_cost_k.std(unbiased=False).clamp_min(self.group_advantage_eps)
                    c_advs[in_group, k] = (g_cost_k - mu_c) / sigma_c
                cost_groupwise_means.append(float(g_ep_costs.mean().item()))
                cost_groupwise_stds.append(float(g_ep_costs.std(unbiased=False).item()))

            else:
                # Unconstrained: standard episode-level GRPO.
                advs[in_group] = (g_returns - mu_r) / sigma_r

        if use_constraints and self.advantage_method == "scalarize_advantages":
            # Combine: w_r * r_adv + sum_k(w_k * c_adv_k). constraint_weights are negative.
            cost_adv_combined = (c_advs * constraint_weights.unsqueeze(0)).sum(dim=-1)
            advs = reward_weight * advs + cost_adv_combined

        advs = advs.view(1, -1)

        if self.advantage_clamp_min is not None:
            clamped_mask = advs < self.advantage_clamp_min
            advs = torch.clamp(advs, min=self.advantage_clamp_min)
            adv_clamped_frac = float(clamped_mask.float().mean().item())
        else:
            adv_clamped_frac = 0.0

        adv_stats = {
            "return_mean": float(episode_returns.mean().item()),
            "return_std": float(
                episode_returns.std(unbiased=False)
                .clamp_min(self.group_advantage_eps)
                .item()
            ),
            "return_max": float(episode_returns.max().item()),
            "return_min": float(episode_returns.min().item()),
            "groupwise_return_mean": float(
                sum(groupwise_means) / max(len(groupwise_means), 1)
            ),
            "groupwise_return_std": float(
                sum(groupwise_stds) / max(len(groupwise_stds), 1)
            ),
            "adv_clamped_frac": adv_clamped_frac,
        }
        if use_constraints and cost_groupwise_means:
            adv_stats["cost_groupwise_return_mean"] = float(
                sum(cost_groupwise_means) / len(cost_groupwise_means)
            )
            adv_stats["cost_groupwise_return_std"] = float(
                sum(cost_groupwise_stds) / len(cost_groupwise_stds)
            )

        clip_param = self.clip_param
        old_action_log_probs = self._squeeze_trailing_dim(
            cast(torch.FloatTensor, batch["old_action_log_probs"])
        )
        log_ratio = action_log_probs - old_action_log_probs
        ratio_for_loss = torch.exp(log_ratio)

        while advs.dim() < ratio_for_loss.dim():
            advs = advs.unsqueeze(-1)

        clamped_ratio = torch.clamp(ratio_for_loss, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio_for_loss * advs
        surr2 = clamped_ratio * advs
        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        masks_for_loss = masks
        while masks_for_loss.dim() < action_loss.dim():
            masks_for_loss = masks_for_loss.unsqueeze(-1)

        if self.use_grpo_lambda:
            # ε-weight: weight per-step loss by w_t = Σ_{l=0}^{t} (γλ)^l within each episode.
            trace_weights = self._compute_epsilon_weights(masks)
            while trace_weights.dim() < action_loss.dim():
                trace_weights = trace_weights.unsqueeze(-1)
            effective_weights = trace_weights * masks_for_loss
        else:
            effective_weights = masks_for_loss

        action_loss = (
            (action_loss * effective_weights).sum(dim=0)
            / effective_weights.sum(dim=0).clamp(min=1.0)
        ).mean()

        for idx in range(masks.shape[1]):
            assert masks[:, idx].sum() > 0, f"GRPO expected non-empty valid mask for step {idx}."
        
        entropy_mean = dist_entropy.mean()
        entropy_bonus = -self.entropy_coef * entropy_mean  # minimizing -H*coef maximizes H
        total_loss = action_loss + entropy_bonus

        result = (
            total_loss,
            {
                "action": float(action_loss.item()),
                "entropy": float(-entropy_mean.item()),  # negative = policy is more random (good)
                "entropy_bonus": float(entropy_bonus.item()),
                "log_ratio_mean": float(log_ratio.mean().item()),
                "ratio_mean": float(ratio_for_loss.mean().item()),
                "clamped_ratio_mean": float(clamped_ratio.mean().item()),
                "rollout_num_steps": actions.shape[0],
                "rollout_avg_num_steps": float(masks.sum(dim=0).float().mean().item()),
                "mean_reward_per_step": float((rewards * masks).sum().item() / masks.sum().clamp(min=1.0).item()),
                **adv_stats
            },
        )

        return result
    

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
