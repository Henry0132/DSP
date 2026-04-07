import copy
from typing import Sequence

import flax
import flax.linen as nn
from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from jaxrl_m.common import TrainState
from jaxrl_m.networks import MLP, Policy, default_init
from jaxrl_m.typing import PRNGKey
from src.special_networks import LayerNormMLP, MonolithicVF


class UnconditionalEmbedding(nn.Module):
    goal_dim: int

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        return self.param("embedding", nn.initializers.normal(stddev=1.0), (self.goal_dim,))


class StatePlannerVectorField(nn.Module):
    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: int = 1

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        noisy_subgoals: jnp.ndarray,
        times: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, noisy_subgoals, times, goals], axis=-1)
        if self.layer_norm:
            hidden = LayerNormMLP(self.hidden_dims, activate_final=True, activations=nn.gelu)(inputs)
        else:
            hidden = MLP(self.hidden_dims, activate_final=True, activations=nn.gelu)(inputs)
        return nn.Dense(self.state_dim, kernel_init=default_init())(hidden)


class DSPActorCritic(nn.Module):
    value_net: nn.Module
    target_value_net: nn.Module
    low_actor_net: nn.Module
    high_actor_flow_net: nn.Module
    high_unc_embed_net: nn.Module

    def value(self, observations, goals, **kwargs):
        return self.value_net(observations, goals, **kwargs)

    def target_value(self, observations, goals, **kwargs):
        return self.target_value_net(observations, goals, **kwargs)

    def low_actor(self, observations, goals, **kwargs):
        return self.low_actor_net(jnp.concatenate([observations, goals], axis=-1), **kwargs)

    def high_actor_flow(self, observations, noisy_subgoals, times, goals, **kwargs):
        return self.high_actor_flow_net(observations, noisy_subgoals, times, goals, **kwargs)

    def high_unc_embed(self):
        return self.high_unc_embed_net()

    def __call__(self, observations, goals):
        batch_shape = observations.shape[:-1]
        times = jnp.zeros((*batch_shape, 1))
        return {
            "value": self.value(observations, goals),
            "target_value": self.target_value(observations, goals),
            "low_actor": self.low_actor(observations, goals),
            "high_actor_flow": self.high_actor_flow(observations, goals, times, goals),
            "high_unc_embed": self.high_unc_embed(),
        }


def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)


class DSPAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    network: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def _value_targets(self, batch):
        rewards = batch["rewards"] - 1.0
        masks = 1.0 - batch["rewards"]
        return rewards, masks

    def value_loss(self, batch, network_params):
        rewards, masks = self._value_targets(batch)

        next_v1_t, next_v2_t = self.network(
            batch["next_observations"], batch["goals"], method="target_value"
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = rewards + self.config["discount"] * masks * next_v_t

        v1_t, v2_t = self.network(
            batch["observations"], batch["goals"], method="target_value"
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.config["discount"] * masks * next_v1_t
        q2 = rewards + self.config["discount"] * masks * next_v2_t
        v1, v2 = self.network(
            batch["observations"], batch["goals"], method="value", params=network_params
        )
        v = (v1 + v2) / 2

        value_loss1 = expectile_loss(adv, q1 - v1, self.config["pretrain_expectile"]).mean()
        value_loss2 = expectile_loss(adv, q2 - v2, self.config["pretrain_expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
            "adv_mean": adv.mean(),
        }

    def low_actor_loss(self, batch, network_params):
        v1, v2 = self.network(batch["observations"], batch["low_goals"], method="value")
        nv1, nv2 = self.network(batch["next_observations"], batch["low_goals"], method="value")
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["low_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network(
            batch["observations"],
            batch["low_goals"],
            method="low_actor",
            params=network_params,
        )
        log_prob = dist.log_prob(batch["actions"])
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(dist.scale_diag),
        }

    def high_actor_loss(self, batch, network_params, rng):
        batch_size, goal_dim = batch["high_targets"].shape
        x_rng, t_rng, cfg_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, goal_dim))
        x_1 = batch["high_targets"]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0

        unc_embed = self.network(method="high_unc_embed", params=network_params)
        do_cfg = jax.random.bernoulli(
            cfg_rng, p=self.config["cfg_dropout_prob"], shape=(batch_size,)
        )
        goals = jnp.where(do_cfg[:, None], unc_embed, batch["high_goals"])

        pred = self.network(
            batch["observations"],
            x_t,
            t,
            goals,
            method="high_actor_flow",
            params=network_params,
        )
        actor_loss = jnp.mean((pred - target_velocity) ** 2)

        return actor_loss, {
            "actor_loss": actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, network_params, rng):
        info = {}
        rng, high_actor_rng = jax.random.split(rng)

        value_loss, value_info = self.value_loss(batch, network_params)
        info.update({f"value/{k}": v for k, v in value_info.items()})

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, network_params)
        info.update({f"low_actor/{k}": v for k, v in low_actor_info.items()})

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, network_params, high_actor_rng)
        info.update({f"high_actor/{k}": v for k, v in high_actor_info.items()})

        return value_loss + low_actor_loss + high_actor_loss, info

    def _plan_subgoal(self, observations, goals, seed):
        subgoals = jax.random.normal(seed, goals.shape)
        unc_goal = self.network(method="high_unc_embed")
        unc_goal = jnp.broadcast_to(unc_goal, goals.shape)

        def body_fn(i, cur_subgoals):
            t = jnp.full((*cur_subgoals.shape[:-1], 1), i / self.config["flow_steps"])
            unc_vel = self.network(
                observations,
                cur_subgoals,
                t,
                unc_goal,
                method="high_actor_flow",
            )
            cond_vel = self.network(
                observations,
                cur_subgoals,
                t,
                goals,
                method="high_actor_flow",
            )
            vels = unc_vel + self.config["cfg"] * (cond_vel - unc_vel)
            return cur_subgoals + vels / self.config["flow_steps"]

        return jax.lax.fori_loop(0, self.config["flow_steps"], body_fn, subgoals)

    def pretrain_update(self, batch, seed=None):
        del seed
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(network_params):
            return self.total_loss(batch, network_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn, has_aux=True)

        params = unfreeze(new_network.params)
        params["target_value_net"] = jax.tree_map(
            lambda p, tp: p * self.config["target_update_rate"] + tp * (1 - self.config["target_update_rate"]),
            params["value_net"],
            params["target_value_net"],
        )
        new_network = new_network.replace(params=freeze(params))

        return self.replace(network=new_network, rng=new_rng), info

    pretrain_update = jax.jit(pretrain_update)

    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        low_dim_goals: bool = False,
        seed: PRNGKey,
        temperature: float = 1.0,
        discrete: int = 0,
        num_samples: int = None,
    ) -> jnp.ndarray:
        del discrete

        if low_dim_goals:
            subgoals = goals
        else:
            subgoals = self._plan_subgoal(observations, goals, seed)

        dist = self.network(
            observations,
            subgoals,
            temperature=temperature,
            method="low_actor",
        )
        if num_samples is None:
            actions = dist.sample(seed=seed)
        else:
            actions = dist.sample(seed=seed, sample_shape=num_samples)
        return jnp.clip(actions, -1, 1)

    sample_actions = jax.jit(
        sample_actions, static_argnames=("low_dim_goals", "discrete", "num_samples")
    )

    def sample_high_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        num_samples: int = None,
    ) -> jnp.ndarray:
        del temperature

        if num_samples is None:
            subgoals = self._plan_subgoal(observations, goals, seed)
            return subgoals - observations

        seeds = jax.random.split(seed, num_samples)
        subgoals = jax.vmap(lambda cur_seed: self._plan_subgoal(observations, goals, cur_seed))(seeds)
        return subgoals - observations

    sample_high_actions = jax.jit(sample_high_actions, static_argnames=("num_samples",))

    @jax.jit
    def get_policy_rep(self, *, targets: np.ndarray, bases: np.ndarray = None) -> jnp.ndarray:
        del bases
        return targets


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    lr: float = 3e-4,
    low_actor_hidden_dims: Sequence[int] = (512, 512, 512),
    high_actor_hidden_dims: Sequence[int] = (512, 512, 512, 512),
    value_hidden_dims: Sequence[int] = (512, 512, 512),
    discount: float = 0.99,
    tau: float = 0.005,
    pretrain_expectile: float = 0.7,
    low_alpha: float = 3.0,
    flow_steps: int = 20,
    cfg: float = 3.0,
    cfg_dropout_prob: float = 0.1,
    use_layer_norm: int = 1,
    use_rep: int = 0,
    use_waypoints: int = 1,
    **kwargs,
):
    print("Extra kwargs:", kwargs)

    if use_rep:
        raise NotImplementedError("DSP in this HIQL codebase currently supports state goals only. Please run with --use_rep 0.")
    if not use_waypoints:
        raise ValueError("DSP requires hierarchical subgoals. Please run with --use_waypoints 1.")

    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)

    goal_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    value_def = MonolithicVF(
        hidden_dims=value_hidden_dims,
        use_layer_norm=use_layer_norm,
        rep_dim=goal_dim,
    )
    target_value_def = copy.deepcopy(value_def)
    low_actor_def = Policy(
        low_actor_hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
        state_dependent_std=False,
        tanh_squash_distribution=False,
    )
    high_actor_flow_def = StatePlannerVectorField(
        hidden_dims=high_actor_hidden_dims,
        state_dim=goal_dim,
        layer_norm=use_layer_norm,
    )
    high_unc_embed_def = UnconditionalEmbedding(goal_dim=goal_dim)

    network_def = DSPActorCritic(
        value_net=value_def,
        target_value_net=target_value_def,
        low_actor_net=low_actor_def,
        high_actor_flow_net=high_actor_flow_def,
        high_unc_embed_net=high_unc_embed_def,
    )
    network_params = network_def.init(init_key, observations, observations)["params"]
    params = unfreeze(network_params)
    params["target_value_net"] = copy.deepcopy(params["value_net"])
    network = TrainState.create(
        network_def,
        freeze(params),
        tx=optax.chain(optax.zero_nans(), optax.adam(learning_rate=lr)),
    )

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            target_update_rate=tau,
            pretrain_expectile=pretrain_expectile,
            low_alpha=low_alpha,
            flow_steps=flow_steps,
            cfg=cfg,
            cfg_dropout_prob=cfg_dropout_prob,
            use_rep=use_rep,
            use_waypoints=use_waypoints,
        )
    )

    return DSPAgent(rng, network=network, config=config)


def get_default_config():
    return ml_collections.ConfigDict(
        {
            "lr": 3e-4,
            "low_actor_hidden_dims": (512, 512, 512),
            "high_actor_hidden_dims": (512, 512, 512, 512),
            "value_hidden_dims": (512, 512, 512),
            "discount": 0.99,
            "tau": 0.005,
            "pretrain_expectile": 0.7,
            "low_alpha": 3.0,
            "flow_steps": 20,
            "cfg": 3.0,
            "cfg_dropout_prob": 0.1,
            "use_rep": 0,
            "use_waypoints": 1,
        }
    )
