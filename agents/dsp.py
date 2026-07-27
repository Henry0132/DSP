from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, GCStatePlannerVectorField, UnconditionalEmbedding


class DSPAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['high_value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['high_value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['high_value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }
    
    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        if self.config['encoder'] is not None:
            goal_reps = self.network.select('goal_rep')(
                jnp.concatenate(
                    [batch['observations'], batch['low_actor_goals']],
                    axis=-1,
                ),
                params=grad_params,
            )
            if not self.config['low_actor_rep_grad']:
                goal_reps = jax.lax.stop_gradient(goal_reps)
            dist = self.network.select('low_actor')(
                batch['observations'],
                goal_reps,
                goal_encoded=True,
                params=grad_params,
            )
        else:
            dist = self.network.select('low_actor')(
                batch['observations'],
                batch['low_actor_goals'],
                params=grad_params,
            )
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params, rng=None):
        """Compute the high-level flow BC loss."""
        if self.config['encoder'] is not None:
            x_1 = self.network.select('goal_rep')(
                jnp.concatenate(
                    [batch['observations'], batch['high_actor_actions']],
                    axis=-1,
                )
            )
            x_1 = jax.lax.stop_gradient(x_1)
            conditional_goals = self.network.select('goal_rep')(
                jnp.concatenate(
                    [batch['observations'], batch['high_actor_goals']],
                    axis=-1,
                ),
                params=grad_params,
            )
        else:
            x_1 = batch['high_actor_actions']
            conditional_goals = batch['high_actor_goals']

        batch_size, action_dim = x_1.shape
        x_rng, t_rng, cfg_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        unc_embed = self.network.select('high_unc_embed')(
            params=grad_params
        )
        do_cfg = jax.random.bernoulli(
            cfg_rng,
            p=0.1,
            shape=(batch_size,),
        )
        goals = jnp.where(
            do_cfg[:, None],
            unc_embed,
            conditional_goals,
        )

        pred = self.network.select('high_actor_flow')(
            batch['observations'],
            x_t,
            t,
            goals,
            params=grad_params,
        )
        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, high_actor_rng = jax.random.split(rng, 2)
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """
        Sample actions from the actor.
        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        if self.config['encoder'] is not None:
            high_seed, low_seed = jax.random.split(seed)
            encoded_observations = self.network.select(
                'high_actor_flow_encoder'
            )(observations)
            goal_reps = self.network.select('goal_rep')(
                jnp.concatenate([observations, goals], axis=-1)
            )

            subgoals = jax.random.normal(
                high_seed,
                (self.config['num_samples'], self.config['goal_dim']),
            )
            repeated_observations = jnp.repeat(
                encoded_observations[None],
                self.config['num_samples'],
                axis=0,
            )
            repeated_goals = jnp.repeat(
                goal_reps[None],
                self.config['num_samples'],
                axis=0,
            )
            unconditional_goals = jnp.broadcast_to(
                self.network.select('high_unc_embed')(),
                repeated_goals.shape,
            )

            for i in range(self.config['flow_steps']):
                t = jnp.full(
                    (self.config['num_samples'], 1),
                    i / self.config['flow_steps'],
                )
                unc_vels = self.network.select('high_actor_flow')(
                    repeated_observations,
                    subgoals,
                    t,
                    unconditional_goals,
                    is_encoded=True,
                )
                cond_vels = self.network.select('high_actor_flow')(
                    repeated_observations,
                    subgoals,
                    t,
                    repeated_goals,
                    is_encoded=True,
                )
                subgoals = subgoals + (
                    unc_vels
                    + self.config['cfg'] * (cond_vels - unc_vels)
                ) / self.config['flow_steps']

            subgoal = subgoals[
                jax.random.randint(
                    low_seed,
                    (),
                    0,
                    self.config['num_samples'],
                )
            ]
            subgoal = subgoal / jnp.maximum(
                jnp.linalg.norm(subgoal),
                1e-6,
            ) * jnp.sqrt(subgoal.shape[-1])

            low_dist = self.network.select('low_actor')(
                observations,
                subgoal,
                goal_encoded=True,
                temperature=temperature,
            )
            actions = low_dist.sample(seed=low_seed)
            if not self.config['discrete']:
                actions = jnp.clip(actions, -1, 1)
            return actions, subgoal

        high_seed, low_seed = jax.random.split(seed)

        subgoals = jax.random.normal(               # [M, goal_dim]
            high_seed,
            (
                *observations.shape[:-1],
                self.config['num_samples'],
                self.config['goal_dim']
            ),
        )
        n_observations = jnp.repeat(jnp.expand_dims(observations, 0), self.config['num_samples'], axis=0)   # [M, state_dim]
        n_goals = jnp.repeat(jnp.expand_dims(goals, 0), self.config['num_samples'], axis=0)     # [M, goal_dim]
        
        high_unc_embed = self.network.select('high_unc_embed')()
        n_high_unc_embed = jnp.repeat(high_unc_embed, self.config['num_samples'], axis=0)   # (M, goal_dim)

        for i in range(self.config['flow_steps']):
            t = jnp.full((self.config['num_samples'], 1), i / self.config['flow_steps'])

            unc_vels = self.network.select('high_actor_flow')(n_observations, subgoals, t, n_high_unc_embed)
            cond_vels = self.network.select('high_actor_flow')(n_observations, subgoals, t, n_goals)
            # cfg
            vels = unc_vels + self.config['cfg'] * (cond_vels - unc_vels)
            subgoals = subgoals + vels / self.config['flow_steps']

        subgoal = subgoals[jax.random.randint(low_seed, (), 0, self.config['num_samples'])]  # [goal_dim]

        low_dist = self.network.select('low_actor')(observations, subgoal, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions, subgoal


    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['high_actor_goals']
        ex_times = ex_actions[..., :1]
        ob_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        goal_dim = ex_goals.shape[-1]
        
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoder.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
            goal_dim = config['rep_dim']
            high_actor_encoder_def = encoder_module()
        else:
            goal_rep_seq = []
            goal_dim = ex_goals.shape[-1]
            high_actor_encoder_def = None
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
        else:
            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None

        # Define networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )

        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        high_actor_flow_def = GCStatePlannerVectorField(
            hidden_dims=config['high_actor_hidden_dims'],
            state_dim=goal_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=high_actor_encoder_def,
            encode_goal=config['encoder'] is None,
        )

        high_unc_embed_def = UnconditionalEmbedding(
            goal_dim=goal_dim,
        )
 
        low_actor_def = GCActor(
                hidden_dims=config['low_actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )
        
        ex_goal_reps = jnp.zeros(
            (*ex_actions.shape[:-1], goal_dim),
            dtype=jnp.float32,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            high_actor_flow=(
                high_actor_flow_def,
                (ex_observations, ex_goal_reps, ex_times, ex_goal_reps),
            ),
            high_unc_embed=(high_unc_embed_def, ()),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
        )
        if high_actor_encoder_def is not None:
            network_info['high_actor_flow_encoder'] = (
                high_actor_encoder_def,
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']

        config['ob_dim'] = ob_dim
        config['action_dim'] = action_dim
        config['goal_dim'] = goal_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dsp',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            mlp_class='mlp',  # MLP class.
            high_actor_hidden_dims=(512, 512, 512, 512),  # High actor network hidden dimensions.
            low_actor_hidden_dims=(512, 512, 512),  # Low actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization for the actor.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            flow_steps=20,  # Number of flow steps.
            cfg=3.0,  # CFG coefficient.
            num_samples=32,  # Number of action samples for evaluation.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            ob_dim=ml_collections.config_dict.placeholder(int),  # Observation dimension (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            goal_dim=ml_collections.config_dict.placeholder(int),  # Goal dimension (will be set automatically).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            subgoal_steps=25,  # Subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
