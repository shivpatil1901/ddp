"""
IQ-Learn Implementation for Safety Ant Environment
Based on: "IQ-Learn: Inverse soft-Q Learning for Imitation" (NeurIPS 2021)
Paper: https://arxiv.org/pdf/2106.12142
Original repo: https://github.com/Div-Infinity/IQ-Learn

IQ-Learn is a non-adversarial imitation learning method that learns from expert demonstrations
by matching the soft-Q values of the expert and learner policies.

Key Features:
- Non-adversarial (no discriminator instability)
- Works with continuous action spaces
- Supports offline and online learning
- More stable than GAIL/AIRL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import pickle
from collections import deque
import importlib
import time
from stable_baselines3.common.monitor import Monitor

try:
    gym = importlib.import_module("gym")
except Exception:
    gym = None

try:
    safety_gym = importlib.import_module("safety_gym")
except Exception:
    safety_gym = None


def SafetyEnvToGymEnvWrapper(env):
    return env


def env_reset(env, seed=None):
    """Reset wrapper compatible with both gym and gymnasium.

    Returns the observation only.
    """
    # Try modern API first (seed kwarg). If the wrapped env does not accept
    # `seed` (Monitor may forward kwargs to an env that uses legacy API),
    # fall back to legacy seeding: call `env.seed(seed)` if available and
    # then `env.reset()` without kwargs.
    try:
        if seed is not None:
            out = env.reset(seed=seed)
        else:
            out = env.reset()
    except TypeError:
        # Legacy environment that doesn't accept `seed` kwarg.
        try:
            if seed is not None and hasattr(env, 'seed'):
                env.seed(seed)
        except Exception:
            pass
        out = env.reset()

    # Normalize return value: gym returns obs, gymnasium may return (obs, info)
    if isinstance(out, tuple):
        return out[0]
    return out


def env_step(env, action):
    """Step wrapper compatible with both gym and gymnasium.

    Returns: obs, reward, done, info
    """
    out = env.step(action)
    # gymnasium: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info

    # classic gym: (obs, reward, done, info)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info

    # Fallback: return whatever was returned
    return out


def get_preferred_device():
    """Return a torch.device, probing CUDA compatibility and falling back to CPU.

    Some PyTorch builds report CUDA available but lack kernel images for the
    local GPU (e.g., A100) which raises runtime errors when allocating tensors
    or creating CUDA tensors. Probe by allocating a tiny tensor and catch any
    errors to safely fall back to CPU.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    try:
        # Try a tiny allocation on the default CUDA device.
        torch.zeros(1, device='cuda')
        return torch.device('cuda')
    except Exception as e:
        print("Warning: CUDA probe failed, falling back to CPU. Details:", e)
        return torch.device('cpu')


# ========================== Configuration ==========================
SEED = 42
ENV_NAME = "Safexp-PointGoal1-v0"
EXPERT_DATA_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# IQ-Learn Hyperparameters (Matching Original Implementation)
GAMMA = 0.99  # Discount factor
ALPHA = 0.2  # Temperature parameter for soft Q-learning (will be auto-tuned)
CHI2_COEFF = 0.5  # Coefficient for chi-squared divergence regularization (original default)
LEARNING_RATE_CRITIC = 1e-4  # Reduced for stability
LEARNING_RATE_ACTOR = 1e-4  # Reduced for stability
LEARNING_RATE_ALPHA = 3e-4  # For temperature tuning
BATCH_SIZE = 256
BUFFER_SIZE = 1000000
INITIAL_RANDOM_STEPS = 10000
LEARNING_STARTS = 10000
TOTAL_TIMESTEPS = 1000000
EVAL_FREQUENCY = 5000
ACTOR_UPDATE_FREQUENCY = 1  # More conservative: update actor every step
TARGET_UPDATE_FREQUENCY = 1  # More frequent target updates for stability
TAU = 0.005  # Soft update coefficient
MAX_GRAD_NORM = 1.0  # Gradient clipping

# IQ-Learn specific
LOSS_TYPE = "value_expert"  # Options: "value", "value_expert", "v0"
DIVERGENCE_TYPE = "chi"  # Chi-squared is more stable than Wasserstein
USE_TARGET_NETWORK = True
REGULARIZE = False  # DISABLED - chi2 divergence already provides regularization


# ========================== Networks ==========================

class DoubleQCritic(nn.Module):
    """
    Double Q-Critic network for IQ-Learn.
    Outputs two Q-values to reduce overestimation bias.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs, action, both=False):
        """
        Forward pass through critic.
        
        Args:
            obs: State observations
            action: Actions taken
            both: If True, return both Q-values; else return minimum
        
        Returns:
            Q-value(s)
        """
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        
        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """
    Value network V(s) for computing soft value function.
    Used in IQ-Learn to compute V(s) = E_a~π[Q(s,a) - α*log π(a|s)]
    """
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        return self.net(obs)


class SquashedGaussianActor(nn.Module):
    """
    Actor network with squashed Gaussian policy (tanh output).
    Outputs mean and log_std for a Gaussian distribution.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs, deterministic=False, return_log_prob=False):
        """
        Forward pass through actor.
        
        Args:
            obs: State observations
            deterministic: If True, return mean action
            return_log_prob: If True, also return log probability
        
        Returns:
            action, (optional) log_prob
        """
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            if return_log_prob:
                return action, None
            return action
        
        # Sample from Gaussian
        dist = Normal(mean, std)
        z = dist.rsample()  # Reparameterization trick
        action = torch.tanh(z)
        
        if return_log_prob:
            # Compute log probability with tanh correction
            log_prob = dist.log_prob(z)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            return action, log_prob
        
        return action
    
    def get_log_prob(self, obs, action):
        """Get log probability of action given observation."""
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Inverse tanh to get pre-tanh action
        action_clamped = torch.clamp(action, -0.999, 0.999)
        z = torch.atanh(action_clamped)
        
        # Compute log probability
        dist = Normal(mean, std)
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return log_prob


# ========================== Replay Buffer ==========================

class ReplayBuffer:
    """
    Simple replay buffer for storing transitions.
    """
    def __init__(self, obs_dim, act_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, obs, next_obs, action, reward, done):
        """Add a transition to the buffer."""
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size, device):
        """Sample a batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        batch = (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device)
        )
        
        return batch
    
    def get_size(self):
        """Return current buffer size."""
        return self.size
    
    def get_all_data(self):
        """Get all data from buffer."""
        return (
            self.obs[:self.size],
            self.next_obs[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.dones[:self.size]
        )


# ========================== Evaluation Metrics ==========================

def compute_action_mse_mae(agent, obs, true_actions, device):
    """
    Compute MSE and MAE between agent's predicted actions and true expert actions.
    
    Args:
        agent: IQLearnAgent instance
        obs: Observations (numpy array)
        true_actions: True expert actions (numpy array)
        device: Device for computation
    
    Returns:
        mse: Mean Squared Error
        mae: Mean Absolute Error
    """
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).to(device)
        
        # Get predicted actions (deterministic)
        predicted_actions = []
        batch_size = 256
        for i in range(0, len(obs), batch_size):
            batch_obs = obs_tensor[i:i+batch_size]
            batch_actions = agent.actor(batch_obs, deterministic=True)
            predicted_actions.append(batch_actions.cpu().numpy())
        
        predicted_actions = np.concatenate(predicted_actions, axis=0)
    
    # Compute metrics
    mse = np.mean((predicted_actions - true_actions) ** 2)
    mae = np.mean(np.abs(predicted_actions - true_actions))
    
    return mse, mae


def evaluate_policy_performance(agent, env, n_episodes=10, deterministic=True):
    """
    Evaluate policy performance in the environment.
    
    Args:
        agent: IQLearnAgent instance
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
    
    Returns:
        mean_reward: Mean episode reward
        std_reward: Standard deviation of rewards
        mean_length: Mean episode length
    """
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs = env_reset(env)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, _ = env_step(env, action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return np.mean(episode_rewards), np.std(episode_rewards), np.mean(episode_lengths)


# ========================== IQ-Learn Loss ==========================

def iq_loss(agent, current_Q, current_v, next_v, batch, gamma, alpha, args):
    """
    Calculate IQ-Learn loss.
    
    Based on the original implementation from:
    https://github.com/Div-Infinity/IQ-Learn/blob/main/iq_learn/iq.py
    
    Args:
        agent: IQLearn agent
        current_Q: Q(s, a) values
        current_v: V(s) values  
        next_v: V(s') values
        batch: Tuple of (obs, next_obs, action, reward, done, is_expert)
        gamma: Discount factor
        alpha: Temperature parameter
        args: Configuration arguments
    
    Returns:
        loss: Total IQ-Learn loss
        loss_dict: Dictionary with loss components
    """
    obs, next_obs, action, env_reward, done, is_expert = batch
    
    loss_dict = {}
    
    # Track value of initial expert states
    v0 = current_v[is_expert.squeeze(1), ...].mean()
    loss_dict['v0'] = v0.item()
    
    # Calculate 1st term for IQ loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - done) * gamma * next_v
    reward = current_Q - y
    
    # No reward clipping - let rewards flow naturally (original behavior)
    # reward = torch.clamp(reward, -10, 10)  # REMOVED to match original
    
    # Apply divergence-specific gradient
    if args['divergence'] == "hellinger":
        phi_grad = 1 / (1 + reward) ** 2
    elif args['divergence'] == "kl":
        phi_grad = torch.exp(-reward - 1)
    elif args['divergence'] == "kl2":
        phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
    elif args['divergence'] == "js":
        phi_grad = torch.exp(-reward) / (2 - torch.exp(-reward))
    elif args['divergence'] == "wasserstein":
        # Wasserstein distance: φ(x) = x, so φ'(x) = 1
        # This is equivalent to chi-squared for the gradient
        phi_grad = 1
    else:  # chi or default
        phi_grad = 1
    
    loss = -(phi_grad * reward[is_expert]).mean()
    loss_dict['softq_loss'] = loss.item()
    
    # Calculate 2nd term for IQ loss (different sampling strategies)
    if args['loss_type'] == "value":
        # Sample using all states (expert + policy)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()
    
    elif args['loss_type'] == "value_expert":
        # Sample using expert states only (preferred for offline)
        # E_(ρ_expert)[V(s) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_expert_loss'] = value_loss.item()
    
    elif args['loss_type'] == "v0":
        # Use initial state distribution
        # (1-γ)E_(ρ0)[V(s0)]
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
        loss_dict['v0_loss'] = v0_loss.item()
    
    # Add chi-squared divergence regularization
    if args['divergence'] == "chi":
        # Calculate regularization term using expert states
        chi2_loss = 1 / (4 * alpha) * (reward[is_expert] ** 2).mean()
        loss += args['chi2_coeff'] * chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()
    
    # Add regularization using all states (for online learning)
    if args['regularize']:
        reg_loss = 1 / (4 * alpha) * (reward ** 2).mean()
        loss += args['chi2_coeff'] * reg_loss
        loss_dict['regularize_loss'] = reg_loss.item()
    
    loss_dict['total_loss'] = loss.item()
    
    return loss, loss_dict


# ========================== IQ-Learn Agent ==========================

class IQLearnAgent:
    """
    IQ-Learn agent for imitation learning.
    """
    def __init__(
        self,
        obs_dim,
        act_dim,
        device='cpu',
        gamma=0.99,
        alpha=0.2,
        chi2_coeff=0.5,
        lr_critic=3e-4,
        lr_actor=3e-4,
        lr_alpha=3e-4,
        loss_type="value_expert",
        divergence="chi",
        use_target=True,
        regularize=True,
        tau=0.005
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.chi2_coeff = chi2_coeff
        self.loss_type = loss_type
        self.divergence = divergence
        self.use_target = use_target
        self.regularize = regularize
        self.tau = tau
        
        # Initialize networks
        self.critic = DoubleQCritic(obs_dim, act_dim).to(device)
        self.critic_target = DoubleQCritic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor = SquashedGaussianActor(obs_dim, act_dim).to(device)
        
        # Automatic entropy tuning
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        
        # Optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
    
    def getV(self, obs):
        """
        Calculate soft value function V(s) = E_a~π[Q(s,a) - α*log π(a|s)]
        
        IMPORTANT: Gradients flow through this function (not detached).
        This matches the original IQ-Learn implementation.
        """
        action, log_prob = self.actor(obs, return_log_prob=True)
        q_value = self.critic(obs, action)
        alpha = torch.exp(self.log_alpha)  # No clamping - original behavior
        v_value = q_value - alpha * log_prob
        return v_value
    
    def get_targetV(self, obs):
        """Calculate soft value using target network."""
        with torch.no_grad():
            action, log_prob = self.actor(obs, return_log_prob=True)
            q_value = self.critic_target(obs, action)
            alpha = torch.exp(self.log_alpha)  # No clamping - original behavior
            v_value = q_value - alpha * log_prob
        return v_value
    
    def select_action(self, obs, deterministic=False):
        """Select action using current policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor, deterministic=deterministic)
            action = action.cpu().numpy()[0]
        return action
    
    def update_critic(self, policy_batch, expert_batch):
        """
        Update critic using IQ-Learn loss.
        
        CRITICAL CHANGE: V(s) is NOT detached, allowing gradients to flow.
        This matches the original IQ-Learn implementation.
        
        Args:
            policy_batch: Batch from policy rollout buffer
            expert_batch: Batch from expert demonstration buffer
        
        Returns:
            loss_dict: Dictionary with loss values
        """
        policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
        expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
        
        # Concatenate policy and expert batches
        obs = torch.cat([policy_obs, expert_obs], dim=0)
        next_obs = torch.cat([policy_next_obs, expert_next_obs], dim=0)
        action = torch.cat([policy_action, expert_action], dim=0)
        done = torch.cat([policy_done, expert_done], dim=0)
        
        # Create expert mask
        batch_size = policy_obs.shape[0]
        is_expert = torch.cat([
            torch.zeros(batch_size, 1, device=self.device),
            torch.ones(batch_size, 1, device=self.device)
        ], dim=0).bool()
        
        # CRITICAL: Do NOT detach current_v! Gradients must flow through V(s)
        # This is the key difference from the broken implementation
        current_v = self.getV(obs)  # NOT detached!
        
        # Only detach next_v (target values should not propagate gradients)
        if self.use_target:
            next_v = self.get_targetV(next_obs)  # Already has torch.no_grad()
        else:
            with torch.no_grad():
                next_v = self.getV(next_obs)
        
        # Compute Q values for both critics
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        
        # Compute IQ-Learn loss for both critics
        args = {
            'loss_type': self.loss_type,
            'divergence': self.divergence,
            'chi2_coeff': self.chi2_coeff,
            'regularize': self.regularize
        }
        
        batch = (obs, next_obs, action, None, done, is_expert)
        
        q1_loss, loss_dict1 = iq_loss(
            self, current_Q1, current_v, next_v, batch, 
            self.gamma, self.alpha, args
        )
        q2_loss, loss_dict2 = iq_loss(
            self, current_Q2, current_v, next_v, batch,
            self.gamma, self.alpha, args
        )
        
        critic_loss = 0.5 * (q1_loss + q2_loss)
        
        # Update critic with gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Average loss dicts
        loss_dict = {}
        for key in loss_dict1.keys():
            loss_dict[key] = (loss_dict1[key] + loss_dict2[key]) / 2
        
        return loss_dict
    
    def update_actor(self, obs):
        """
        Update actor using soft policy improvement.
        
        Args:
            obs: State observations
        
        Returns:
            loss_dict: Dictionary with actor and alpha losses
        """
        # Sample actions and log probs
        action, log_prob = self.actor(obs, return_log_prob=True)
        q_value = self.critic(obs, action)
        alpha = torch.exp(self.log_alpha)  # No clamping - original behavior
        
        # Actor loss: maximize Q - α*log_prob
        actor_loss = (alpha * log_prob - q_value).mean()
        
        # Update actor with gradient clipping
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Update temperature (alpha) with gradient clipping
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optimizer.step()
        
        # No clamping of log_alpha - original behavior
        # (Alpha optimizer handles stability automatically)
        
        loss_dict = {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
        
        return loss_dict
    
    def soft_update_target(self):
        """Soft update of target network."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath):
        """Save agent networks."""
        torch.save({
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor': self.actor.state_dict(),
            'log_alpha': self.log_alpha,
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict()
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent networks."""
        checkpoint = torch.load(filepath)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.log_alpha = checkpoint['log_alpha']
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        print(f"Agent loaded from {filepath}")


# ========================== Training Loop ==========================

def train_iq_learn(
    env,
    eval_env,
    expert_buffer,
    train_expert_obs,
    train_expert_actions,
    test_expert_obs,
    test_expert_actions,
    agent,
    device,
    total_timesteps=1000000,
    initial_random_steps=10000,
    learning_starts=10000,
    batch_size=256,
    eval_frequency=5000,
    actor_update_frequency=2,
    target_update_frequency=2,
    save_path="iq_learn_ant.pth"
):
    """
    Train IQ-Learn agent with comprehensive evaluation.
    
    Args:
        env: Training environment
        eval_env: Evaluation environment
        expert_buffer: Buffer containing expert demonstrations
        train_expert_obs: Training set expert observations
        train_expert_actions: Training set expert actions
        test_expert_obs: Test set expert observations
        test_expert_actions: Test set expert actions
        agent: IQLearnAgent instance
        device: Device for computation
        total_timesteps: Total training timesteps
        initial_random_steps: Random exploration steps
        learning_starts: When to start learning
        batch_size: Batch size for updates
        eval_frequency: Evaluation frequency
        actor_update_frequency: Actor update frequency
        target_update_frequency: Target network update frequency
        save_path: Path to save best model
    
    Returns:
        agent: Trained agent
        training_metrics: Dictionary with training metrics
    """
    # Initialize policy buffer
    policy_buffer = ReplayBuffer(agent.obs_dim, agent.act_dim, max_size=BUFFER_SIZE)
    
    # Training metrics
    episode_rewards = []
    episode_steps_list = []
    best_eval_reward = -np.inf
    
    # Tracking metrics over time
    training_metrics = {
        'timesteps': [],
        'eval_rewards': [],
        'eval_stds': [],
        'train_mse': [],
        'train_mae': [],
        'test_mse': [],
        'test_mae': [],
        'episode_rewards': [],
        'critic_losses': [],
        'actor_losses': []
    }
    
    # Training loop
    obs = env_reset(env, seed=SEED)
    episode_reward = 0
    episode_steps = 0
    
    print(f"\n{'='*60}")
    print(f"Starting IQ-Learn Training")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Expert buffer size: {expert_buffer.get_size()}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for t in range(1, total_timesteps + 1):
        # Select action
        if t < initial_random_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)
        
        # Execute action
        next_obs, reward, done, info = env_step(env, action)
        
        # Store transition
        policy_buffer.add(obs, next_obs, action, reward, float(done))
        
        episode_reward += reward
        episode_steps += 1
        
        obs = next_obs
        
        # Reset if done
        if done:
            episode_rewards.append(episode_reward)
            episode_steps_list.append(episode_steps)
            obs = env_reset(env)
            episode_reward = 0
            episode_steps = 0
        
        # Start learning
        if t >= learning_starts and policy_buffer.get_size() >= batch_size:
            # Sample batches
            policy_batch = policy_buffer.sample(batch_size, device)
            expert_batch = expert_buffer.sample(batch_size, device)
            
            # Update critic
            critic_losses = agent.update_critic(policy_batch, expert_batch)
            
            # Update actor
            if t % actor_update_frequency == 0:
                obs_batch = torch.cat([policy_batch[0], expert_batch[0]], dim=0)
                actor_losses = agent.update_actor(obs_batch)
            
            # Update target network
            if t % target_update_frequency == 0:
                agent.soft_update_target()
        
        # Evaluation
        if t % eval_frequency == 0 and t >= learning_starts:
            # 1. Policy performance evaluation
            mean_eval_reward, std_eval_reward, mean_episode_length = evaluate_policy_performance(
                agent, eval_env, n_episodes=10, deterministic=True
            )
            
            # 2. Compute MSE/MAE on train set
            train_mse, train_mae = compute_action_mse_mae(
                agent, train_expert_obs, train_expert_actions, device
            )
            
            # 3. Compute MSE/MAE on test set
            test_mse, test_mae = compute_action_mse_mae(
                agent, test_expert_obs, test_expert_actions, device
            )
            
            # Store metrics
            training_metrics['timesteps'].append(t)
            training_metrics['eval_rewards'].append(mean_eval_reward)
            training_metrics['eval_stds'].append(std_eval_reward)
            training_metrics['train_mse'].append(train_mse)
            training_metrics['train_mae'].append(train_mae)
            training_metrics['test_mse'].append(test_mse)
            training_metrics['test_mae'].append(test_mae)
            training_metrics['critic_losses'].append(critic_losses.get('total_loss', 0))
            if 'actor_losses' in locals():
                training_metrics['actor_losses'].append(actor_losses.get('actor_loss', 0))
            
            # Print progress
            elapsed_time = time.time() - start_time
            mean_episode_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            
            print(f"{'='*70}")
            print(f"Timestep: {t}/{total_timesteps} | Time: {elapsed_time:.1f}s | Episodes: {len(episode_rewards)}")
            print(f"{'='*70}")
            print(f"POLICY PERFORMANCE:")
            print(f"  Mean Episode Reward (last 100): {mean_episode_reward:.2f}")
            print(f"  Eval Reward: {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")
            print(f"  Eval Episode Length: {mean_episode_length:.1f}")
            print(f"\nACTION PREDICTION METRICS:")
            print(f"  Train MSE: {train_mse:.6f} | Train MAE: {train_mae:.6f}")
            print(f"  Test MSE:  {test_mse:.6f} | Test MAE:  {test_mae:.6f}")
            print(f"  Generalization Gap (MSE): {test_mse - train_mse:.6f}")
            if t >= learning_starts:
                print(f"\nTRAINING LOSSES:")
                print(f"  Critic Loss: {critic_losses.get('total_loss', 0):.4f}")
                print(f"  V0 (Expert Value): {critic_losses.get('v0', 0):.4f}")
                if 'actor_losses' in locals():
                    print(f"  Actor Loss: {actor_losses.get('actor_loss', 0):.4f}")
                    print(f"  Alpha (Temperature): {actor_losses.get('alpha', 0):.4f}")
            print(f"{'='*70}\n")
            
            # Save best model (based on eval reward)
            if mean_eval_reward > best_eval_reward:
                best_eval_reward = mean_eval_reward
                agent.save(save_path)
                print(f"✅ New best model saved! Eval Reward: {best_eval_reward:.2f}")
                print(f"   Test MSE: {test_mse:.6f} | Test MAE: {test_mae:.6f}\n")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Eval Reward: {best_eval_reward:.2f}")
    print(f"Total Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"{'='*60}\n")
    
    # Store final episode rewards
    training_metrics['episode_rewards'] = episode_rewards
    
    return agent, training_metrics


# ========================== Main Script ==========================

def main():
    """Main training script."""
    print("\n" + "="*60)
    print("IQ-Learn for Safety Ant Environment")
    print("="*60 + "\n")
    
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Device: probe CUDA compatibility and fall back to CPU if necessary
    device = get_preferred_device()
    print(f"Using device: {device}\n")
    
    # Load expert demonstrations
    print(f"Loading expert demonstrations from {EXPERT_DATA_PATH}...")
    with open(EXPERT_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    def _to_array(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    if isinstance(data, dict) and "states" in data and "actions" in data:
        states = _to_array(data["states"]).astype(np.float32)
        actions = _to_array(data["actions"]).astype(np.float32)
        next_states = _to_array(data.get("next_states", data["states"])).astype(np.float32)
        rewards = _to_array(data.get("rewards", np.zeros((len(states), 1), dtype=np.float32))).astype(np.float32)
        dones = _to_array(data.get("dones", np.zeros((len(states), 1), dtype=np.float32))).astype(np.float32)
    elif isinstance(data, dict) and "trajectories" in data:
        state_chunks = []
        action_chunks = []
        next_state_chunks = []
        reward_chunks = []
        done_chunks = []
        for traj in data["trajectories"]:
            if not isinstance(traj, dict) or "states" not in traj or "actions" not in traj:
                continue
            traj_states = np.stack([_to_array(s) for s in traj["states"]]).astype(np.float32)
            traj_actions = np.stack([_to_array(a) for a in traj["actions"]]).astype(np.float32)
            traj_rewards = _to_array(traj.get("rewards", np.zeros((len(traj_actions), 1), dtype=np.float32))).astype(np.float32)
            traj_dones = _to_array(traj.get("dones", np.zeros((len(traj_actions), 1), dtype=np.float32))).astype(np.float32)
            n = min(len(traj_states) - 1, len(traj_actions))
            if n <= 0:
                continue
            state_chunks.append(traj_states[:n])
            action_chunks.append(traj_actions[:n])
            next_state_chunks.append(traj_states[1:n + 1])
            reward_chunks.append(traj_rewards[:n])
            done_chunks.append(traj_dones[:n])
        if not state_chunks:
            raise ValueError("No usable trajectories found in dataset")
        states = np.concatenate(state_chunks, axis=0)
        actions = np.concatenate(action_chunks, axis=0)
        next_states = np.concatenate(next_state_chunks, axis=0)
        rewards = np.concatenate(reward_chunks, axis=0)
        dones = np.concatenate(done_chunks, axis=0)
    else:
        raise ValueError("Unsupported dataset format: expected dict with states/actions or trajectories")

    total_transitions = len(states)
    obs_dim = states.shape[1]
    act_dim = actions.shape[1]

    print(f"Loaded {total_transitions} transitions\n")

    indices = np.random.permutation(total_transitions)
    n_train = int(total_transitions * TRAIN_TEST_SPLIT)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_obs = states[train_indices]
    train_actions = actions[train_indices]
    test_expert_obs = states[test_indices]
    test_expert_actions = actions[test_indices]

    # Create expert buffer from training transitions only
    expert_buffer = ReplayBuffer(obs_dim, act_dim, max_size=len(train_indices) + 1000)
    for idx in train_indices:
        expert_buffer.add(states[idx], next_states[idx], actions[idx], rewards[idx], dones[idx])

    train_expert_obs = train_obs
    train_expert_actions = train_actions
    
    print(f"Expert buffer size: {expert_buffer.get_size()}")
    print(f"Train samples: {len(train_expert_obs)}")
    print(f"Test samples: {len(test_expert_obs)}\n")
    
    # Create environments
    print(f"Creating environment: {ENV_NAME}")
    
    def make_safety_env():
        if gym is None:
            raise ImportError("gym not available. Install gym to use safety_gym environments")
        if safety_gym is None:
            raise ImportError("safety_gym not available. Install safety_gym to use Safety Gym environments")
        env = gym.make(ENV_NAME)
        env = SafetyEnvToGymEnvWrapper(env)
        env = Monitor(env)
        return env
    
    env = make_safety_env()
    eval_env = make_safety_env()
    
    env_reset(env, seed=SEED)
    env_reset(eval_env, seed=SEED + 100)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")
    
    # Create agent
    print("Initializing IQ-Learn agent...")
    agent = IQLearnAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        gamma=GAMMA,
        alpha=ALPHA,
        chi2_coeff=CHI2_COEFF,
        lr_critic=LEARNING_RATE_CRITIC,
        lr_actor=LEARNING_RATE_ACTOR,
        lr_alpha=LEARNING_RATE_ALPHA,
        loss_type=LOSS_TYPE,
        divergence=DIVERGENCE_TYPE,
        use_target=USE_TARGET_NETWORK,
        regularize=REGULARIZE,
        tau=TAU
    )
    
    print(f"Agent initialized with:")
    print(f"  - Observation dim: {obs_dim}")
    print(f"  - Action dim: {act_dim}")
    print(f"  - Gamma: {GAMMA}")
    print(f"  - Alpha: {ALPHA}")
    print(f"  - Loss type: {LOSS_TYPE}")
    print(f"  - Divergence: {DIVERGENCE_TYPE}\n")
    
    # Train agent
    agent, training_metrics = train_iq_learn(
        env=env,
        eval_env=eval_env,
        expert_buffer=expert_buffer,
        train_expert_obs=train_expert_obs,
        train_expert_actions=train_expert_actions,
        test_expert_obs=test_expert_obs,
        test_expert_actions=test_expert_actions,
        agent=agent,
        device=device,
        total_timesteps=TOTAL_TIMESTEPS,
        initial_random_steps=INITIAL_RANDOM_STEPS,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        eval_frequency=EVAL_FREQUENCY,
        actor_update_frequency=ACTOR_UPDATE_FREQUENCY,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        save_path="iq_learn_pointgoal_best.pth"
    )
    
    # Final comprehensive evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # 1. Policy performance evaluation
    print("\n1. Policy Performance (20 episodes):")
    final_rewards = []
    for i in range(20):
        obs = env_reset(eval_env)
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, _ = env_step(eval_env, action)
            episode_reward += reward
        final_rewards.append(episode_reward)
        print(f"   Episode {i+1}/20: Reward = {episode_reward:.2f}")
    
    print(f"\n   Policy Performance Summary:")
    print(f"   Mean Reward: {np.mean(final_rewards):.2f}")
    print(f"   Std Reward:  {np.std(final_rewards):.2f}")
    print(f"   Min Reward:  {np.min(final_rewards):.2f}")
    print(f"   Max Reward:  {np.max(final_rewards):.2f}")
    
    # 2. Final MSE/MAE on train and test sets
    print(f"\n2. Action Prediction Accuracy:")
    final_train_mse, final_train_mae = compute_action_mse_mae(
        agent, train_expert_obs, train_expert_actions, device
    )
    final_test_mse, final_test_mae = compute_action_mse_mae(
        agent, test_expert_obs, test_expert_actions, device
    )
    
    print(f"   Train Set:")
    print(f"     MSE: {final_train_mse:.6f}")
    print(f"     MAE: {final_train_mae:.6f}")
    print(f"   Test Set:")
    print(f"     MSE: {final_test_mse:.6f}")
    print(f"     MAE: {final_test_mae:.6f}")
    print(f"   Generalization Gap:")
    print(f"     MSE Gap: {final_test_mse - final_train_mse:.6f}")
    print(f"     MAE Gap: {final_test_mae - final_train_mae:.6f}")
    
    # 3. Save metrics to file
    print(f"\n3. Saving training metrics...")
    np.savez('iq_learn_training_metrics.npz',
             timesteps=training_metrics['timesteps'],
             eval_rewards=training_metrics['eval_rewards'],
             eval_stds=training_metrics['eval_stds'],
             train_mse=training_metrics['train_mse'],
             train_mae=training_metrics['train_mae'],
             test_mse=training_metrics['test_mse'],
             test_mae=training_metrics['test_mae'],
             critic_losses=training_metrics['critic_losses'],
             actor_losses=training_metrics['actor_losses'],
             final_train_mse=final_train_mse,
             final_train_mae=final_train_mae,
             final_test_mse=final_test_mse,
             final_test_mae=final_test_mae,
             final_mean_reward=np.mean(final_rewards),
             final_std_reward=np.std(final_rewards))
    print(f"   ✅ Metrics saved to: iq_learn_training_metrics.npz")
    
    # 4. Create summary plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        print(f"\n4. Creating training plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Eval Rewards
        ax = axes[0, 0]
        ax.plot(training_metrics['timesteps'], training_metrics['eval_rewards'], 'b-', linewidth=2)
        ax.fill_between(training_metrics['timesteps'],
                        np.array(training_metrics['eval_rewards']) - np.array(training_metrics['eval_stds']),
                        np.array(training_metrics['eval_rewards']) + np.array(training_metrics['eval_stds']),
                        alpha=0.3)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title('Policy Performance Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: MSE (Train vs Test)
        ax = axes[0, 1]
        ax.plot(training_metrics['timesteps'], training_metrics['train_mse'], 'g-', label='Train MSE', linewidth=2)
        ax.plot(training_metrics['timesteps'], training_metrics['test_mse'], 'r-', label='Test MSE', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Action Prediction MSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 3: MAE (Train vs Test)
        ax = axes[1, 0]
        ax.plot(training_metrics['timesteps'], training_metrics['train_mae'], 'g-', label='Train MAE', linewidth=2)
        ax.plot(training_metrics['timesteps'], training_metrics['test_mae'], 'r-', label='Test MAE', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Action Prediction MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Losses
        ax = axes[1, 1]
        ax.plot(training_metrics['timesteps'], training_metrics['critic_losses'], 'b-', label='Critic Loss', linewidth=2)
        if training_metrics['actor_losses']:
            ax.plot(training_metrics['timesteps'], training_metrics['actor_losses'], 'r-', label='Actor Loss', linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('iq_learn_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Training curves saved to: iq_learn_training_curves.png")
        plt.close()
        
    except ImportError:
        print(f"\n4. Matplotlib not available - skipping plots")
        print(f"   Install with: pip install matplotlib")
    
    env.close()
    eval_env.close()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nSaved Files:")
    print(f"  - Model checkpoint: iq_learn_pointgoal_best.pth")
    print(f"  - Training metrics: iq_learn_training_metrics.npz")
    print(f"  - Training curves: iq_learn_training_curves.png (if matplotlib available)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
