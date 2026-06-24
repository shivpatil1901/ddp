import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicShapedRewardNet

from SafeRL.tests.BC.bc_policy import GaussianPolicy
from SafeRL.tests.AIRL.airl_utils import SafetyEnvToGymEnvWrapper, make_env  # Import from airl_utils instead of airl

# --- Configuration ---
SEED = 42
ENV_NAME = "SafetyAntVelocity-v1"
EXPERT_DATA_PATH = "expert_demos_safety-ant/expert_demos_ant_velocity_300.npz" ## (Change Path here)
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test


# --- Custom PPO Policy using GaussianPolicy ---
class CustomGaussianActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy that uses the pre-trained GaussianPolicy as the actor network.
    """
    def __init__(self, observation_space, action_space, lr_schedule, bc_policy_state_dict=None, *args, **kwargs):
        # Store the BC policy state dict to load after initialization
        self.bc_policy_state_dict = bc_policy_state_dict
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
    
    def _build(self, lr_schedule):
        """
        Override _build to prevent parent class from overwriting our custom networks.
        """
        # Build the MLP extractor and our custom networks
        self._build_mlp_extractor()
        
        # Don't let parent class create action_net and value_net
        # We've already created them in _build_mlp_extractor
        
        # Set up optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(
                self.parameters(),
                lr=lr_schedule(1),
                **self.optimizer_kwargs
            )
    
    def _build_mlp_extractor(self) -> None:
        """Override to create a dummy mlp_extractor and our custom networks."""
        # Get dimensions
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        
        # Create a minimal mlp_extractor to satisfy stable-baselines3's requirements
        # This is a dummy that won't actually be used
        class DummyExtractor(nn.Module):
            def __init__(self, latent_dim):
                super().__init__()
                self.latent_dim_pi = latent_dim
                self.latent_dim_vf = latent_dim
                
            def forward(self, features):
                return features, features
        
        self.mlp_extractor = DummyExtractor(obs_dim)
        
        # Create the actor using GaussianPolicy with trainable std
        self.action_net = GaussianPolicy(obs_dim, act_dim, fixed_std=False)
        
        # Load pre-trained weights if provided
        if self.bc_policy_state_dict is not None:
            # Load the mean_net weights - compatible between fixed and trainable std
            mean_net_dict = {k.replace('mean_net.', ''): v for k, v in self.bc_policy_state_dict.items() if 'mean_net' in k}
            self.action_net.mean_net.load_state_dict(mean_net_dict)
            
            # Load log_std if available (will be zeros from BC, but PPO can train it)
            if 'log_std' in self.bc_policy_state_dict:
                self.action_net.log_std.data.copy_(self.bc_policy_state_dict['log_std'])
            print("✅ Loaded BC policy weights (mean_net) into PPO actor network")
        
        # Create a simple value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def _get_action_dist_from_latent(self, latent_pi):
        """
        Retrieve action distribution given the latent codes.
        """
        # Ensure latent_pi is float32
        latent_pi = latent_pi.float()
        mean, std = self.action_net(latent_pi)
        return Normal(mean, std)
    
    def _predict(self, observation, deterministic=False):
        """
        Get the action according to the policy for a given observation.
        """
        # Ensure observation is float32
        observation = observation.float()
        
        # Debug: Check what action_net returns
        result = self.action_net(observation)
        
        # Handle both single return (mean only) or tuple (mean, std)
        if isinstance(result, tuple):
            mean, std = result
        else:
            # If only mean is returned, create a fixed std
            mean = result
            std = torch.ones_like(mean)
        
        distribution = Normal(mean, std)
        
        if deterministic:
            actions = mean
        else:
            actions = distribution.sample()
        
        return actions
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass for the policy.
        Returns actions, values, and log probabilities.
        """
        # Ensure obs is float32
        obs = obs.float()
        
        # Get mean and std from the actor (GaussianPolicy)
        mean, std = self.action_net(obs)
        
        # Create distribution
        distribution = Normal(mean, std)
        
        # Sample or use mean
        if deterministic:
            actions = mean
        else:
            actions = distribution.sample()
        
        # Compute log prob
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        # Get value from critic
        values = self.value_net(obs)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy.
        """
        # Ensure obs is float32
        obs = obs.float()
        
        # Get mean and std from the actor
        mean, std = self.action_net(obs)
        
        # Create distribution
        distribution = Normal(mean, std)
        
        # Compute log prob and entropy
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        # Get value from critic
        values = self.value_net(obs)
        
        return values, log_prob, entropy
    
    def get_distribution(self, obs):
        """Get the current action distribution."""
        # Ensure obs is float32
        obs = obs.float()
        mean, std = self.action_net(obs)
        return Normal(mean, std)
    
    def predict_values(self, obs):
        """Predict values for given observations."""
        # Ensure obs is float32
        obs = obs.float()
        return self.value_net(obs)

# --- Load Expert Demonstrations ---
print(f"Loading expert demonstrations from {EXPERT_DATA_PATH}...")
data = np.load(EXPERT_DATA_PATH, allow_pickle=True)
trajectories = data["trajectories"]

# --- Train/Test Split ---
print(f"\n--- Splitting data into train/test ({TRAIN_TEST_SPLIT:.0%} train, {1-TRAIN_TEST_SPLIT:.0%} test) ---")
np.random.seed(SEED)
n_trajectories = len(trajectories)
indices = np.random.permutation(n_trajectories)
n_train = int(n_trajectories * TRAIN_TEST_SPLIT)

train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_trajectories = [trajectories[i] for i in train_indices]
test_trajectories = [trajectories[i] for i in test_indices]

print(f"Total trajectories: {n_trajectories}")
print(f"Training trajectories: {len(train_trajectories)}")
print(f"Test trajectories: {len(test_trajectories)}")

# Prepare data for BC training (using only train trajectories)
print("\n--- Preparing BC training data ---")
obs_list = []
act_list = []

for traj in train_trajectories:
    states = np.array(traj["states"])
    actions = np.array(traj["actions"])

    # Map obs[i] -> act[i]
    obs_list.append(states[:-1])
    act_list.append(actions)

all_obs = np.concatenate(obs_list)
all_acts = np.concatenate(act_list)

print(f"Loaded {len(all_obs)} (state, action) pairs from training set for BC policy.")

# Prepare data for AIRL (imitation format) - using only train trajectories
print("\n--- Preparing demonstrations for AIRL (training set) ---")
demonstrations_list = []
total_transitions = 0
for episode in train_trajectories:
    ep_states = np.array(episode['states'])
    ep_actions = np.array(episode['actions'])
    terminal = np.zeros(len(ep_actions), dtype=bool) 
    terminal[-1] = True
    demonstrations_list.append(
        types.Trajectory(
            obs=ep_states, acts=ep_actions, infos=None, terminal=terminal,
        ),
    )
    total_transitions += len(ep_actions)
print(f"Loaded {len(demonstrations_list)} training trajectories with {total_transitions} total transitions.")

# Flatten trajectories for AIRL
print("Manually flattening training trajectories...")
all_obs_airl = np.concatenate([traj.obs[:-1] for traj in demonstrations_list])
all_acts_airl = np.concatenate([traj.acts for traj in demonstrations_list])
all_next_obs = np.concatenate([traj.obs[1:] for traj in demonstrations_list])
all_dones = np.concatenate([traj.terminal for traj in demonstrations_list])
all_infos = np.array([{} for _ in range(len(all_obs_airl))])
demonstrations = Transitions(
    obs=all_obs_airl, acts=all_acts_airl, next_obs=all_next_obs, dones=all_dones, infos=all_infos,
)
print(f"Flattening complete. Created Transitions object with {len(all_obs_airl)} steps.")

# Prepare test data for evaluation
print("\n--- Preparing test data ---")
test_obs_list = []
test_act_list = []

for traj in test_trajectories:
    states = np.array(traj["states"])
    actions = np.array(traj["actions"])
    test_obs_list.append(states[:-1])
    test_act_list.append(actions)

test_obs = np.concatenate(test_obs_list)
test_acts = np.concatenate(test_act_list)
print(f"Loaded {len(test_obs)} (state, action) pairs from test set for evaluation.")

# --- Train BC Policy from Scratch ---
print("\n--- Training BC policy from scratch on training data ---")
from torch.utils.data import Dataset, DataLoader, random_split

class ExpertDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]

def gaussian_nll(actions, mean, std):
    """Negative log likelihood for diagonal Gaussian"""
    var = std ** 2
    nll = 0.5 * (((actions - mean) ** 2) / var + torch.log(2 * torch.pi * var))
    return nll.sum(dim=1).mean()

obs_dim = all_obs.shape[1]
act_dim = all_acts.shape[1]

# Create BC policy
bc_policy = GaussianPolicy(obs_dim, act_dim, fixed_std=True)
optimizer = torch.optim.Adam(bc_policy.parameters(), lr=1e-4)

# Create dataset and dataloader
bc_dataset = ExpertDataset(all_obs, all_acts)
val_split = 0.1
val_size = int(len(bc_dataset) * val_split)
train_size = len(bc_dataset) - val_size
bc_train_data, bc_val_data = random_split(bc_dataset, [train_size, val_size])

bc_train_loader = DataLoader(bc_train_data, batch_size=256, shuffle=True)
bc_val_loader = DataLoader(bc_val_data, batch_size=256, shuffle=False)

# Train BC
BC_EPOCHS = 10
best_val_loss = float('inf')

print(f"Training BC on {train_size} samples, validating on {val_size} samples...")
for epoch in range(BC_EPOCHS):
    # Train
    bc_policy.train()
    train_loss = 0
    for batch_obs, batch_act in bc_train_loader:
        mean, std = bc_policy(batch_obs)
        loss = gaussian_nll(batch_act, mean, std)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validate
    bc_policy.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_obs, batch_act in bc_val_loader:
            mean, std = bc_policy(batch_obs)
            loss = gaussian_nll(batch_act, mean, std)
            val_loss += loss.item()
    
    train_loss /= len(bc_train_loader)
    val_loss /= len(bc_val_loader)
    
    print(f"Epoch {epoch+1}/{BC_EPOCHS} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(bc_policy.state_dict(), "bc_gaussian_policy_train_set.pth")
        print("   -> best BC model saved")

# Load best BC model
bc_policy.load_state_dict(torch.load("bc_gaussian_policy_train_set.pth"))
bc_policy.eval()
print(f"✅ BC policy trained and loaded from bc_gaussian_policy_ttrain_set.pth")

# --- Create Environments ---
print(f"\nCreating training environment: {ENV_NAME}")
N_ENVS = 8
# Define the wrapper order
train_post_wrappers = [
    lambda env, rank: SafetyEnvToGymEnvWrapper(env),  # Convert 6-tuple to 5-tuple
    lambda env, rank: Monitor(env),                   # Add Monitor
    lambda env, rank: RolloutInfoWrapper(env)         # Add imitation's wrapper
]
# Create a list of env-making functions
train_env_fns = [make_env(ENV_NAME, SEED, i, train_post_wrappers) for i in range(N_ENVS)]
# Pass them to DummyVecEnv
train_env = DummyVecEnv(train_env_fns)

print("Creating evaluation environment...")
N_EVAL_ENVS = 10
eval_post_wrappers = [
    lambda env, rank: SafetyEnvToGymEnvWrapper(env),
    lambda env, rank: Monitor(env),
    lambda env, rank: RolloutInfoWrapper(env)
]
eval_env_fns = [make_env(ENV_NAME, SEED, i, eval_post_wrappers) for i in range(N_EVAL_ENVS)]
eval_env = DummyVecEnv(eval_env_fns)

# --- Set up the Learner (PPO) with Custom Policy ---
print("\nSetting up PPO learner with BC-initialized policy...")

# Create a policy class with the BC weights baked in
class CustomPolicyWithBC(CustomGaussianActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, bc_policy_state_dict=bc_policy.state_dict(), **kwargs)

learner = PPO(
    env=train_env,
    policy=CustomPolicyWithBC,
    n_steps=2048,           
    batch_size=1024,
    ent_coef=0.0,
    learning_rate=5e-5,
    gamma=0.99,
    clip_range=0.2,
    vf_coef=0.5,
    n_epochs=10,
    seed=SEED,
    device="cpu"
)

print("✅ PPO learner initialized with BC policy weights")

# For now, we'll evaluate the random PPO policy before AIRL
print("\nEvaluating PPO policy BEFORE AIRL training...")
learner_rewards_before_training, _ = evaluate_policy(
    learner, eval_env, 10, return_episode_rewards=True,
)
print(f"Mean reward (before AIRL): {np.mean(learner_rewards_before_training):.2f}")

# --- Set up the Reward Net ---
print("\nSetting up Reward Net...")
reward_net = BasicShapedRewardNet(
    observation_space=train_env.observation_space,
    action_space=train_env.action_space,
    normalize_input_layer=None,
)

# --- Set up the AIRL Trainer ---
print("\nSetting up AIRL Trainer...")
airl_trainer = AIRL(
    demonstrations=demonstrations, 
    demo_batch_size=1024, 
    gen_replay_buffer_capacity=2048, 
    n_disc_updates_per_round=4,
    venv=train_env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)

# --- Train the AIRL model ---
print("\n--- Starting AIRL Training ---")
airl_trainer.train(total_timesteps=1_000_000)
print("✅ AIRL Training complete.")

# --- Evaluate the final trained learner ---
print("\nEvaluating policy AFTER AIRL training...")
learner_rewards_after_airl, _ = evaluate_policy(
    learner, eval_env, 10, return_episode_rewards=True,
)

print("-" * 40)
print(f"Mean reward (before AIRL): {np.mean(learner_rewards_before_training):.2f}")
print(f"Mean reward (after AIRL): {np.mean(learner_rewards_after_airl):.2f}")
print("-" * 40)

# --- Verify and save the reward function ---
print("\nVerifying learned reward function against first training trajectory...")
first_traj_data = train_trajectories[0]
true_rewards = np.array(first_traj_data['rewards'])
learned_reward_net = airl_trainer._reward_net

obs = np.array(first_traj_data['states'])[:-1]
acts = np.array(first_traj_data['actions'])
next_obs = np.array(first_traj_data['states'])[1:]
dones = np.zeros(len(acts), dtype=bool)
dones[-1] = True

device = learner.device
predicted_rewards = learned_reward_net.predict_processed(
    state=torch.as_tensor(obs, device=device),
    action=torch.as_tensor(acts, device=device),
    next_state=torch.as_tensor(next_obs, device=device),
    done=torch.as_tensor(dones, device=device),
)
predicted_rewards_np = predicted_rewards

print(f"\nComparison of first 10 steps of the first training episode:")
print("True Reward | Predicted Reward")
for i in range(min(10, len(true_rewards))):
    print(f"{true_rewards[i]:11.4f} | {predicted_rewards_np[i]:17.4f}")

# --- Save the rewards and network ---
print("\nSaving rewards to learned_reward_comparison.npz...")
np.savez(
    "learned_reward_comparison.npz",
    true_rewards=true_rewards[:-1],
    predicted_rewards=predicted_rewards_np
)
print("✅ Rewards saved.")

print("\nSaving reward network to reward_net_train.pt...")
torch.save(learned_reward_net.state_dict(), "reward_net_train.pt")
print("✅ Reward network saved.")

# --- Compare Final Learned Policy Actions vs Expert Actions (TRAINING SET) ---
print("\n=== Evaluating on TRAINING SET ===")
print("Computing MSE and MAD between learned policy actions and expert actions (training)...")

# Expert observation and action arrays from "demonstrations" (training set)
expert_obs_train = demonstrations.obs
expert_actions_train = demonstrations.acts

# Get learned policy actions
with torch.no_grad():
    learned_actions_tensor_train, _, _ = learner.policy(torch.tensor(expert_obs_train, dtype=torch.float32))
    learned_actions_train = learned_actions_tensor_train.cpu().numpy()

# Compute MSE
mse_train = np.mean((learned_actions_train - expert_actions_train) ** 2)

# Compute MAD (Mean Absolute Deviation = MAE)
mad_train = np.mean(np.abs(learned_actions_train - expert_actions_train))

print(f"✅ [TRAIN] MSE between learned and expert actions: {mse_train:.6f}")
print(f"✅ [TRAIN] MAD between learned and expert actions: {mad_train:.6f}")

# --- Per-Dimension MSE (Training) ---
print("\n[TRAIN] Per-Dimension MSE between learned and expert actions:")
differences_train = learned_actions_train - expert_actions_train
per_dim_mse_train = np.mean(differences_train ** 2, axis=0)

for i, mse_value in enumerate(per_dim_mse_train):
    print(f"Dimension {i}: MSE = {mse_value:.6f}")

# --- Compare Final Learned Policy Actions vs Expert Actions (TEST SET) ---
print("\n=== Evaluating on TEST SET ===")
print("Computing MSE and MAD between learned policy actions and expert actions (test)...")

# Get learned policy actions on test set
with torch.no_grad():
    learned_actions_tensor_test, _, _ = learner.policy(torch.tensor(test_obs, dtype=torch.float32))
    learned_actions_test = learned_actions_tensor_test.cpu().numpy()

# Compute MSE
mse_test = np.mean((learned_actions_test - test_acts) ** 2)

# Compute MAD (Mean Absolute Deviation = MAE)
mad_test = np.mean(np.abs(learned_actions_test - test_acts))

print(f"✅ [TEST] MSE between learned and expert actions: {mse_test:.6f}")
print(f"✅ [TEST] MAD between learned and expert actions: {mad_test:.6f}")

# --- Show side-by-side comparison for first 15 steps (TEST SET) ---
print("\n[TEST] First 15 action comparisons (Expert vs Learned):")
print("Step | Expert Action               | Learned Action")
print("-" * 65)

num_show = 15
for i in range(min(num_show, len(test_acts))):
    exp_a = test_acts[i]
    lea_a = learned_actions_test[i]
    print(f"{i:4d} | {exp_a} | {lea_a}")

# --- Per-Dimension MSE (Test) ---
print("\n[TEST] Per-Dimension MSE between learned and expert actions:")
differences_test = learned_actions_test - test_acts
per_dim_mse_test = np.mean(differences_test ** 2, axis=0)

for i, mse_value in enumerate(per_dim_mse_test):
    print(f"Dimension {i}: MSE = {mse_value:.6f}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY: Train vs Test Performance")
print("=" * 60)
print(f"Training Set:")
print(f"  MSE: {mse_train:.6f}")
print(f"  MAD: {mad_train:.6f}")
print(f"\nTest Set:")
print(f"  MSE: {mse_test:.6f}")
print(f"  MAD: {mad_test:.6f}")
print(f"\nGeneralization Gap:")
print(f"  MSE difference: {abs(mse_test - mse_train):.6f}")
print(f"  MAD difference: {abs(mad_test - mad_train):.6f}")
print("=" * 60)