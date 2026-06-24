"""
Conditional Variational Autoencoder (CVAE) for Behavioral Cloning

This implementation handles multi-modal action distributions by learning
a latent variable model: p(a|s) = ∫ p(a|s,z) p(z|s) dz

The CVAE learns to:
1. Encode expert actions into a latent distribution: q(z|s,a) (encoder)
2. Decode latent codes to actions: p(a|s,z) (decoder)  
3. Learn a prior over latents: p(z|s) (prior network)

This is particularly useful when expert demonstrations have multiple
valid action modes for the same state (multi-modality).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import Tuple


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
EXPERT_DATA_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointButton1_s0.pickle"
TRAIN_TEST_SPLIT = 0.8

# CVAE Hyperparameters - ANTI-COLLAPSE SETTINGS
LATENT_DIM = 16  # Dimension of latent space z (increased for more capacity)
HIDDEN_DIM = 512  # Hidden layer size
LEARNING_RATE = 3e-5  # Lower learning rate for stability
BATCH_SIZE = 256
EPOCHS = 40

# Beta annealing (prevent early KL collapse)
BETA_START = 0.0
BETA_END = 0.01  # Much lower than 0.5 to prevent mode collapse
BETA_ANNEAL_EPOCHS = 40

# Free bits threshold (force minimum KL per dimension)
FREE_BITS = 0.4

def _select_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        probe = torch.zeros(1, device="cuda")
        del probe
        return torch.device("cuda")
    except Exception as exc:
        print(f"CUDA probe failed, falling back to CPU: {exc}")
        return torch.device("cpu")


DEVICE = _select_device()
print(f"Using device: {DEVICE}")

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# 2. Load and Prepare Data
# ============================================================

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
elif isinstance(data, dict) and "trajectories" in data:
    state_chunks = []
    action_chunks = []
    for traj in data["trajectories"]:
        if not isinstance(traj, dict) or "states" not in traj or "actions" not in traj:
            continue
        traj_states = np.stack([_to_array(s) for s in traj["states"]]).astype(np.float32)
        traj_actions = np.stack([_to_array(a) for a in traj["actions"]]).astype(np.float32)
        n = min(len(traj_states), len(traj_actions))
        state_chunks.append(traj_states[:n])
        action_chunks.append(traj_actions[:n])
    if not state_chunks:
        raise ValueError("No usable trajectories found in dataset")
    states = np.concatenate(state_chunks, axis=0).astype(np.float32)
    actions = np.concatenate(action_chunks, axis=0).astype(np.float32)
else:
    raise ValueError("Unsupported dataset format: expected dict with states/actions or trajectories")

print(f"Loaded {len(states)} transitions.")
print(f"State dim: {states.shape[1]} | Action dim: {actions.shape[1]}")

# Transition-level train/test split
print(f"\n--- Splitting data ({TRAIN_TEST_SPLIT:.0%} train, {1-TRAIN_TEST_SPLIT:.0%} test) ---")
indices = np.random.permutation(len(states))
n_train = int(len(states) * TRAIN_TEST_SPLIT)
train_indices = indices[:n_train]
test_indices = indices[n_train:]

train_obs = states[train_indices]
train_acts = actions[train_indices]
test_obs = states[test_indices]
test_acts = actions[test_indices]

print(f"Training samples: {len(train_obs)}")
print(f"Test samples: {len(test_obs)}")

print(f"Training samples: {len(train_obs)}")
print(f"Test samples: {len(test_obs)}")

# Verify shapes
print(f"\nData shapes after processing:")
print(f"  train_obs: {train_obs.shape}")
print(f"  train_acts: {train_acts.shape}")
print(f"  test_obs: {test_obs.shape}")
print(f"  test_acts: {test_acts.shape}")

OBS_DIM = train_obs.shape[1]
ACT_DIM = train_acts.shape[1]
print(f"\nObservation dim: {OBS_DIM}, Action dim: {ACT_DIM}")

# ============================================================
# Linear Normalization Functions
# ============================================================

def linear_normalize(actions, min_val, max_val, safety_margin):
    """
    Linear normalization to [-1, 1] range with safety margin.
    
    Args:
        actions: Input actions to normalize
        min_val: Minimum value in the original range
        max_val: Maximum value in the original range
        safety_margin: Margin to shrink the range (e.g., 0.001 for [-0.999, 0.999])
    """
    # Scale to [0, 1]
    denom = max_val - min_val + 1e-8
    normalized = (actions - min_val) / denom
    
    # Scale to [-1, 1]
    normalized = 2.0 * normalized - 1.0
    
    # Shrink slightly to safe range (e.g. [-0.999, 0.999])
    normalized = normalized * (1.0 - safety_margin)
    return normalized


def inverse_linear_normalize(normalized_actions, min_val, max_val, safety_margin):
    """Reverse the linear normalization to get actions in original scale."""
    # Reverse the shrink to [-1, 1]
    unshrunk = normalized_actions / (1.0 - safety_margin)
    
    # Reverse scaling from [-1, 1] to [0, 1]
    unscaled_01 = (unshrunk + 1.0) / 2.0
    
    # Reverse scaling from [0, 1] to original range
    denom = max_val - min_val + 1e-8
    original_actions = unscaled_01 * denom + min_val
    
    return original_actions


# Compute normalization parameters from training data
obs_mean = train_obs.mean(axis=0)
obs_std = train_obs.std(axis=0) + 1e-8

# For actions, use linear normalization instead of z-score
action_min = train_acts.min()
action_max = train_acts.max()
safety_margin = 1e-3  # 0.001 margin → range becomes [-0.999, 0.999]

print(f"\n--- Normalization Statistics ---")
print(f"Observation mean: {obs_mean[:5]} ...")
print(f"Observation std:  {obs_std[:5]} ...")
print(f"Action min: {action_min}")
print(f"Action max: {action_max}")
print(f"Safety margin: {safety_margin}")

# Normalize data
train_obs = (train_obs - obs_mean) / obs_std
test_obs = (test_obs - obs_mean) / obs_std
train_acts = linear_normalize(train_acts, action_min, action_max, safety_margin)
test_acts = linear_normalize(test_acts, action_min, action_max, safety_margin)

print(f"\nAfter normalization:")
print(f"  Train obs: mean={train_obs.mean():.4f}, std={train_obs.std():.4f}")
print(f"  Train acts: mean={train_acts.mean():.4f}, std={train_acts.std():.4f}")
print(f"  Train acts range: [{train_acts.min():.4f}, {train_acts.max():.4f}]")

# Persist normalization stats for standalone environment evaluation
np.savez(
    "bc_cvae_policy_stats_button.npz",
    obs_mean=obs_mean,
    obs_std=obs_std,
    action_min=np.asarray(action_min, dtype=np.float32),
    action_max=np.asarray(action_max, dtype=np.float32),
    safety_margin=np.asarray(safety_margin, dtype=np.float32),
    obs_dim=np.asarray(OBS_DIM, dtype=np.int32),
    act_dim=np.asarray(ACT_DIM, dtype=np.int32),
    latent_dim=np.asarray(LATENT_DIM, dtype=np.int32),
    hidden_dim=np.asarray(HIDDEN_DIM, dtype=np.int32),
)
print("Saved normalization stats to bc_cvae_policy_stats_button.npz")

# Verify normalization worked
print("\n--- Verifying Normalization ---")
print(f"Train actions after linear normalization:")
print(f"  Mean: {train_acts.mean(axis=0)}")
print(f"  Std:  {train_acts.std(axis=0)}")
print(f"  Min:  {train_acts.min(axis=0)}")
print(f"  Max:  {train_acts.max(axis=0)}")
print(f"  Expected range: approximately [-0.999, 0.999]")

assert train_acts.min() >= -1.0 and train_acts.max() <= 1.0, "Actions outside [-1, 1] range!"
print("✅ Linear normalization verified!")


# ============================================================
# 3. Dataset
# ============================================================

class ExpertDataset(Dataset):
    def __init__(self, obs, acts):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


# Create datasets and loaders
train_dataset = ExpertDataset(train_obs, train_acts)
test_dataset = ExpertDataset(test_obs, test_acts)

# Further split training data for validation
val_split = 0.1
val_size = int(len(train_dataset) * val_split)
train_size = len(train_dataset) - val_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")


# ============================================================
# 4. CVAE Model Architecture
# ============================================================

class Encoder(nn.Module):
    """
    Encoder: q(z|s,a)
    Takes state and action as input, outputs mean and log_std of latent z
    """
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logstd_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        mean = self.mean_layer(h)
        logstd = self.logstd_layer(h)
        return mean, logstd


class Decoder(nn.Module):
    """
    Decoder: p(a|s,z)
    Takes state and latent z as input, outputs action
    """
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )
    
    def forward(self, state, latent):
        x = torch.cat([state, latent], dim=-1)
        action = self.net(x)
        return action


class PriorNetwork(nn.Module):
    """
    Prior: p(z|s)
    Takes state as input, outputs mean and log_std of latent z
    Used during inference when we don't have the expert action
    """
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logstd_layer = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, state):
        h = self.net(state)
        mean = self.mean_layer(h)
        logstd = self.logstd_layer(h)
        return mean, logstd


class CVAE(nn.Module):
    """
    Complete Conditional VAE model
    """
    def __init__(self, obs_dim, act_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        
        self.encoder = Encoder(obs_dim, act_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(obs_dim, act_dim, latent_dim, hidden_dim)
        self.prior = PriorNetwork(obs_dim, latent_dim, hidden_dim)
        
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean, logstd):
        """Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)"""
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def forward(self, state, action):
        """
        Forward pass during training
        Returns: reconstructed action, encoder params, prior params
        """
        # Encode: q(z|s,a)
        enc_mean, enc_logstd = self.encoder(state, action)
        
        # Sample latent
        z = self.reparameterize(enc_mean, enc_logstd)
        
        # Decode: p(a|s,z)
        recon_action = self.decoder(state, z)
        
        # Prior: p(z|s)
        prior_mean, prior_logstd = self.prior(state)
        
        return recon_action, enc_mean, enc_logstd, prior_mean, prior_logstd
    
    def sample_action(self, state, num_samples=1, use_mean=False):
        """
        Sample action at test time using prior: p(a|s) = ∫ p(a|s,z) p(z|s) dz
        
        Args:
            state: observation
            num_samples: number of action samples to generate
            use_mean: if True, use mean of prior (deterministic)
        """
        self.eval()
        with torch.no_grad():
            # Get prior distribution
            prior_mean, prior_logstd = self.prior(state)
            
            if use_mean:
                # Use mean of prior (deterministic)
                z = prior_mean
            else:
                # Sample from prior
                z = self.reparameterize(prior_mean, prior_logstd)
            
            # Decode
            if num_samples > 1:
                # Sample multiple actions
                state_expanded = state.repeat(num_samples, 1)
                z_samples = self.reparameterize(
                    prior_mean.repeat(num_samples, 1),
                    prior_logstd.repeat(num_samples, 1)
                )
                actions = self.decoder(state_expanded, z_samples)
                return actions
            else:
                action = self.decoder(state, z)
                return action


# ============================================================
# 5. Loss Functions
# ============================================================

def cvae_loss(recon_action, true_action, enc_mean, enc_logstd, 
              prior_mean, prior_logstd, beta=1.0, free_bits=0.0):
    """
    CVAE Loss with Free Bits to prevent mode collapse
    
    Free bits: Ensures each latent dimension maintains minimum information
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_action, true_action, reduction='sum')
    
    # KL divergence per dimension (don't sum yet)
    enc_var = torch.exp(2 * enc_logstd)
    prior_var = torch.exp(2 * prior_logstd)
    
    kl_per_dim = (
        prior_logstd - enc_logstd +
        (enc_var + (enc_mean - prior_mean) ** 2) / (2 * prior_var) -
        0.5
    )  # Shape: (batch, latent_dim)
    
    # Apply free bits: max(KL_per_dim, free_bits)
    if free_bits > 0:
        kl_per_dim = torch.maximum(
            kl_per_dim,
            torch.full_like(kl_per_dim, free_bits)
        )
    
    kl_div = kl_per_dim.sum()
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


# ============================================================
# 6. Training Loop
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CVAE WITH MODE COLLAPSE PREVENTION")
print("=" * 70)

# Initialize model
cvae = CVAE(OBS_DIM, ACT_DIM, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

# Training metrics
train_losses = []
val_losses = []
kl_divergences = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # Beta annealing schedule
    if epoch < BETA_ANNEAL_EPOCHS:
        current_beta = BETA_START + (BETA_END - BETA_START) * (epoch / BETA_ANNEAL_EPOCHS)
    else:
        current_beta = BETA_END
    
    # TRAINING
    cvae.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0
    
    for batch_obs, batch_act in train_loader:
        batch_obs = batch_obs.to(DEVICE)
        batch_act = batch_act.to(DEVICE)
        
        # Forward pass
        recon_act, enc_mean, enc_logstd, prior_mean, prior_logstd = cvae(batch_obs, batch_act)
        
        # Compute loss with free bits
        loss, recon_loss, kl_loss = cvae_loss(
            recon_act, batch_act, enc_mean, enc_logstd,
            prior_mean, prior_logstd, 
            beta=current_beta,
            free_bits=FREE_BITS
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()
    
    # VALIDATION
    cvae.eval()
    val_loss = 0
    val_recon = 0
    val_kl = 0
    
    with torch.no_grad():
        for batch_obs, batch_act in val_loader:
            batch_obs = batch_obs.to(DEVICE)
            batch_act = batch_act.to(DEVICE)
            
            recon_act, enc_mean, enc_logstd, prior_mean, prior_logstd = cvae(batch_obs, batch_act)
            
            loss, recon_loss, kl_loss = cvae_loss(
                recon_act, batch_act, enc_mean, enc_logstd,
                prior_mean, prior_logstd, 
                beta=current_beta,
                free_bits=FREE_BITS
            )
            
            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()
    
    # Average losses
    avg_train_loss = epoch_loss / len(train_loader.dataset)
    avg_train_recon = epoch_recon / len(train_loader.dataset)
    avg_train_kl = epoch_kl / len(train_loader.dataset)
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_recon = val_recon / len(val_loader.dataset)
    avg_val_kl = val_kl / len(val_loader.dataset)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    kl_divergences.append(avg_train_kl)
    
    # Print progress with KL monitoring
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Beta={current_beta:.4f} | "
              f"Train: {avg_train_loss:.4f} (recon: {avg_train_recon:.4f}, kl: {avg_train_kl:.4f}) | "
              f"Val: {avg_val_loss:.4f}")
        
        # Warning if KL too low
        if avg_train_kl < FREE_BITS * LATENT_DIM * 0.5:
            print(f"  WARNING: KL={avg_train_kl:.4f} is very low (target: >{FREE_BITS * LATENT_DIM * 0.5:.2f})")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(cvae.state_dict(), "bc_cvae_policy_button.pth")
        if (epoch + 1) % 20 == 0:
            print(f"  Best model saved (val_loss: {best_val_loss:.4f})")

print("\nTraining complete!")
print(f"Final KL divergence: {kl_divergences[-1]:.4f}")
print(f"Target KL (for healthy latent): >{FREE_BITS * LATENT_DIM:.2f}")
if kl_divergences[-1] < FREE_BITS * LATENT_DIM * 0.5:
    print("WARNING: KL is too low - model may have collapsed!")
else:
    print("KL divergence is healthy - latent space is being used")


# ============================================================
# 7. Visualize Latent Space and Action Diversity
# ============================================================

print("\n" + "=" * 70)
print("ANALYZING ACTION DIVERSITY")
print("=" * 70)

# Sample multiple actions for the same state
n_test_states = 10
n_samples_per_state = 20

test_state_indices = np.random.choice(len(test_obs), n_test_states, replace=False)

print(f"\nSampling {n_samples_per_state} actions for {n_test_states} test states...")

action_variances = []

for idx in test_state_indices:
    state = torch.tensor(test_obs[idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    true_action = test_acts[idx]
    
    # Sample multiple actions
    sampled_actions = cvae.sample_action(state, num_samples=n_samples_per_state, use_mean=False)
    sampled_actions = sampled_actions.cpu().numpy()
    
    # Compute variance across samples
    action_var = np.var(sampled_actions, axis=0).mean()
    action_variances.append(action_var)
    
    if idx == test_state_indices[0]:
        print(f"\nExample state {idx}:")
        print(f"  True action: {true_action}")
        print(f"  Mean sampled: {sampled_actions.mean(axis=0)}")
        print(f"  Std sampled:  {sampled_actions.std(axis=0)}")
        print(f"  Action variance: {action_var:.6f}")

avg_action_variance = np.mean(action_variances)
print(f"\nAverage action variance across {n_test_states} states: {avg_action_variance:.6f}")

if avg_action_variance > 0.01:
    print("✅ CVAE captures action diversity (multi-modality)!")
else:
    print("⚠️ Low action variance - may indicate mode collapse")


print("\nTo evaluate in Safety Gym, use bc/eval_cvae_policy.py")