import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import gc

try:
    import gym
    import safety_gym  # noqa: F401
except Exception:
    gym = None


# ============================================================
# 1. Configuration and Load expert demonstrations
# ============================================================

SEED = 42
EXPERT_DATA_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointButton1_s0.pickle"
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test (same as airl_new.py)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

print(f"Loading expert demonstrations from {EXPERT_DATA_PATH}...")
with open(EXPERT_DATA_PATH, "rb") as f:
    data = pickle.load(f)


def _to_array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if isinstance(data, dict) and "states" in data and "actions" in data:
    states = _to_array(data["states"])
    actions = _to_array(data["actions"])
elif isinstance(data, dict) and "trajectories" in data:
    state_chunks = []
    action_chunks = []
    for traj in data["trajectories"]:
        if not isinstance(traj, dict) or "states" not in traj or "actions" not in traj:
            continue
        traj_states = np.stack([_to_array(s) for s in traj["states"]])
        traj_actions = np.stack([_to_array(a) for a in traj["actions"]])
        n = min(len(traj_states), len(traj_actions))
        state_chunks.append(traj_states[:n])
        action_chunks.append(traj_actions[:n])
    if not state_chunks:
        raise ValueError("No usable trajectories found in dataset")
    states = np.concatenate(state_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
else:
    raise ValueError("Unsupported dataset format: expected dict with states/actions or trajectories")

print(f"Loaded {len(states)} transitions.")
print(f"State dim: {states.shape[1]} | Action dim: {actions.shape[1]}")

# --- Train/Test Split at transition level ---
print(f"\n--- Splitting data into train/test ({TRAIN_TEST_SPLIT:.0%} train, {1-TRAIN_TEST_SPLIT:.0%} test) ---")
np.random.seed(SEED)
n_transitions = len(states)
indices = np.random.permutation(n_transitions)
n_train = int(n_transitions * TRAIN_TEST_SPLIT)

train_indices = indices[:n_train]
test_indices = indices[n_train:]

# Instead of materializing large in-memory tensors, save and use memory-mapped .npy files
def _prepare_memmaps(pickle_path: str, states_array, actions_array):
    p = Path(pickle_path)
    states_npy = p.parent / (p.stem + "_states.npy")
    actions_npy = p.parent / (p.stem + "_actions.npy")

    if states_npy.exists() and actions_npy.exists():
        return str(states_npy), str(actions_npy)

    print("Creating memory-mapped .npy files (one-time)...")
    np.save(states_npy, states_array)
    np.save(actions_npy, actions_array)
    gc.collect()
    return str(states_npy), str(actions_npy)

states_npy_path, actions_npy_path = _prepare_memmaps(EXPERT_DATA_PATH, states, actions)

print(f"Training set: {n_train} (state, action) pairs.")
print(f"Test set: {n_transitions - n_train} (state, action) pairs.")


# ============================================================
# 2. Dataset
# ============================================================

class ExpertDataset(Dataset):
    def __init__(self, obs, acts):
        # kept for compatibility but avoid using for large datasets
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.acts = torch.tensor(acts, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]



# Use a lazy memory-mapped dataset to load samples on demand
class MmapDataset(Dataset):
    def __init__(self, states_path, actions_path):
        # load as mmap so arrays are not fully materialized in RAM
        self.states = np.load(states_path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')
        if len(self.states) != len(self.actions):
            raise ValueError('states/actions length mismatch')

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        s = np.asarray(self.states[idx], dtype=np.float32)
        a = np.asarray(self.actions[idx], dtype=np.float32)
        return torch.tensor(s, dtype=torch.float32), torch.tensor(a, dtype=torch.float32)


full_dataset = MmapDataset(states_npy_path, actions_npy_path)
obs_dim = full_dataset.states.shape[1]
act_dim = full_dataset.actions.shape[1]

# Create train/test subsets (these are light Subset wrappers)
from torch.utils.data import Subset
train_subset = Subset(full_dataset, train_indices)
test_subset = Subset(full_dataset, test_indices)

# Split train subset into train/val
val_split = 0.1
val_size = int(len(train_subset) * val_split)
train_size = len(train_subset) - val_size
train_data, val_data = random_split(train_subset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=2)

print(f"BC Training: {train_size} samples, Validation: {val_size} samples")


# ============================================================
# 3. Gaussian Policy
# ============================================================

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, fixed_std=True):
        super().__init__()

        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )

        if fixed_std:
            # Fixed std = 1 → log_std = 0
            self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, obs):
        mean, std = self.forward(obs)
        return mean + torch.randn_like(mean) * std


# ============================================================
# 4. Negative Log-Likelihood loss
# ============================================================

def gaussian_nll(actions, mean, std):
    """
    Computes the exact negative log likelihood for a diagonal Gaussian policy:
    NLL = 0.5 * [ ((a-μ)/σ)^2 + log(2πσ^2) ]
    """
    var = std ** 2

    nll = 0.5 * (((actions - mean) ** 2) / var + torch.log(2 * torch.pi * var))
    return nll.sum(dim=1).mean()   # sum over action dims, mean over batch


# ============================================================
# 5. Train using NLL
# ============================================================

policy = GaussianPolicy(obs_dim, act_dim, fixed_std=True)
policy = policy.to(DEVICE)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

EPOCHS = 10

print("\n=== Starting BC Training (NLL) ===")
best_loss = 1e9

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    policy.train()
    train_loss = 0
    for batch_obs, batch_act in train_loader:
        batch_obs = batch_obs.to(DEVICE, non_blocking=True)
        batch_act = batch_act.to(DEVICE, non_blocking=True)
        mean, std = policy(batch_obs)
        loss = gaussian_nll(batch_act, mean, std)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------- VALIDATION --------
    policy.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_obs, batch_act in val_loader:
            batch_obs = batch_obs.to(DEVICE, non_blocking=True)
            batch_act = batch_act.to(DEVICE, non_blocking=True)
            mean, std = policy(batch_obs)
            loss = gaussian_nll(batch_act, mean, std)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(policy.state_dict(), "bc_gaussian_policy_train_set_button.pth")
        print("   -> best model saved")


print("Training done.")



# ============================================================
# 6. Evaluation in Safety Gym
# ============================================================

def evaluate_policy(policy, env_name="SafetyAntVelocity-v1", n_episodes=10, seed=42):
    if gym is None:
        print("Skipping Safety Gym evaluation because gym/safety_gym is not available in this environment.")
        return None, None

    env = gym.make(env_name)

    rewards = []
    costs = []
    for ep in range(n_episodes):
        if hasattr(env, "seed"):
            env.seed(seed + ep)
        obs = env.reset()
        done = False
        total_reward = 0
        total_cost =0

        reward_ep = []
        cost_ep = []
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, std = policy(obs_t)
            action = mean.squeeze(0).detach().numpy()   # deterministic eval
            action = np.clip(action, env.action_space.low, env.action_space.high)
            # print('std:',std.squeeze(0).detach().numpy())

            obs, reward, done, info = env.step(action)
            cost = float(info.get("cost", 0.0))
            reward_ep.append(reward)
            cost_ep.append(cost)
            total_reward += reward
            total_cost += cost
            # print('truncated:',truncated)
            
        # print("reward per step:",reward_ep)
        # print("cost per step:",cost_ep)
        rewards.append(total_reward)
        # print("printing reward array",rewards)
        costs.append(total_cost)
        # print(costs[:20])
        print(f"Episode {ep+1}: {total_reward:.2f} | Total Cost: {total_cost:.2f}")

    env.close()
    print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
    return np.mean(rewards), np.mean(costs)


# Load best model and evaluate only if the env package is present
policy.load_state_dict(torch.load("bc_gaussian_policy_train_set_button.pth", map_location=DEVICE))
policy.eval()

if gym is not None:
    evaluate_policy(policy, env_name="Safexp-PointGoal1-v0")
else:
    print("Skipping Safety Gym rollout evaluation.")

# ============================================================
# 7. Evaluate BC policy on TRAINING and TEST sets
# ============================================================

def evaluate_subset(policy, subset, subset_name, batch_size=4096):
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(DEVICE.type == "cuda"))
    total_sq = np.zeros(act_dim, dtype=np.float64)
    total_abs = 0.0
    total_count = 0
    first_pairs = []

    with torch.no_grad():
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(DEVICE, non_blocking=True)
            batch_act = batch_act.to(DEVICE, non_blocking=True)
            mean, std = policy(batch_obs)
            diff = mean - batch_act
            total_sq += np.sum(diff.detach().cpu().numpy() ** 2, axis=0)
            total_abs += np.sum(np.abs(diff.detach().cpu().numpy()))
            total_count += diff.numel()

            if len(first_pairs) < 15:
                batch_act_np = batch_act.detach().cpu().numpy()
                batch_mean_np = mean.detach().cpu().numpy()
                take = min(15 - len(first_pairs), len(batch_act_np))
                for i in range(take):
                    first_pairs.append((batch_act_np[i], batch_mean_np[i]))

    mse = float(np.sum(total_sq) / total_count)
    mad = float(total_abs / total_count)
    per_dim_mse = total_sq / max(1, len(subset))

    print(f"✅ [{subset_name}] MSE between learned and expert actions: {mse:.6f}")
    print(f"✅ [{subset_name}] MAD between learned and expert actions: {mad:.6f}")
    print(f"\n[{subset_name}] Per-Dimension MSE between learned and expert actions:")
    for i, mse_value in enumerate(per_dim_mse):
        print(f"Dimension {i}: MSE = {mse_value:.6f}")

    return mse, mad, first_pairs

print("\n" + "=" * 60)
print("EVALUATING TRAINED BC POLICY ON TRAIN AND TEST SETS")
print("=" * 60)

# --- TRAINING SET EVALUATION ---
print("\n=== Evaluating on TRAINING SET ===")
policy.eval()
train_mse, train_mad, _ = evaluate_subset(policy, train_subset, "TRAIN")

# --- TEST SET EVALUATION ---
print("\n=== Evaluating on TEST SET ===")
test_mse, test_mad, test_pairs = evaluate_subset(policy, test_subset, "TEST")

# Show first 15 test comparisons
print("\n[TEST] First 15 action comparisons (Expert vs Learned):")
print("Step | Expert Action               | Learned Action")
print("-" * 65)

for i in range(min(15, len(test_pairs))):
    exp_a, lea_a = test_pairs[i]
    print(f"{i:4d} | {exp_a} | {lea_a}")

# --- SUMMARY ---
print("\n" + "=" * 60)
print("SUMMARY: Train vs Test Performance")
print("=" * 60)
print(f"Training Set ({len(train_subset)} samples):")
print(f"  MSE: {train_mse:.6f}")
print(f"  MAD: {train_mad:.6f}")
print(f"\nTest Set ({len(test_subset)} samples):")
print(f"  MSE: {test_mse:.6f}")
print(f"  MAD: {test_mad:.6f}")
print(f"\nGeneralization Gap:")
print(f"  MSE difference: {abs(test_mse - train_mse):.6f}")
print(f"  MAD difference: {abs(test_mad - train_mad):.6f}")
print("=" * 60)