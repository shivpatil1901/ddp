"""
State-Based Rule Extraction with Action Recommendations

For each bad state region (identified by LIME):
1. Find the state conditions that lead to low Q-values
2. For those states, recommend which actions to take vs avoid
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from lime.lime_tabular import LimeTabularExplainer
import re
from collections import defaultdict
import joblib
import os
from datetime import datetime
import warnings
from pathlib import Path

# Root of the repository (one level up from rules/)
ROOT = Path(__file__).parent.parent

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EXPERT_DEMOS_PATH = ROOT / "SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
Q_NETWORK_PATH = ROOT / "q_function_models/q_network_rollout_only.pt"
SCALER_PATH = ROOT / "q_function_models/q_scaler_rollout_only.pkl"
Y_STATS_PATH = ROOT / "q_function_models/y_stats.npz"
POLICY_PATH = ROOT / "SafeDICE/weights/antidice_PointGoal1_seed0_20260330_205404_iter1000000.pickle"
FITTED_DISTRIBUTIONS_PATH = ROOT / "action_distributions_fitted.pkl"

GAMMA = 0.99
BAD_PERCENTILE = 20  # bottom 20% by average Q-value across actions

NUM_LIME_FEATURES = 3
NUM_LIME_SAMPLES = 3000
NUM_STATES_TO_EXPLAIN = 200

# Use a subset of trajectories for faster iteration/debug runs.
# Set to None to use all trajectories in the dataset.
MAX_TRAJECTORIES = 300

# Concept Model Paths
CONCEPT_MODEL_PATH = ROOT / "q_function_models/concept_value_network.pt"
CONCEPT_SCALER_PATH = ROOT / "q_function_models/concept_scaler.pkl"
TARGET_SCALER_PATH = ROOT / "q_function_models/target_scaler.pkl"

MIN_RULE_SUPPORT = 30
MIN_RULE_PRECISION = 0.6

if torch.cuda.is_available():
    best_cuda_idx = 0
    best_free_bytes = -1
    for i in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(i):
                free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = -1
        if free_bytes > best_free_bytes:
            best_free_bytes = free_bytes
            best_cuda_idx = i
    DEVICE = torch.device(f"cuda:{best_cuda_idx}")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

print(f"Using compute device: {DEVICE}")

# --------------------------------------------------
# CONCEPT FEATURE CONFIGURATION  (optional layer)
# --------------------------------------------------
# When True, LIME operates in concept space (12 interpretable features)
# instead of the raw 15-feature PCA-selected space.
# V(s) computation is unaffected — always uses raw 27-dim states.
USE_CONCEPT_FEATURES = False


# Set to True to force recalculation of V(s), False to use cached values
RECALCULATE_VALUE_FUNCTION = True

# Action sampling method: "random", "policy", or "distribution"
# - "random": Sample from N(0, 1) for each action dimension
# - "policy": Sample from expert policy π(a|s) (RECOMMENDED)
# - "distribution": Sample from fitted marginal distributions p(a)
ACTION_SAMPLING_METHOD = "policy"  # Options: "random", "policy", "distribution"

# Number of actions to sample per state for V(s) calculation
# Lower default keeps runtime manageable on large datasets; increase for final high-fidelity runs.
NUM_ACTION_SAMPLES_MAIN = 30  # For main V(s) calculation
NUM_ACTION_SAMPLES_LIME = 20   # For LIME perturbations (faster)

VALUE_COMPUTATION_STATE_BATCH = 256

# --------------------------------------------------
# FEATURE NAME MAPPINGS (PointGoal)
# --------------------------------------------------
# Specific PointGoal observation space:
#  - accelerometer: 3 values
#  - velocimeter: 3 values
#  - gyro: 3 values
#  - magnetometer: 3 values
# Total raw observation size: 12
#
# Specific PointGoal action space:
#  - action[0]: forward/backward force (x)
#  - action[1]: rotational velocity around z-axis (z)
# Total action size: 2

STATE_FEATURE_NAMES = {
    0: "accelerometer_x",
    1: "accelerometer_y",
    2: "accelerometer_z",
    3: "velocimeter_x",
    4: "velocimeter_y",
    5: "velocimeter_z",
    6: "gyro_x",
    7: "gyro_y",
    8: "gyro_z",
    9: "magnetometer_x",
    10: "magnetometer_y",
    11: "magnetometer_z",
}

ACTION_FEATURE_NAMES = {
    0: "force_x",
    1: "velocity_z",
}

def get_feature_name(feature_idx, feature_type="state"):
    """Get human-readable name for a feature index."""
    if feature_type == "state":
        return STATE_FEATURE_NAMES.get(feature_idx, f"State_{feature_idx}")
    else:
        return ACTION_FEATURE_NAMES.get(feature_idx, f"Action_{feature_idx}")

# --------------------------------------------------
# 1. LOAD TRAJECTORIES (STATES ONLY)
# --------------------------------------------------

print("Loading trajectories...")
data = np.load(EXPERT_DEMOS_PATH, allow_pickle=True)

if isinstance(data, dict) and "trajectories" in data:
    print("Data file memory-mapped. Accessing 'trajectories' (this may take a minute if pickled tensors are present)...")
    trajectories = data["trajectories"]
elif isinstance(data, dict) and {"states", "next_states", "dones"}.issubset(data.keys()):
    print("Data file contains flat transitions; reconstructing trajectories from 'dones'...")
    states_array = np.asarray(data["states"])
    next_states_array = np.asarray(data["next_states"])
    dones_array = np.asarray(data["dones"]).reshape(-1)

    trajectories = []
    start_idx = 0
    for end_idx in np.where(dones_array > 0.5)[0]:
        traj_states = states_array[start_idx:end_idx + 1]
        traj_next_states = next_states_array[start_idx:end_idx + 1]
        if len(traj_states) == 0:
            start_idx = end_idx + 1
            continue

        # Preserve the existing downstream contract: traj["states"] includes
        # the terminal state so traj["states"][:-1] returns the non-terminal states.
        trajectories.append({
            "states": np.concatenate([traj_states, traj_next_states[-1:]], axis=0)
        })
        start_idx = end_idx + 1

    print(f"Data file contained {len(trajectories)} reconstructed trajectories.")
else:
    raise KeyError("Dataset does not contain 'trajectories' or transition keys 'states'/'next_states'/'dones'.")

total_trajectories_available = len(trajectories)
if MAX_TRAJECTORIES is not None and MAX_TRAJECTORIES < total_trajectories_available:
    trajectories = trajectories[:MAX_TRAJECTORIES]
    print(f"✅ Loaded {len(trajectories)} trajectories (subset of {total_trajectories_available}).")
else:
    print(f"✅ Loaded {len(trajectories)} trajectories.")

# Flatten states only
states = []
for traj in trajectories:
    for s in traj["states"][:-1]:
        states.append(s)

X_states = np.array(states)
print(f"Total states: {len(X_states)}")

state_dim = X_states.shape[1]

# Load Q-network scaler first so PointGoal state/action dims come from the saved training artifacts.
print(f"\nLoading scaler from {SCALER_PATH}...")
scaler = joblib.load(SCALER_PATH)
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Scaler scale shape: {scaler.scale_.shape}")

total_input_dim = int(scaler.mean_.shape[0])
action_dim = total_input_dim - state_dim
if action_dim <= 0:
    raise ValueError(
        f"Invalid inferred action_dim={action_dim}. state_dim={state_dim}, scaler_dim={total_input_dim}. "
        "Check that the expert-demo states and Q-network scaler belong to the same environment."
    )

print(f"Inferred action dimension: {action_dim}")

if state_dim >= 12:
    print("\nPointGoal observation-space mapping:")
    for idx in range(12):
        print(f"  obs[{idx}]: {get_feature_name(idx, 'state')}")
else:
    print(f"\nWarning: state_dim={state_dim} is smaller than the 12-dim PointGoal observation template.")

if action_dim >= 2:
    print("PointGoal action-space mapping:")
    for idx in range(2):
        print(f"  action[{idx}]: {get_feature_name(idx, 'action')}")

# --------------------------------------------------
# 1.5. APPLY PCA FEATURE REDUCTION (STATES ONLY)
# --------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\nApplying PCA feature reduction to states...")

scaler_pca = StandardScaler()
X_states_scaled = scaler_pca.fit_transform(X_states)

pca = PCA()
pca.fit(X_states_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Components for 95% variance: {n_components_95}")

# Get top 15 state features
n_top_features = 15
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
feature_importance = np.sum(np.abs(loadings[:, :n_components_95]), axis=1)
top_15_indices = np.argsort(feature_importance)[::-1][:n_top_features]
top_15_sorted = sorted(top_15_indices)

print(f"Top 15 state feature indices: {top_15_sorted}")
print("\nSelected features:")
for idx in top_15_sorted:
    print(f"  [{idx}] {get_feature_name(idx, 'state')}")

# Reduce states to top 15 features
X_states_reduced = X_states[:, top_15_sorted]
state_dim_reduced = X_states_reduced.shape[1]
print(f"\nReduced state dimension: {state_dim_reduced}")

original_state_feature_indices = top_15_sorted

# --------------------------------------------------
# OPTIONAL: CONCEPT FEATURE TRANSFORMATION
# --------------------------------------------------
if USE_CONCEPT_FEATURES:
    import sys as _sys
    _sys.path.insert(0, str(ROOT))  # so 'concepts' package is found
    from concepts.concept_features import ConceptTransformer

    concept_transformer = ConceptTransformer()
    # Transform ALL raw states (27-dim) into concept space (12-dim)
    X_lime = concept_transformer.transform(X_states)   # (N, 12)
    lime_feature_names = concept_transformer.feature_names()
    print(f"\n✅ Concept features ON: LIME will use {len(lime_feature_names)} concepts")
    for i, name in enumerate(lime_feature_names):
        print(f"  [{i}] {name}: {concept_transformer.describe(name)}")
else:
    # Default: use PCA-selected raw features
    X_lime = X_states_reduced                                          # (N, 15)
    lime_feature_names = [f"State_{idx}" for idx in original_state_feature_indices]
    print(f"\n  Concept features OFF: LIME will use {len(lime_feature_names)} raw features")

# --------------------------------------------------
# 2. LOAD Q-NETWORK
# --------------------------------------------------

print("\nLoading Q-network...")
print(f"Loading Q-network from {Q_NETWORK_PATH}...")

# Define Q-network architecture (must match training)
class RobustQNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)

# Concept Value Network (Matches train_concept_value_function.py)
class ConceptValueNet(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[512, 256, 128], dropout=0.1, use_batchnorm=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# Load target normalization stats for Q-network outputs
print(f"\nLoading Q-target stats from {Y_STATS_PATH}...")
y_stats = np.load(Y_STATS_PATH)
q_target_mean = float(y_stats["mean"])
q_target_std = float(y_stats["std"])
print(f"Q-target mean: {q_target_mean:.6f} | std: {q_target_std:.6f}")

# Load model
q_network_full = RobustQNetwork(input_dim=total_input_dim)
ckpt = torch.load(Q_NETWORK_PATH, map_location=DEVICE)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    ckpt = ckpt["model_state_dict"]
q_network_full.load_state_dict(ckpt, strict=True)
q_network_full.eval()
q_network_full.to(DEVICE)

print("✅ Q-network loaded successfully")

# Extract normalization parameters
state_mean_full = scaler.mean_[:state_dim]
state_std_full = scaler.scale_[:state_dim]
state_mean_selected = state_mean_full[original_state_feature_indices]
state_std_selected = state_std_full[original_state_feature_indices]

action_mean = scaler.mean_[state_dim:]
action_std = scaler.scale_[state_dim:]

print(f"Using normalization for {state_dim_reduced} state features + {action_dim} action features")

# Load Concept Model if enabled
concept_model = None
concept_x_scaler = None
concept_y_scaler = None

if USE_CONCEPT_FEATURES:
    print(f"\n🧠 Loading Concept Value Network from {CONCEPT_MODEL_PATH}...")
    # Matches hidden_dims in config_training.yaml
    concept_model = ConceptValueNet(input_dim=12, hidden_dims=[512, 256, 128])
    concept_model.load_state_dict(torch.load(CONCEPT_MODEL_PATH, map_location=DEVICE))
    concept_model.eval()
    concept_model.to(DEVICE)
    
    concept_x_scaler = joblib.load(CONCEPT_SCALER_PATH)
    concept_y_scaler = joblib.load(TARGET_SCALER_PATH)
    print("✅ Concept Value Network and scalers loaded successfully")

# --------------------------------------------------
# 3. LOAD POLICY FOR ACTION SAMPLING
# --------------------------------------------------

print("\nLoading policy for action sampling...")
policy = None
policy_obs_mean = None
policy_obs_std = None
policy_requires_obs_normalization = False
policy_backend = None
policy_state_dict = None

policy_checkpoint = None
policy_load_error = None

# Try loading as pickle first (for TensorFlow SavedModel-based checkpoints)
try:
    import pickle as pickle_module
    with open(str(POLICY_PATH), 'rb') as f:
        policy_checkpoint = pickle_module.load(f)
    print("✅ Loaded policy pickle successfully")
except Exception as e_pickle:
    # Try torch.load as fallback
    try:
        policy_checkpoint = torch.load(str(POLICY_PATH), map_location=DEVICE, weights_only=False)
        print("✅ Loaded policy with torch.load")
    except Exception as e_torch:
        # Try joblib as last resort
        try:
            policy_checkpoint = joblib.load(str(POLICY_PATH))
            print("✅ Loaded policy with joblib")
        except Exception as e_joblib:
            policy_load_error = (e_pickle, e_torch, e_joblib)

class SafeDICETanhPolicyNumpy:
    """Numpy forward pass for SafeDICE TanhActor from serialized TensorFlow weights."""

    def __init__(self, actor_params, mean_range=(-7.0, 7.0), logstd_range=(-5.0, 2.0)):
        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range

        dense_kernels = []
        dense_biases = []
        mean_w = mean_b = None
        logstd_w = logstd_b = None

        for name, value in actor_params:
            arr = np.array(value)
            if "mlp/dense/kernel" in name:
                dense_kernels.append(arr)
            elif "mlp/dense/bias" in name:
                dense_biases.append(arr)
            elif "policy_mean/dense/kernel" in name:
                mean_w = arr
            elif "policy_mean/dense/bias" in name:
                mean_b = arr
            elif "policy_logstd/dense/kernel" in name:
                logstd_w = arr
            elif "policy_logstd/dense/bias" in name:
                logstd_b = arr

        if len(dense_kernels) < 2 or len(dense_biases) < 2:
            raise ValueError("SafeDICE actor params missing MLP layers")
        if mean_w is None or mean_b is None or logstd_w is None or logstd_b is None:
            raise ValueError("SafeDICE actor params missing policy mean/logstd heads")

        # SafeDICE TanhActor has 2 hidden ReLU layers.
        self.w1, self.b1 = dense_kernels[0], dense_biases[0]
        self.w2, self.b2 = dense_kernels[1], dense_biases[1]
        self.wm, self.bm = mean_w, mean_b
        self.ws, self.bs = logstd_w, logstd_b

    def _forward_mean_logstd(self, obs_1d):
        x = np.asarray(obs_1d, dtype=np.float32)
        h1 = np.maximum(x @ self.w1 + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0.0)
        mean = np.clip(h2 @ self.wm + self.bm, self.mean_min, self.mean_max)
        logstd = np.clip(h2 @ self.ws + self.bs, self.logstd_min, self.logstd_max)
        return mean, logstd

    def sample_action(self, obs_1d, deterministic=False):
        mean, logstd = self._forward_mean_logstd(obs_1d)
        if deterministic:
            return np.tanh(mean).astype(np.float32)
        std = np.exp(logstd)
        pretanh = mean + np.random.randn(*mean.shape).astype(np.float32) * std
        return np.tanh(pretanh).astype(np.float32)

    def sample_actions(self, obs_1d, num_samples, deterministic=False):
        mean, logstd = self._forward_mean_logstd(obs_1d)
        if deterministic:
            return np.tile(np.tanh(mean).astype(np.float32), (num_samples, 1))
        std = np.exp(logstd)
        noise = np.random.randn(num_samples, mean.shape[0]).astype(np.float32)
        pretanh = mean.reshape(1, -1) + noise * std.reshape(1, -1)
        return np.tanh(pretanh).astype(np.float32)


if policy_checkpoint is None:
    if ACTION_SAMPLING_METHOD == "policy":
        print("[WARN] Could not load policy checkpoint at", POLICY_PATH)
        if policy_load_error is not None:
            print("[WARN] pickle error:", repr(policy_load_error[0])[:100])
            print("[WARN] torch.load error:", repr(policy_load_error[1])[:100])
            print("[WARN] joblib error:", repr(policy_load_error[2])[:100])
        print("[WARN] Falling back ACTION_SAMPLING_METHOD from 'policy' to 'random'.")
        ACTION_SAMPLING_METHOD = "random"
else:
    if not isinstance(policy_checkpoint, dict):
        raise TypeError(f"Unsupported policy checkpoint type: {type(policy_checkpoint)}")

    checkpoint_keys = list(policy_checkpoint.keys())
    print("✅ Policy checkpoint loaded. Top-level keys:", checkpoint_keys[:10])

    # Handle TensorFlow-style checkpoint with 'training_state' and 'training_info'
    if "training_state" in policy_checkpoint:
        print("[INFO] Detected TensorFlow training checkpoint format")
        training_state = policy_checkpoint.get("training_state", {})
        actor_params = training_state.get("actor_params")

        if isinstance(actor_params, list) and len(actor_params) > 0:
            try:
                policy = SafeDICETanhPolicyNumpy(actor_params)
                policy_backend = "safedice_tf"
                policy_requires_obs_normalization = False
                print(f"✅ Reconstructed SafeDICE TF actor from {len(actor_params)} tensors")
                print("[INFO] Policy sampling will use raw observations (no checkpoint normalizer required)")
            except Exception as e_policy:
                if ACTION_SAMPLING_METHOD == "policy":
                    print("[WARN] Failed to reconstruct SafeDICE actor:", repr(e_policy))
                    print("[WARN] Falling back ACTION_SAMPLING_METHOD from 'policy' to 'random'.")
                    ACTION_SAMPLING_METHOD = "random"
        else:
            if ACTION_SAMPLING_METHOD == "policy":
                print("[WARN] TensorFlow checkpoint missing usable 'actor_params'.")
                print("[WARN] Falling back ACTION_SAMPLING_METHOD from 'policy' to 'random'.")
                ACTION_SAMPLING_METHOD = "random"
    
    # Handle standard PyTorch checkpoint format
    elif "pi" in policy_checkpoint and "obs_normalizer" in policy_checkpoint:
        policy_state_dict = policy_checkpoint['pi']
        obs_normalizer = policy_checkpoint['obs_normalizer']
        policy_obs_mean = obs_normalizer['_mean'].numpy()
        policy_obs_std = obs_normalizer['_std'].numpy()
        policy_requires_obs_normalization = True
        policy_backend = "pytorch"
        print("✅ Loaded standard PyTorch policy checkpoint with 'pi' and 'obs_normalizer'")
    else:
        if ACTION_SAMPLING_METHOD == "policy":
            print("[WARN] Policy checkpoint format not recognized:")
            print("[WARN]   Missing expected keys: 'training_state' OR ('pi' + 'obs_normalizer')")
            print("[WARN]   Available top-level keys:", checkpoint_keys[:20])
            print("[WARN] Falling back ACTION_SAMPLING_METHOD from 'policy' to 'random'.")
            ACTION_SAMPLING_METHOD = "random"

# Define policy network
class GaussianPolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mean_net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim)
        )
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean
        else:
            return mean + torch.randn_like(mean) * std

if policy_backend == "pytorch" and ACTION_SAMPLING_METHOD == "policy":
    policy = GaussianPolicy(state_dim, action_dim)
    policy.load_state_dict(policy_state_dict, strict=False)
    policy.to(DEVICE)
    policy.eval()
    print("✅ Policy loaded successfully (PyTorch backend)")
elif policy_backend == "safedice_tf" and ACTION_SAMPLING_METHOD == "policy":
    print("✅ Policy loaded successfully (SafeDICE TensorFlow backend)")
else:
    print("Policy model not used (sampling method: %s)" % ACTION_SAMPLING_METHOD)

# --------------------------------------------------
# LOAD FITTED DISTRIBUTIONS (if using distribution method)
# --------------------------------------------------

FITTED_DISTRIBUTIONS_GLOBAL = None
if ACTION_SAMPLING_METHOD == "distribution":
    print(f"\n📊 Loading fitted action distributions (once)...")
    FITTED_DISTRIBUTIONS_GLOBAL = joblib.load(FITTED_DISTRIBUTIONS_PATH)
    print(f"✅ Loaded distributions for {len(FITTED_DISTRIBUTIONS_GLOBAL)} action dimensions")
# 4. COMPUTE AVERAGE Q-VALUE FOR EACH STATE
# --------------------------------------------------

print("\nComputing average Q-values across actions for each state...")
print(f"Using FULL {state_dim}-dim states and {ACTION_SAMPLING_METHOD.upper()}-based action sampling")

def compute_state_value(states_full, num_action_samples=100, method="policy"):
    """
    Compute average Q-value for states by sampling actions.
    If USE_CONCEPT_FEATURES is True, uses the pre-trained Concept Value Network.
    """
    if USE_CONCEPT_FEATURES and concept_model is not None:
        print("  Using Concept Value Network for V(s) calculation...")
        # 1. Transform raw states to concepts
        X_concepts = concept_transformer.transform(states_full)
        
        # 2. Scale concepts
        X_concepts_scaled = concept_x_scaler.transform(X_concepts).astype(np.float32)
        
        # 3. Predict V(s) in scaled space
        with torch.no_grad():
            x_t = torch.tensor(X_concepts_scaled).to(DEVICE)
            v_scaled = concept_model(x_t).cpu().numpy()
            
        # 4. Inverse scale V(s)
        v_final = concept_y_scaler.inverse_transform(v_scaled).flatten()
        return v_final
    
    # Fallback to action sampling (original logic)
    N = len(states_full)
    print(f"Computing V(s) for {N:,} states with {num_action_samples} sampled actions/state...")
    all_q_values = np.empty(N, dtype=np.float32)
    fitted_dists = FITTED_DISTRIBUTIONS_GLOBAL
    state_batch = VALUE_COMPUTATION_STATE_BATCH
    start = 0
    while start < N:
        end = min(start + state_batch, N)
        states_chunk = np.asarray(states_full[start:end], dtype=np.float32)
        chunk_size = len(states_chunk)

        if method == "random":
            actions_sampled = np.random.randn(chunk_size, num_action_samples, action_dim).astype(np.float32)
        elif method == "policy":
            if policy is None:
                raise RuntimeError("Policy sampling requested but policy is not loaded")

            if policy_requires_obs_normalization:
                if policy_obs_mean is None or policy_obs_std is None:
                    raise RuntimeError("Policy requires obs normalization, but normalizer was not loaded")
                policy_inputs = (states_chunk - policy_obs_mean) / (policy_obs_std + 1e-8)
            else:
                policy_inputs = states_chunk

            if policy_backend == "pytorch":
                with torch.inference_mode():
                    state_tensor = torch.as_tensor(policy_inputs, dtype=torch.float32, device=DEVICE)
                    state_batch = state_tensor.repeat_interleave(num_action_samples, dim=0)
                    mean, std = policy.forward(state_batch)
                    sampled = mean + torch.randn_like(mean) * std
                    actions_sampled = sampled.reshape(chunk_size, num_action_samples, action_dim).cpu().numpy()
            else:
                actions_sampled = np.empty((chunk_size, num_action_samples, action_dim), dtype=np.float32)
                for j, state_full in enumerate(policy_inputs):
                    actions_sampled[j] = policy.sample_actions(state_full, num_action_samples, deterministic=False)
        elif method == "distribution":
            actions_sampled = np.empty((chunk_size, num_action_samples, action_dim), dtype=np.float32)
            for j in range(chunk_size):
                for k in range(action_dim):
                    gmm = fitted_dists[k]["gmm"]
                    samples, _ = gmm.sample(num_action_samples)
                    actions_sampled[j, :, k] = samples.flatten()
        else:
            raise ValueError(f"Unknown action sampling method: {method}")

        states_norm_q = (states_chunk - state_mean_full) / state_std_full
        states_repeated_norm = np.repeat(states_norm_q, num_action_samples, axis=0)
        actions_norm_q = (actions_sampled.reshape(-1, action_dim) - action_mean) / action_std
        inputs = np.concatenate([states_repeated_norm, actions_norm_q], axis=1).astype(np.float32, copy=False)

        try:
            with torch.inference_mode():
                x = torch.as_tensor(inputs, dtype=torch.float32, device=DEVICE)
                q_vals = q_network_full(x).cpu().numpy().reshape(-1)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if DEVICE.type == "cuda" and "out of memory" in err_msg:
                if state_batch <= 16:
                    raise RuntimeError(
                        "CUDA OOM even at minimal state batch size (16). "
                        "Reduce NUM_ACTION_SAMPLES_MAIN or switch to CPU."
                    ) from e
                new_state_batch = max(16, state_batch // 2)
                print(
                    f"  [WARN] CUDA OOM at state_batch={state_batch}. "
                    f"Retrying with state_batch={new_state_batch}..."
                )
                state_batch = new_state_batch
                torch.cuda.empty_cache()
                continue
            raise

        q_vals = q_vals * q_target_std + q_target_mean
        all_q_values[start:end] = q_vals.reshape(chunk_size, num_action_samples).mean(axis=1)
        start = end

        if start % 1000 == 0 or start == N:
            print(f"  Processed {start:,}/{N:,} states...")

    return all_q_values


# --------------------------------------------------
# COMPUTE OR LOAD STATE VALUES V(s)
# --------------------------------------------------

# Cache file based on sampling method and trajectory subset size
traj_tag = f"traj{len(trajectories)}"
V_CACHE_FILE = ROOT / f"state_values_{ACTION_SAMPLING_METHOD}_{traj_tag}.npz"

print(f"\n{'='*70}")
print(f"VALUE FUNCTION COMPUTATION")
print(f"{'='*70}")
print(f"Action sampling method: {ACTION_SAMPLING_METHOD}")
print(f"Number of action samples: {NUM_ACTION_SAMPLES_MAIN}")
print(f"Recalculate: {RECALCULATE_VALUE_FUNCTION}")

# Check if cached V values exist and if we should use them
if os.path.exists(V_CACHE_FILE) and not RECALCULATE_VALUE_FUNCTION:
    print(f"\n✅ Loading cached state values from {V_CACHE_FILE}...")
    cached_data = np.load(V_CACHE_FILE)
    V = cached_data['V']
    print(f"Loaded {len(V)} state values")
    print(f"State value stats: mean={V.mean():.2f}, std={V.std():.2f}, min={V.min():.2f}, max={V.max():.2f}")
else:
    if RECALCULATE_VALUE_FUNCTION:
        print(f"\n🔄 Recalculating state values (RECALCULATE_VALUE_FUNCTION=True)...")
    else:
        print(f"\n⚠️  No cached file found. Computing state values...")
    
    print(f"\nUsing '{ACTION_SAMPLING_METHOD}' sampling method...")
    if ACTION_SAMPLING_METHOD == "random":
        print("⚠️  WARNING: Random sampling may give inaccurate V(s) estimates!")
    elif ACTION_SAMPLING_METHOD == "distribution":
        print("⚠️  WARNING: Distribution sampling ignores state-dependency!")
        print("   This may give biased V(s) estimates.")
    elif ACTION_SAMPLING_METHOD == "policy":
        print("✅ Using policy-based sampling (recommended)")
    
    V = compute_state_value(
        X_states, 
        num_action_samples=NUM_ACTION_SAMPLES_MAIN,
        method=ACTION_SAMPLING_METHOD
    )
    
    print(f"\nState value stats: mean={V.mean():.2f}, std={V.std():.2f}, min={V.min():.2f}, max={V.max():.2f}")
    
    # Save for future use
    np.savez(V_CACHE_FILE, V=V, method=ACTION_SAMPLING_METHOD, num_samples=NUM_ACTION_SAMPLES_MAIN)
    print(f"✅ Saved state values to {V_CACHE_FILE}")

print(f"{'='*70}\n")
# 4. IDENTIFY BAD STATES
# --------------------------------------------------

threshold = np.percentile(V, BAD_PERCENTILE)
labels = (V <= threshold).astype(int)

num_bad = labels.sum()
num_good = len(labels) - num_bad

# Save state values and labels for reuse
print("Saving state values and labels...")
(ROOT / "q_function_models").mkdir(exist_ok=True)
np.savez(str(ROOT / "q_function_models/state_values_and_labels.npz"),
         state_values=V,
         labels=labels,
         top_15_features=original_state_feature_indices,
         bad_percentile=BAD_PERCENTILE)
print("✅ Saved to q_function_models/state_values_and_labels.npz")

print("\n" + "="*70)
print("STATE CLASSIFICATION STATISTICS")
print("="*70)
print(f"Total states: {len(X_states_reduced):,}")
print(f"Bad states (bottom {BAD_PERCENTILE}%): {num_bad:,} ({num_bad/len(labels)*100:.1f}%)")
print(f"Good states: {num_good:,} ({num_good/len(labels)*100:.1f}%)")
print(f"Bad-value threshold: {threshold:.3f}")
print(f"V(s) range: [{V.min():.2f}, {V.max():.2f}]")
print("="*70)

# --------------------------------------------------
# 5. LIME EXPLAINER FOR STATES
# --------------------------------------------------

explainer = LimeTabularExplainer(
    training_data=X_lime,
    feature_names=lime_feature_names,
    mode="regression",
    discretize_continuous=True,
    verbose=False,
)

def value_predict_fn(query_lime_states):
    """
    LIME prediction wrapper.
    """
    # OPTION A: Use distilled concept model (Direct Mapping)
    if USE_CONCEPT_FEATURES and concept_model is not None:
        with torch.no_grad():
            x_norm = concept_x_scaler.transform(query_lime_states).astype(np.float32)
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).to(DEVICE)
            y_norm = concept_model(x_tensor).cpu().numpy().reshape(-1, 1)
            # Un-scale to original Q-value magnitude
            return concept_y_scaler.inverse_transform(y_norm).flatten()

    # OPTION B: Reconstruction-based prediction (Historical Method)
    N = len(query_lime_states)
    query_states_full = np.tile(state_mean_full, (N, 1))  # base: dataset mean

    if USE_CONCEPT_FEATURES:
        # 1:1 invertible concept → raw feature mappings:
        query_states_full[:, 3]  = query_lime_states[:, 0]   # Forward_Speed
        query_states_full[:, 20] = query_lime_states[:, 3]   # Body_Height
        query_states_full[:, 7]  = query_lime_states[:, 4]   # Pitch_Rate
        query_states_full[:, 6]  = query_lime_states[:, 5]   # Roll_Rate
        query_states_full[:, 8]  = query_lime_states[:, 6]   # Yaw_Rate
    else:
        # Raw mode: fill in the 15 selected features
        for j, orig_idx in enumerate(original_state_feature_indices):
            query_states_full[:, orig_idx] = query_lime_states[:, j]

    return compute_state_value(
        query_states_full,
        num_action_samples=NUM_ACTION_SAMPLES_LIME,
        method=ACTION_SAMPLING_METHOD
    )

# --------------------------------------------------
# 6. EXTRACT STATE RULES
# --------------------------------------------------

print("\n" + "="*70)
print("RULE EXTRACTION PARAMETERS")
print("="*70)
print(f"NUM_STATES_TO_EXPLAIN: {NUM_STATES_TO_EXPLAIN}")
print(f"  (Explaining {NUM_STATES_TO_EXPLAIN} randomly sampled bad states)")
print(f"NUM_LIME_FEATURES: {NUM_LIME_FEATURES}")
print(f"  (Each rule will have up to {NUM_LIME_FEATURES} conditions)")
print(f"NUM_LIME_SAMPLES: {NUM_LIME_SAMPLES}")
print(f"  (LIME generates {NUM_LIME_SAMPLES} perturbed samples per state)")
print("="*70)

print("\nExtracting state-based rules...")

def parse_lime(exp):
    """
    Parse LIME explanation into rule conditions.
    Handles both concept-feature names (e.g. 'Pitch_Rate') and raw 'State_N' names.
    Each rule condition: (feature_idx_in_lime, op, val, feature_display_name)
    """
    rules = []
    for text, weight in exp.as_list():
        if abs(weight) < 1e-3:
            continue

        if USE_CONCEPT_FEATURES:
            # Concept mode: LIME names are like 'Pitch_Rate', 'Body_Height <= 0.32', etc.
            # Match any named feature from lime_feature_names
            matched_idx = None
            matched_name = None
            for fi, fname in enumerate(lime_feature_names):
                if fname in text:
                    matched_idx = fi
                    matched_name = fname
                    break
            if matched_idx is None:
                continue
            feature_label = matched_name
            lime_idx = matched_idx
        else:
            # Raw mode: names are 'State_N'
            m = re.search(r"State_(\d+)", text)
            if not m:
                continue
            original_idx = int(m.group(1))
            if original_idx not in original_state_feature_indices:
                continue
            lime_idx = original_state_feature_indices.index(original_idx)
            feature_label = original_idx  # keep original index for downstream use

        if "<=" in text:
            val = float(text.split("<=")[-1])
            rules.append((lime_idx, "<=", val, feature_label))
        elif ">" in text:
            val = float(text.split(">")[-1])
            rules.append((lime_idx, ">", val, feature_label))

    return tuple(sorted(rules))


def rule_mask(X, rule):
    mask = np.ones(len(X), dtype=bool)
    for lime_idx, op, v, _ in rule:
        if op == "<=":
            mask &= (X[:, lime_idx] <= v)
        else:
            mask &= (X[:, lime_idx] > v)
    return mask

# Sample bad states
bad_idxs = np.where(labels == 1)[0]
np.random.shuffle(bad_idxs)
bad_idxs = bad_idxs[:NUM_STATES_TO_EXPLAIN]

rule_db = defaultdict(lambda: {"bad": 0, "good": 0, "states": []})

for idx in bad_idxs:
    exp = explainer.explain_instance(
        X_lime[idx],
        value_predict_fn,
        num_features=NUM_LIME_FEATURES,
        num_samples=NUM_LIME_SAMPLES,
    )

    rule = parse_lime(exp)
    if not rule:
        continue

    mask = rule_mask(X_lime, rule)
    rule_db[rule]["bad"] += labels[mask].sum()
    rule_db[rule]["good"] += (1 - labels[mask]).sum()

    matching_states = X_lime[mask]
    if len(matching_states) > 0:
        rule_db[rule]["states"].extend(matching_states[:5].tolist())

# --------------------------------------------------
# 7. FILTER RULES AND ANALYZE ACTIONS
# --------------------------------------------------

print("\n" + "="*70)
print("RULE FILTERING STATISTICS")
print("="*70)
print(f"Total unique rules extracted: {len(rule_db)}")

final_rules = []
filtered_by_support = 0
filtered_by_precision = 0

for rule, cnt in rule_db.items():
    support = cnt["bad"] + cnt["good"]
    if support < MIN_RULE_SUPPORT:
        filtered_by_support += 1
        continue
    
    precision = cnt["bad"] / support
    if precision >= MIN_RULE_PRECISION:
        recall = cnt["bad"] / num_bad  # num_bad defined earlier as labels.sum()
        final_rules.append({
            "rule": rule,
            "precision": precision,
            "support": support,
            "coverage": support / len(X_lime),
            "recall": recall,
            "example_states": cnt["states"][:3],  # Keep 3 examples
        })
    else:
        filtered_by_precision += 1

final_rules.sort(key=lambda x: (-x["precision"], -x["support"]))

print(f"Rules filtered by support (< {MIN_RULE_SUPPORT}): {filtered_by_support}")
print(f"Rules filtered by precision (< {MIN_RULE_PRECISION}): {filtered_by_precision}")
print(f"Final rules passing all filters: {len(final_rules)}")
print("="*70)

# --------------------------------------------------
# 8. FOR EACH RULE, RECOMMEND ACTIONS
# --------------------------------------------------

def compute_action_recommendations(rule, num_actions=100):
    """Compute best and worst action ranges (top/bottom 5%) for a given rule."""
    if not rule["example_states"]:
        return None

    # Take first example state
    example_lime = np.array(rule["example_states"][0])  # (12,) or (15,) depending on mode

    # Reconstruct a full 27-dim state for Q-network input
    example_state_full = state_mean_full.copy()  # start with dataset mean

    if USE_CONCEPT_FEATURES:
        # Partially invert concept → raw (same mapping as value_predict_fn)
        example_state_full[3]  = example_lime[0]   # Forward_Speed  → vel_x
        example_state_full[20] = example_lime[3]   # Body_Height    → torso_z
        example_state_full[7]  = example_lime[4]   # Pitch_Rate     → gyro_y
        example_state_full[6]  = example_lime[5]   # Roll_Rate      → gyro_x
        example_state_full[8]  = example_lime[6]   # Yaw_Rate       → gyro_z
        # Composite concepts are not invertible; dims stay at dataset mean
    else:
        # Raw mode: fill in the 15 selected features
        for j, orig_idx in enumerate(original_state_feature_indices):
            example_state_full[orig_idx] = example_lime[j]

    # Try many different actions
    actions_sampled = np.random.randn(num_actions, action_dim)

    # Normalize full state and actions for Q-network
    state_norm_full = (example_state_full - state_mean_full) / state_std_full   # (27,)
    actions_norm = (actions_sampled - action_mean) / action_std                 # (100, 8)

    # Build Q-network inputs: (100, 35) = concat[state(27), action(8)]
    states_repeated_norm = np.tile(state_norm_full, (num_actions, 1))           # (100, 27)
    inputs = np.concatenate([states_repeated_norm, actions_norm], axis=1)        # (100, 35)

    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32).to(DEVICE)
        q_vals = q_network_full(x).cpu().numpy().flatten()
    
    # Find top 5% and bottom 5%
    top_5_percent_count = max(1, int(num_actions * 0.05))
    bottom_5_percent_count = max(1, int(num_actions * 0.05))
    
    # Get indices of top and bottom 5%
    top_indices = np.argsort(q_vals)[-top_5_percent_count:]
    bottom_indices = np.argsort(q_vals)[:bottom_5_percent_count]
    
    # Compute statistics for top 5% (best actions)
    top_actions = actions_sampled[top_indices]
    top_q_vals = q_vals[top_indices]
    
    best_action_ranges = []
    for i in range(action_dim):
        action_values = top_actions[:, i]
        best_action_ranges.append({
            "action_index": i,
            "action_name": get_feature_name(i, "action"),
            "min": float(action_values.min()),
            "max": float(action_values.max()),
            "mean": float(action_values.mean()),
            "std": float(action_values.std())
        })
    
    # Compute statistics for bottom 5% (worst actions)
    bottom_actions = actions_sampled[bottom_indices]
    bottom_q_vals = q_vals[bottom_indices]
    
    worst_action_ranges = []
    for i in range(action_dim):
        action_values = bottom_actions[:, i]
        worst_action_ranges.append({
            "action_index": i,
            "action_name": get_feature_name(i, "action"),
            "min": float(action_values.min()),
            "max": float(action_values.max()),
            "mean": float(action_values.mean()),
            "std": float(action_values.std())
        })
    
    return {
        "best_actions_top5pct": {
            "q_value_range": {
                "min": float(top_q_vals.min()),
                "max": float(top_q_vals.max()),
                "mean": float(top_q_vals.mean()),
                "std": float(top_q_vals.std())
            },
            "action_ranges": best_action_ranges
        },
        "worst_actions_bottom5pct": {
            "q_value_range": {
                "min": float(bottom_q_vals.min()),
                "max": float(bottom_q_vals.max()),
                "mean": float(bottom_q_vals.mean()),
                "std": float(bottom_q_vals.std())
            },
            "action_ranges": worst_action_ranges
        },
        "overall_q_value_range": {
            "min": float(q_vals.min()),
            "max": float(q_vals.max()),
            "mean": float(q_vals.mean()),
            "std": float(q_vals.std())
        }
    }

print("\n" + "=" * 70)
print("STATE-BASED FAILURE RULES WITH ACTION RECOMMENDATIONS")
print("=" * 70)
print("\nFor each bad state region, we show:")
print("  1. State conditions that define the region")
print("  2. Which actions lead to HIGH Q-values (good actions)")
print("  3. Which actions lead to LOW Q-values (actions to avoid)")
print("=" * 70 + "\n")

if not final_rules:
    print("❌ No stable rules found.")
else:
    for rule_idx, r in enumerate(final_rules[:5], 1):  # Show top 5 rules
        print(f"RULE {rule_idx}:")
        print(f"Failure Probability: {r['precision']:.2f}")
        print(f"Coverage: {r['coverage']*100:.2f}%")
        print(f"Recall: {r['recall']*100:.2f}%")
        print("\nState Conditions:")
        for lime_idx, op, v, feature_label in r["rule"]:
            if isinstance(feature_label, str):
                # Concept mode: feature_label is already a readable name
                feature_name = feature_label
                print(f"  {feature_name} {op} {v:.3f}")
            else:
                # Raw mode: feature_label is the original state index
                feature_name = get_feature_name(feature_label, "state")
                print(f"  {feature_name} {op} {v:.3f}")
                print(f"    (State_{feature_label})")
        
        # Analyze actions for this rule
        if r["example_states"]:
            print("\nAction Analysis (for example states in this region):")
            
            action_recs = compute_action_recommendations(r)
            
            if action_recs:
                best_q = action_recs['best_actions_top5pct']['q_value_range']
                print(f"  ✅ Best Actions (Top 5%, Q-range: [{best_q['min']:.2f}, {best_q['max']:.2f}]):")
                for ar in action_recs['best_actions_top5pct']['action_ranges'][:4]:
                    print(f"    {ar['action_name']}: [{ar['min']:.3f}, {ar['max']:.3f}] (mean: {ar['mean']:.3f})")
                
                worst_q = action_recs['worst_actions_bottom5pct']['q_value_range']
                print(f"\n  ❌ Worst Actions (Bottom 5%, Q-range: [{worst_q['min']:.2f}, {worst_q['max']:.2f}]):")
                for ar in action_recs['worst_actions_bottom5pct']['action_ranges'][:4]:
                    print(f"    {ar['action_name']}: [{ar['min']:.3f}, {ar['max']:.3f}] (mean: {ar['mean']:.3f})")
                
                print(f"\n  Overall Q-value range: [{action_recs['overall_q_value_range']['min']:.2f}, {action_recs['overall_q_value_range']['max']:.2f}]")
        
        print("-" * 70)

# --------------------------------------------------
# 9. RULE MERGING AND SIMPLIFICATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("MERGED/SIMPLIFIED RULES")
print("=" * 70)
print("\nRemoving duplicates and subsumed rules...")

def rules_identical(rule1, rule2):
    """Check if two rules are identical."""
    return set(rule1) == set(rule2)

def rule_subsumes(rule1, rule2):
    """Check if rule1 subsumes rule2 (rule1 is more general)."""
    # rule1 subsumes rule2 if all conditions of rule1 are in rule2
    conditions1 = set((reduced_idx, op, v) for reduced_idx, op, v, _ in rule1)
    conditions2 = set((reduced_idx, op, v) for reduced_idx, op, v, _ in rule2)
    return conditions1.issubset(conditions2) and len(conditions1) < len(conditions2)

def rules_contradict(rule1, rule2):
    """Check if two rules have contradictory conditions on the same feature."""
    features1 = {(reduced_idx, orig_idx): (op, v) for reduced_idx, op, v, orig_idx in rule1}
    features2 = {(reduced_idx, orig_idx): (op, v) for reduced_idx, op, v, orig_idx in rule2}
    
    for key in features1:
        if key in features2:
            op1, v1 = features1[key]
            op2, v2 = features2[key]
            
            # Check for contradictions
            if op1 == "<=" and op2 == ">" and v1 < v2:
                return True
            if op1 == ">" and op2 == "<=" and v1 > v2:
                return True
    
    return False

# Step 1: Remove exact duplicates
unique_rules = []
seen_rules = set()

for r in final_rules:
    rule_tuple = tuple(sorted(r["rule"]))
    if rule_tuple not in seen_rules:
        seen_rules.add(rule_tuple)
        unique_rules.append(r)

print(f"After removing duplicates: {len(unique_rules)} rules (removed {len(final_rules) - len(unique_rules)})")

# Step 2: Remove subsumed rules (keep more general rules)
# A rule is subsumed if there exists a more general rule with equal or better precision
merged_rules = []
removed_subsumed = 0

for i, r1 in enumerate(unique_rules):
    is_subsumed = False
    
    for j, r2 in enumerate(unique_rules):
        if i == j:
            continue
        
        # Check if r1 is subsumed by r2
        # r2 subsumes r1 if:
        # 1. r2 has fewer conditions (more general)
        # 2. All r2's conditions are in r1
        # 3. r2 has similar or better precision
        if rule_subsumes(r2["rule"], r1["rule"]):
            if r2["precision"] >= r1["precision"] * 0.95:  # Allow 5% precision drop
                is_subsumed = True
                removed_subsumed += 1
                break
    
    if not is_subsumed:
        merged_rules.append({
            "rule": r1["rule"],
            "precision": r1["precision"],
            "support": r1["support"],
            "coverage": r1["coverage"],
            "recall": r1.get("recall", 0.0),
            "merged_from": [final_rules.index(r1) + 1],  # Track original index
            "example_states": r1.get("example_states", [])  # Preserve example states
        })

print(f"After removing subsumed rules: {len(merged_rules)} rules (removed {removed_subsumed})")

# Step 3: Try to merge similar rules with overlapping conditions
# Only merge if they share ALL but one condition and don't contradict
final_merged = []
used_indices = set()

for i, r1 in enumerate(merged_rules):
    if i in used_indices:
        continue
    
    merged = False
    for j, r2 in enumerate(merged_rules[i+1:], i+1):
        if j in used_indices:
            continue
        
        # Count common conditions
        conditions1 = set((reduced_idx, op, v) for reduced_idx, op, v, _ in r1["rule"])
        conditions2 = set((reduced_idx, op, v) for reduced_idx, op, v, _ in r2["rule"])
        common = conditions1 & conditions2
        
        # Merge if they share all but one condition
        if len(common) == len(conditions1) - 1 == len(conditions2) - 1:
            if not rules_contradict(r1["rule"], r2["rule"]):
                # Create merged rule with only common conditions
                merged_conditions = []
                for reduced_idx, op, v, orig_idx in r1["rule"]:
                    if (reduced_idx, op, v) in common:
                        merged_conditions.append((reduced_idx, op, v, orig_idx))
                
                merged_rule_tuple = tuple(sorted(merged_conditions))
                
                # Validate merged rule
                mask = rule_mask(X_states_reduced, merged_rule_tuple)
                support = mask.sum()
                
                if support >= MIN_RULE_SUPPORT:
                    bad_count = labels[mask].sum()
                    precision = bad_count / support
                    
                    # Only merge if precision doesn't drop too much
                    if precision >= min(r1["precision"], r2["precision"]) * 0.9:
                        # Combine example states from both rules
                        combined_examples = r1.get("example_states", []) + r2.get("example_states", [])
                        
                        # Calculate recall
                        recall = bad_count / num_bad
                        
                        final_merged.append({
                            "rule": merged_rule_tuple,
                            "precision": precision,
                            "support": support,
                            "coverage": support / len(X_lime),
                            "recall": recall,
                            "merged_from": sorted(r1["merged_from"] + r2["merged_from"]),
                            "example_states": combined_examples[:10]  # Keep first 10 examples
                        })
                        used_indices.add(i)
                        used_indices.add(j)
                        merged = True
                        break
    
    # If not merged, keep original
    if not merged and i not in used_indices:
        final_merged.append(r1)

merged_rules = final_merged
merged_rules.sort(key=lambda x: (-x["precision"], -x["support"]))

print(f"After merging similar rules: {len(merged_rules)} rules")
print(f"\nTotal reduction: {len(final_rules)} → {len(merged_rules)} rules")
print("=" * 70 + "\n")

if merged_rules:
    for idx, r in enumerate(merged_rules[:10], 1):  # Show top 10 merged rules
        print(f"MERGED RULE {idx}:")
        
        if len(r["merged_from"]) > 1:
            print(f"  (Merged from original rules: {', '.join(map(str, r['merged_from']))})")
        else:
            print(f"  (Original rule {r['merged_from'][0]})")
        
        print(f"Failure Probability: {r['precision']:.2f}")
        print(f"Coverage: {r['coverage']*100:.2f}%")
        print(f"Support: {r['support']:,} states")
        print("\nConditions:")
        for reduced_idx, op, v, original_idx in r["rule"]:
            feature_name = get_feature_name(original_idx, "state")
            print(f"  {feature_name} {op} {v:.3f}")
        print("-" * 70)
else:
    print("No rules after merging.")

# --------------------------------------------------
# 10. SAVE RULES TO FILES
# --------------------------------------------------

import json
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save original rules
original_rules_data = {
    "metadata": {
        "timestamp": timestamp,
        "total_states": len(X_states_reduced),
        "bad_percentile": BAD_PERCENTILE,
        "num_states_explained": NUM_STATES_TO_EXPLAIN,
        "num_lime_features": NUM_LIME_FEATURES,
        "min_rule_support": MIN_RULE_SUPPORT,
        "min_rule_precision": MIN_RULE_PRECISION,
        "num_rules": len(final_rules),
    },
    "rules": []
}

for idx, r in enumerate(final_rules, 1):
    rule_dict = {
        "rule_id": idx,
        "precision": float(r["precision"]),
        "coverage": float(r["coverage"]),
        "support": int(r["support"]),
        "recall": float(r["recall"]),
        "conditions": []
    }
    
    for lime_idx, op, v, feature_label in r["rule"]:
        if isinstance(feature_label, str):
            feature_name = feature_label
            feature_index = feature_label # Using the string name for JSON
        else:
            feature_name = get_feature_name(feature_label, "state")
            feature_index = int(feature_label)

        rule_dict["conditions"].append({
            "feature_index": feature_index,
            "feature_name": feature_name,
            "operator": op,
            "threshold": float(v)
        })
    
    # Add action recommendations
    action_recs = compute_action_recommendations(r)
    if action_recs:
        rule_dict["action_recommendations"] = action_recs
    
    original_rules_data["rules"].append(rule_dict)

# Save merged rules
merged_rules_data = {
    "metadata": {
        "timestamp": timestamp,
        "total_states": len(X_states_reduced),
        "bad_percentile": BAD_PERCENTILE,
        "num_original_rules": len(final_rules),
        "num_merged_rules": len(merged_rules),
        "num_rules_merged": len(final_rules) - len([r for r in merged_rules if len(r['merged_from']) == 1]),
    },
    "rules": []
}

for idx, r in enumerate(merged_rules, 1):
    rule_dict = {
        "rule_id": idx,
        "precision": float(r["precision"]),
        "coverage": float(r["coverage"]),
        "support": int(r["support"]),
        "recall": float(r.get("recall", 0.0)),  # Use get() in case recall not present
        "merged_from": r["merged_from"],
        "conditions": []
    }
    
    for lime_idx, op, v, feature_label in r["rule"]:
        if isinstance(feature_label, str):
            feature_name = feature_label
            feature_index = feature_label
        else:
            feature_name = get_feature_name(feature_label, "state")
            feature_index = int(feature_label)

        rule_dict["conditions"].append({
            "feature_index": feature_index,
            "feature_name": feature_name,
            "operator": op,
            "threshold": float(v)
        })
    
    # Add action recommendations
    action_recs = compute_action_recommendations(r)
    if action_recs:
        rule_dict["action_recommendations"] = action_recs
    
    merged_rules_data["rules"].append(rule_dict)

# Save to JSON files
original_json_path = f"extracted_rules/original_rules_{timestamp}.json"
merged_json_path = f"extracted_rules/merged_rules_{timestamp}.json"

import os
os.makedirs("extracted_rules", exist_ok=True)

with open(original_json_path, 'w') as f:
    json.dump(original_rules_data, f, indent=2)

with open(merged_json_path, 'w') as f:
    json.dump(merged_rules_data, f, indent=2)

# Save to human-readable text files
original_txt_path = f"extracted_rules/original_rules_{timestamp}.txt"
merged_txt_path = f"extracted_rules/merged_rules_{timestamp}.txt"

with open(original_txt_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("ORIGINAL EXTRACTED RULES\n")
    f.write("=" * 70 + "\n")
    f.write(f"\nGenerated: {timestamp}\n")
    f.write(f"Total states: {len(X_states_reduced):,}\n")
    f.write(f"Bad percentile: {BAD_PERCENTILE}%\n")
    f.write(f"Number of rules: {len(final_rules)}\n")
    f.write("=" * 70 + "\n\n")
    
    for idx, r in enumerate(final_rules, 1):
        f.write(f"RULE {idx}:\n")
        f.write(f"Failure Probability: {r['precision']:.2f}\n")
        f.write(f"Coverage: {r['coverage']*100:.2f}%\n")
        f.write(f"Support: {r['support']:,} states\n")
        f.write("\nConditions:\n")
        for reduced_idx, op, v, original_idx in r["rule"]:
            feature_name = get_feature_name(original_idx, "state")
            f.write(f"  {feature_name} {op} {v:.3f}\n")
            f.write(f"    (State_{original_idx})\n")
        f.write("-" * 70 + "\n\n")

with open(merged_txt_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MERGED/SIMPLIFIED RULES\n")
    f.write("=" * 70 + "\n")
    f.write(f"\nGenerated: {timestamp}\n")
    f.write(f"Original rules: {len(final_rules)}\n")
    f.write(f"After merging: {len(merged_rules)}\n")
    f.write(f"Rules merged: {len(final_rules) - len([r for r in merged_rules if len(r['merged_from']) == 1])}\n")
    f.write("=" * 70 + "\n\n")
    
    for idx, r in enumerate(merged_rules, 1):
        f.write(f"MERGED RULE {idx}:\n")
        
        if len(r["merged_from"]) > 1:
            f.write(f"  (Merged from original rules: {', '.join(map(str, r['merged_from']))})\n")
        else:
            f.write(f"  (Original rule {r['merged_from'][0]})\n")
        
        f.write(f"Failure Probability: {r['precision']:.2f}\n")
        f.write(f"Coverage: {r['coverage']*100:.2f}%\n")
        f.write(f"Support: {r['support']:,} states\n")
        f.write("\nConditions:\n")
        for reduced_idx, op, v, original_idx in r["rule"]:
            feature_name = get_feature_name(original_idx, "state")
            f.write(f"  {feature_name} {op} {v:.3f}\n")
            f.write(f"    (State_{original_idx})\n")
        f.write("-" * 70 + "\n\n")

print("\n" + "=" * 70)
print("RULES SAVED")
print("=" * 70)
print(f"\nOriginal rules saved to:")
print(f"  JSON: {original_json_path}")
print(f"  Text: {original_txt_path}")
print(f"\nMerged rules saved to:")
print(f"  JSON: {merged_json_path}")
print(f"  Text: {merged_txt_path}")
print("=" * 70)

print("\nDone!")


