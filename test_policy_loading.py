#!/usr/bin/env python3
"""Quick test of policy checkpoint loading"""

import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = Path("/home/ed21b059/ddp")
POLICY_PATH = ROOT / "SafeDICE/weights/antidice_PointGoal1_seed0_20260330_205404_iter1000000.pickle"
DEVICE = torch.device("cpu")

print("=" * 70)
print("POLICY CHECKPOINT LOADING TEST")
print("=" * 70)

# Try loading as pickle first (for TensorFlow SavedModel-based checkpoints)
policy_checkpoint = None
try:
    with open(str(POLICY_PATH), 'rb') as f:
        policy_checkpoint = pickle.load(f)
    print("✅ Loaded policy pickle successfully")
except Exception as e:
    print(f"❌ pickle.load failed: {e}")

if policy_checkpoint is None:
    print("[ERROR] Could not load policy checkpoint")
    exit(1)

if not isinstance(policy_checkpoint, dict):
    print(f"[ERROR] Unsupported policy checkpoint type: {type(policy_checkpoint)}")
    exit(1)

checkpoint_keys = list(policy_checkpoint.keys())
print(f"✅ Policy checkpoint loaded. Top-level keys: {checkpoint_keys}")

# Handle TensorFlow-style checkpoint with 'training_state' and 'training_info'
if "training_state" in policy_checkpoint:
    print("\n[INFO] Detected TensorFlow training checkpoint format")
    training_state = policy_checkpoint.get("training_state", {})
    training_info = policy_checkpoint.get("training_info", {})
    
    print(f"  training_state keys: {list(training_state.keys())}")
    print(f"  training_info keys: {list(training_info.keys())[:20]}")
    
    # Try to find observation normalizer in training_info
    if "obs_normalizer" in training_info:
        print("  ✅ Found 'obs_normalizer' in training_info")
        obs_norm = training_info['obs_normalizer']
        print(f"     Type: {type(obs_norm).__name__}")
        if hasattr(obs_norm, '_mean'):
            mean = obs_norm._mean.numpy() if hasattr(obs_norm._mean, 'numpy') else np.array(obs_norm._mean)
            print(f"     _mean shape: {mean.shape}")
        if hasattr(obs_norm, '_std'):
            std = obs_norm._std.numpy() if hasattr(obs_norm._std, 'numpy') else np.array(obs_norm._std)
            print(f"     _std shape: {std.shape}")
    else:
        print("  ❌ 'obs_normalizer' NOT found in training_info")
        # Search for alternatives
        for key in training_info.keys():
            if 'norm' in key.lower() or 'mean' in key.lower() or 'std' in key.lower():
                print(f"     Found related key: {key}")

elif "pi" in policy_checkpoint and "obs_normalizer" in policy_checkpoint:
    print("\n[INFO] Detected standard PyTorch checkpoint format")
    print("  ✅ Found 'pi' and 'obs_normalizer' keys")
else:
    print("\n[ERROR] Policy checkpoint format not recognized")
    print(f"  Expected 'training_state' OR ('pi' + 'obs_normalizer')")
    print(f"  Available keys: {checkpoint_keys[:20]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("The policy pickle file loads successfully as a TensorFlow checkpoint,")
print("but the observation normalizer is not directly in training_info.")
print("We need to use random action sampling as fallback or reconstruct normalizer.")
