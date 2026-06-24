#!/usr/bin/env python
"""
Quick script to inspect the PointGoal1 observation space structure.
"""
import sys
sys.path.insert(0, '/home/ed21b059/ddp/3rdparty/safety-gym')
sys.path.insert(0, '/home/ed21b059/ddp/3rdparty/safety-starter-agents')

import numpy as np
import gym
import safety_gym

LIDAR_ANGLES = [i * 22.5 for i in range(16)]


def print_lidar_order(key, offset, dim):
    if dim != 16:
        return

    pretty_key = key.replace("_", " ").title()
    angles = ", ".join(f"{angle:g}" for angle in LIDAR_ANGLES)
    print(f"  {pretty_key:20} -> angle order [{angles}] -> dims [{offset:3d}:{offset + dim:3d}]")

# Create environment and immediately check obs_space_dict
print("=" * 80)
print("POINTBUTTON1 OBSERVATION SPACE STRUCTURE")
print("=" * 80)
print()

env = gym.make('Safexp-PointButton1-v0')

# Check the dict mode observation space
print("Dict keys (in order from obs_space_dict):")
for i, (key, space) in enumerate(env.obs_space_dict.items()):
    print(f"  [{i:2d}] {key:20} -> shape={space.shape}")

print()
print("Total dimensions when flattened:")
offset = 0
for key in sorted(env.obs_space_dict.keys()):
    space = env.obs_space_dict[key]
    dim = int(np.prod(space.shape))
    print(f"  {key:20} -> dims [{offset:3d}:{offset+dim:3d}] (size {dim})")
    if key.endswith("_lidar"):
        print_lidar_order(key, offset, dim)
    offset += dim

print()
print(f"Total flattened dimensions: {offset}")

# Reset and check flattened
obs_flat = env.reset()
print()
print("=" * 80)
print("ACTUAL OBSERVATION (FLATTENED - DEFAULT)")
print("=" * 80)
print(f"Shape: {obs_flat.shape}")
print(f"Dtype: {obs_flat.dtype}")
print(f"First 20 values: {obs_flat[:20]}")
