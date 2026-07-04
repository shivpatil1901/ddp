import pickle5 as pickle
import numpy as np

with open('/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle', 'rb') as f:
    data = pickle.load(f)

obs = data['observations']
actions = data['actions']
rewards = data['rewards']
costs = data['costs']
dones = data['dones']

print("Loaded data, converting...")

# next_states: shift obs by 1
next_states = np.empty_like(obs)
next_states[:-1] = obs[1:]
next_states[-1] = 0
next_states[dones == 1] = 0

# init_states: vectorized using repeat
episode_start_flags = np.zeros(len(obs), dtype=bool)
episode_start_flags[0] = True
episode_starts = np.where(dones == 1)[0] + 1
episode_starts = episode_starts[episode_starts < len(obs)]
episode_start_flags[episode_starts] = True

start_indices = np.where(episode_start_flags)[0]
episode_lengths = np.diff(np.append(start_indices, len(obs)))
init_states = np.repeat(obs[start_indices], episode_lengths, axis=0)

converted = {
    'init_states': init_states.astype(np.float32),
    'states':      obs.astype(np.float32),
    'actions':     actions.astype(np.float32),
    'next_states': next_states.astype(np.float32),
    'costs':       costs.reshape(-1, 1).astype(np.float32),
    'rewards':     rewards.reshape(-1, 1).astype(np.float32),
    'dones':       dones.reshape(-1, 1).astype(np.float32),
}

print("Converted shapes:")
for k, v in converted.items():
    print(f"  {k}: {v.shape}, {v.dtype}")

out_path = '/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0_converted.pickle'
with open(out_path, 'wb') as f:
    pickle.dump(converted, f)

print(f"\nSaved to: {out_path}")