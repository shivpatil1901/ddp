#!/usr/bin/env python
"""Convert trajectory-list pickles into the flat SafetyGym dataset format.

This script is meant for datasets produced by the new trajectory builders,
where the pickle contains:

{
  "trajectories": [
    {
      "states": ...,
      "actions": ...,
      "next_states": ...,
      "rewards": ...,
      "costs": ...,
      "dones": ...,
    },
    ...
  ],
  "metadata": {...}
}

It converts that structure into the flat dict expected by main_safetygym.py:

{
  "init_states": ..., "states": ..., "actions": ..., "next_states": ...,
  "costs": ..., "rewards": ..., "dones": ...
}
"""

import argparse
import os

import numpy as np

try:
    import pickle5 as pickle
except ImportError:
    import pickle


DEFAULT_INPUT_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle"
DEFAULT_OUTPUT_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0_modified.pickle"


def _ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _load_pickle_compat(path):
    try:
        with open(path, 'rb') as fr:
            return pickle.load(fr)
    except ValueError as exc:
        if 'unsupported pickle protocol: 5' in str(exc):
            raise RuntimeError(
                'This pickle uses protocol 5. Use Python >= 3.8 or install pickle5 in Python 3.6/3.7.'
            ) from exc
        raise


def _trajectory_dataset_to_flat_dict(dataset):
    trajectories = dataset.get('trajectories', [])
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError('Expected a non-empty "trajectories" list in the input pickle')

    init_states_parts = []
    states_parts = []
    actions_parts = []
    next_states_parts = []
    costs_parts = []
    rewards_parts = []
    dones_parts = []

    for traj in trajectories:
        if not isinstance(traj, dict):
            continue

        states = np.asarray(traj.get('states', []), dtype=np.float32)
        actions = np.asarray(traj.get('actions', []), dtype=np.float32)
        next_states = np.asarray(traj.get('next_states', []), dtype=np.float32)
        rewards = np.asarray(traj.get('rewards', []), dtype=np.float32)
        costs = np.asarray(traj.get('costs', []), dtype=np.float32)
        dones = np.asarray(traj.get('dones', []), dtype=np.float32)

        n = int(min(len(states), len(actions), len(next_states), len(rewards), len(costs), len(dones)))
        if n <= 0:
            continue

        states = states[:n]
        actions = actions[:n]
        next_states = next_states[:n]
        rewards = rewards[:n]
        costs = costs[:n]
        dones = dones[:n]

        init_state = np.repeat(states[:1], n, axis=0)

        init_states_parts.append(_ensure_2d(init_state).astype(np.float32))
        states_parts.append(_ensure_2d(states).astype(np.float32))
        actions_parts.append(_ensure_2d(actions).astype(np.float32))
        next_states_parts.append(_ensure_2d(next_states).astype(np.float32))
        costs_parts.append(_ensure_2d(costs).astype(np.float32))
        rewards_parts.append(_ensure_2d(rewards).astype(np.float32))
        dones_parts.append(_ensure_2d(dones).astype(np.float32))

    if not states_parts:
        raise ValueError('No valid trajectories were found in the input pickle')

    return {
        'init_states': np.concatenate(init_states_parts, axis=0).astype(np.float32),
        'states': np.concatenate(states_parts, axis=0).astype(np.float32),
        'actions': np.concatenate(actions_parts, axis=0).astype(np.float32),
        'next_states': np.concatenate(next_states_parts, axis=0).astype(np.float32),
        'costs': np.concatenate(costs_parts, axis=0).astype(np.float32),
        'rewards': np.concatenate(rewards_parts, axis=0).astype(np.float32),
        'dones': np.concatenate(dones_parts, axis=0).astype(np.float32),
    }


def _maybe_flatten_dataset(data):
    if isinstance(data, dict) and 'trajectories' in data:
        return _trajectory_dataset_to_flat_dict(data)
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Convert a trajectory-list SafetyGym pickle into the flat init_states/states/actions format.'
    )
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_PATH, help='Input pickle file')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH, help='Output pickle file')
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input))
    output_path = os.path.abspath(os.path.expanduser(args.output))

    data = _load_pickle_compat(input_path)
    converted = _maybe_flatten_dataset(data)

    if not isinstance(converted, dict):
        raise TypeError('Expected a dict after conversion, got %s' % type(converted))

    required_keys = ['init_states', 'states', 'actions', 'next_states', 'costs', 'rewards', 'dones']
    missing = [key for key in required_keys if key not in converted]
    if missing:
        raise KeyError('Converted dataset is missing keys: %s' % ', '.join(missing))

    flat = {}
    for key in required_keys:
        flat[key] = _ensure_2d(converted[key]).astype(np.float32)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(flat, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Loaded :', input_path)
    print('Saved  :', output_path)
    for key in required_keys:
        print('  %s: %s' % (key, flat[key].shape))


if __name__ == '__main__':
    main()