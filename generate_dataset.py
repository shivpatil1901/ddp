#!/usr/bin/env python

import os
import numpy as np
from safe_rl.utils.load_utils import load_policy
import pickle
from tqdm import tqdm


def generate_dataset(env, get_action, max_ep_len=None, num_episodes=100):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("

    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'costs': [],
        'dones': [],
    }

    obs, reward, done, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    pbar = tqdm(desc='run_policy', ncols=80)
    while n < num_episodes:
        action = get_action(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = env.step(action)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_len += 1
        pbar.update(1)

        dataset['observations'].append(obs)
        dataset['actions'].append(action)
        dataset['rewards'].append(reward)
        dataset['costs'].append(info.get('cost', 0))
        dataset['dones'].append(done)

        obs = next_obs
        if done or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            obs, reward, done, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            pbar = tqdm(desc='run_policy', ncols=80)
            n += 1

    dataset['observations'] = np.array(dataset['observations'], dtype=np.float32)
    dataset['actions'] = np.array(dataset['actions'], dtype=np.float32)
    dataset['rewards'] = np.array(dataset['rewards'], dtype=np.float32)
    dataset['costs'] = np.array(dataset['costs'], dtype=np.float32)
    dataset['dones'] = np.array(dataset['dones'], dtype=np.float32)

    os.makedirs('datasets_new', exist_ok=True)
    with open(f"datasets_new/{args.fpath.split('/')[-1]}.pickle", 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--max_ep_len', '-l', type=int, default=0)
    parser.add_argument('--num_episodes', '-n', type=int, default=100)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=100)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)

    generate_dataset(env, get_action, args.max_ep_len, args.num_episodes)
