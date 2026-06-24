#!/usr/bin/env python

import os
import numpy as np
import gym
import safety_gym
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import cv2
from tqdm import tqdm


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=False):
    if env is None:
        raise RuntimeError(
            "Environment not found. Pass --env_id to create one manually, "
            "for example --env_id Safexp-PointGoal1-v0"
        )

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    frames = []
    pbar = tqdm(desc='run_policy', ncols=80)
    while n < num_episodes:
        if render:
            frame = env.render(mode='rgb_array')
            frames.append(frame)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1
        pbar.update(1)

        if d or (ep_len == max_ep_len):
            if render:
                os.makedirs('videos', exist_ok=True)
                video_filepath = f"videos/{args.fpath.split('/')[-1]}_{n}.mp4"
                fps = 30
                width, height, channels = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_filepath, fourcc, float(fps), (width, height))
                for frame in frames:
                    video.write(frame)
                video.release()
                print(f'Video saved to: {video_filepath}')
                os.system(f"ffmpeg -y -i {video_filepath} -vcodec libx264 -f mp4 {video_filepath.split('.mp4')[0] + '_encoded.mp4'}")
                frames = []

            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            pbar = tqdm(desc='run_policy', ncols=80)
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--max_ep_len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=10)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--env_id', '-e', type=str, default=None,
                        help='Gym env id to use if env was not saved in vars*.pkl')
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)

    if env is None and args.env_id is not None:
        env = gym.make(args.env_id)
        print(f'Loaded policy without saved env. Created env: {args.env_id}')

    run_policy(env, get_action, args.max_ep_len, args.episodes, bool(args.render))
