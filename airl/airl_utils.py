"""
AIRL Utility Functions - Contains only helper functions and wrappers without training code.
This module can be safely imported without running any training.
"""

import gymnasium as gym
import safety_gymnasium
from stable_baselines3.common.monitor import Monitor
from imitation.data.wrappers import RolloutInfoWrapper


class SafetyEnvToGymEnvWrapper(gym.Wrapper):
    """
    Wraps a 6-return Safety-Gymnasium env to return 5 values,
    dropping the 'cost' for compatibility with stable-baselines3 evaluation.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


def make_env(env_id, seed, rank, post_wrappers):
    """
    Utility function for multiprocessing_vec
    """
    def _init():
        # Call safety_gymnasium.make directly and disable the checker
        env = safety_gymnasium.make(env_id, disable_env_checker=True)
        
        # Apply all our wrappers IN ORDER
        for wrapper_fn in post_wrappers:
            env = wrapper_fn(env, rank)  # rank is needed for Monitor
            
        env.reset(seed=seed + rank)
        return env
    return _init
