import pickle
import numpy as np

files = [
    '/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle',
    '/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0.pickle',
]

for fpath in files:
    print(f"\n{'='*60}")
    print(f"File: {fpath.split('/')[-1]}")
    print('='*60)

    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    rewards = data['rewards'].flatten()
    costs   = data['costs'].flatten()
    dones   = data['dones'].flatten()

    ep_rewards, ep_costs = [], []
    ep_ret, ep_cost = 0, 0

    for r, c, d in zip(rewards, costs, dones):
        ep_ret  += r
        ep_cost += c
        if d == 1:
            ep_rewards.append(ep_ret)
            ep_costs.append(ep_cost)
            ep_ret, ep_cost = 0, 0

    ep_rewards = np.array(ep_rewards)
    ep_costs   = np.array(ep_costs)

    # print(f"Episodes : {len(ep_rewards)}")
    # print(f"EpReward : mean={ep_rewards.mean():.3f}, std={ep_rewards.std():.3f}")
    # print(f"EpCost   : mean={ep_costs.mean():.3f},  std={ep_costs.std():.3f}")
    # print(f"EpCost>0 : {np.mean(ep_costs > 0):.3f}")

    print(f"Episodes : {len(ep_rewards)}")

    print(
        f"EpReward : "
        f"mean={ep_rewards.mean():.3f}, "
        f"std={ep_rewards.std():.3f}, "
        f"min={ep_rewards.min():.3f}, "
        f"max={ep_rewards.max():.3f}, "
        f">2={np.sum(ep_rewards > 2)}, "
        f">6={np.sum(ep_rewards > 6)}"
        f">6 & cost<25={np.sum((ep_rewards > 6) & (ep_costs < 25))}, "
        f"<2 & cost>25={np.sum((ep_rewards < 2.5) & (ep_costs > 25))}"

    )

    print(
        f"EpCost   : "
        f"mean={ep_costs.mean():.3f}, "
        f"std={ep_costs.std():.3f}, "
        f"min={ep_costs.min():.3f}, "
        f"max={ep_costs.max():.3f},"
        f">25={np.sum(ep_costs > 25)}"

    )

    print(f"EpCost>0 : {np.mean(ep_costs > 0):.3f}")