# 0. Install conda environment
```
./make_conda_env.sh
conda activate safeil
```


# 1. Train data-collection policy
```
python train.py --algo=ppo --robot=point --task=goal1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo --robot=point --task=button1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo --robot=point --task=push1 --seed=0 --cost_lim=25 --cpu=10

python train.py --algo=ppo_lagrangian --robot=point --task=goal1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo_lagrangian --robot=point --task=button1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo_lagrangian --robot=point --task=push1 --seed=0 --cost_lim=25 --cpu=10

python train.py --algo=ppo --robot=car --task=goal1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo --robot=car --task=button1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo --robot=car --task=push1 --seed=0 --cost_lim=25 --cpu=10

python train.py --algo=ppo_lagrangian --robot=car --task=goal1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo_lagrangian --robot=car --task=button1 --seed=0 --cost_lim=25 --cpu=10
python train.py --algo=ppo_lagrangian --robot=car --task=push1 --seed=0 --cost_lim=25 --cpu=10
```
This will generate the checkpoints in `data` directory, which will be used for dataset generation later.
* cost_lim: cost violation threshold

# 2. Test the saved policy (Can skip this)

```
python test_policy.py data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0 --itr=300 --render
```
* itr: the number of epochs


# 3. Generate datasets
```
python generate_dataset.py data/ppo_PointGoal1/ppo_PointGoal1_s0 --itr=300 --num_episodes=1000
python generate_dataset.py data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0 --itr=300 --num_episodes=1000
(...)
```
* itr: the number of epochs
