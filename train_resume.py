#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
import os
import glob
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.utils.mpi_tools import proc_id
import time
import os.path as osp

try:
    import wandb
except ImportError:
    wandb = None


def find_latest_simple_save(output_dir):
    """Returns (epoch_number, path) of the latest simple_saveXXX dir, or (None, None)."""
    save_dirs = glob.glob(osp.join(output_dir, 'simple_save*'))
    if not save_dirs:
        return None, None
    latest = max(save_dirs, key=lambda d: int(osp.basename(d).replace('simple_save', '')))
    epoch = int(osp.basename(latest).replace('simple_save', ''))
    return epoch, latest


def main(robot, task, algo, seed, cost_lim, num_cpus, use_wandb=True, restore_path=None, start_epoch=0):
    # Verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'button1', 'push1', 'goal2', 'button2', 'push2']
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    total_epochs = int(num_steps / steps_per_epoch)
    save_freq = 10
    target_kl = 0.01

    # Fork for parallelizing
    mpi_fork(num_cpus)

    # Prepare Logger
    exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir='./data_new', datestamp=False)

    # ------------------------------------------------------------------ #
    # Resume logic: auto-detect latest checkpoint if --resume is passed
    # ------------------------------------------------------------------ #
    if restore_path == 'auto':
        output_dir = logger_kwargs['output_dir']
        policy_epoch, policy_path = find_latest_simple_save(output_dir)
        if policy_path is not None:
            restore_path = policy_path
            start_epoch  = policy_epoch
            print(f"[Resume] Found checkpoint at epoch {policy_epoch}: {policy_path}")
        else:
            restore_path = None
            start_epoch  = 0
            print(f"[Resume] No checkpoint found in {output_dir}, starting from scratch.")

    remaining_epochs = total_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Training already complete ({start_epoch}/{total_epochs} epochs). Exiting.")
        return

    if start_epoch > 0:
        print(f"[Info] Resuming: epochs {start_epoch} → {total_epochs} ({remaining_epochs} remaining)")
    # ------------------------------------------------------------------ #

    if proc_id() == 0 and use_wandb:
        if wandb is None:
            print("wandb is not installed; continuing without wandb logging.")
        else:
            try:
                wandb.init(
                    project='saferl',
                    name=f'{exp_name}_s{seed}',
                    resume='allow' if start_epoch > 0 else None,
                    config={
                        'robot': robot,
                        'task': task,
                        'algo': algo,
                        'seed': seed,
                        'total_epochs': total_epochs,
                        'start_epoch': start_epoch,
                        'save_freq': save_freq,
                        'target_kl': target_kl,
                        'cost_lim': cost_lim,
                    }
                )
            except Exception as exc:
                print("wandb init failed (%s); continuing without wandb logging." % str(exc))

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'

    algo(
        env_fn=lambda: gym.make(env_name),
        ac_kwargs=dict(hidden_sizes=(256, 256),),
        epochs=remaining_epochs,
        steps_per_epoch=steps_per_epoch,
        save_freq=save_freq,
        save_critic=True,
        save_critic_freq=1,
        critic_checkpoint_dir=f'critic_checkpoints_{robot.lower()}_{task.lower()}_s{seed}',
        target_kl=target_kl,
        cost_lim=cost_lim,
        seed=seed,
        logger_kwargs=logger_kwargs,
        restore_path=restore_path,   # passed to run_agent.py line 56
        start_epoch=start_epoch,     # passed to run_agent.py line 57
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=10)
    parser.add_argument('--cost_lim', type=float, default=25)
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases logging')
    # Resume args
    parser.add_argument('--resume', action='store_true',
                        help='Auto-detect and resume from latest checkpoint')
    parser.add_argument('--restore_path', type=str, default=None,
                        help='Path to specific simple_saveXXX dir to restore from')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Epoch number to resume from (used with --restore_path)')
    args = parser.parse_args()

    # --resume sets restore_path to 'auto' for auto-detection
    restore_path = 'auto' if args.resume else args.restore_path
    start_epoch  = 0 if args.resume else args.start_epoch

    main(args.robot, args.task, args.algo, args.seed, args.cost_lim, args.cpu,
         use_wandb=(not args.disable_wandb),
         restore_path=restore_path,
         start_epoch=start_epoch)