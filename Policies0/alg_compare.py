import gymnasium as gym
from stable_baselines3 import TD3, SAC, PPO, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from success_callback import StopTrainingOnSuccessRate
import tensorboard
import numpy as np
from typing import Callable
import datetime
from git import Repo, InvalidGitRepositoryError
import argparse
import re
import wandb

#repo_path = '/home/catherine/FractureSoftGym/fracturesurgeryenv'
repo_path = "/users/cop21cma/FractureSoftGym/fracturesurgeryenv"
def get_git_commit_hash(repo_path):
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        print(f"Invalid Git repository at {repo_path}")
    except Exception as e:
        print(f"An error occurred while getting the commit hash: {e}")
        return None

def train(threshold_pos=0.001, threshold_ori=np.deg2rad(6), action_type='pos_only', seed=42, reward='sparse', model='TD3', ran=1):
    commit = get_git_commit_hash(repo_path)
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type
    threshold_pos = threshold_pos
    threshold_ori = np.deg2rad(threshold_ori)
    model_name = model  # keep the requested model name separate from the instantiated model
    eval_seed = 42
    #print(model_name)
    tag = 'alg_compare_3'
    wandb.init(project="Chapter1-Results", name = (f'{train_date}-{model}-{reward}-{seed}'),tags=[tag],notes= (f"Git Commit: {commit}, seed: {seed}"),sync_tensorboard=True, save_code=True)  # Initialize W&B
    
    env_kwargs = {
        'reward_type': reward,
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'dv': 0.001,
        'distance_threshold_ori': threshold_ori,
        'action_type': action_type,
        'render_mode':'direct'}

    #env = SubprocVecEnv([make_env(threshold_pos, threshold_ori, action_type) for _ in range(2)])
    env = make_vec_env('gym_fracture:anklesurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=DummyVecEnv, seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    #env = gym.make('gym_fracture:anklesurg-v0', **env_kwargs)

    if model_name == 'TD3':
        m='t'
        if reward == 'sparse':
            model = TD3(policy="MultiInputPolicy",
                        env=env, verbose=0,
                        replay_buffer_class=HerReplayBuffer,
                        replay_buffer_kwargs=dict(n_sampled_goal=4),
                        seed=seed,
                        tensorboard_log=f'./logs/{ran}')
        elif reward.startswith('dense'):
            model = TD3(policy="MultiInputPolicy",
                        env=env, verbose=0,
                        seed=seed,
                        tensorboard_log=f'./logs/{ran}')
    elif model_name == 'SAC':
        m='s'
        if reward == 'sparse':
            model = SAC(policy="MultiInputPolicy",
                        env=env, verbose=0,
                        replay_buffer_class=HerReplayBuffer,
                        replay_buffer_kwargs=dict(n_sampled_goal=4),
                        seed=seed,
                        tensorboard_log=f'./logs/{ran}')
        elif reward.startswith('dense'):
            model = SAC(policy="MultiInputPolicy",
                        env=env, verbose=0,
                        seed=seed,
                        tensorboard_log=f'./logs/{ran}')
    elif model_name == 'PPO':
        m='p'
        model = PPO(policy="MultiInputPolicy",
                    env=env, verbose=0,
                    seed=seed,
                    tensorboard_log=f'./logs/{ran}')


    
    # Separate evaluation env
    eval_env = make_vec_env('gym_fracture:anklesurg-v0', env_kwargs=env_kwargs, n_envs=10, vec_env_cls=SubprocVecEnv, seed=eval_seed)
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, max_no_improvement_evals=1,
                                                                success_threshold=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(eval_env,  eval_freq=10000, 
                                deterministic=True, n_eval_episodes=100, callback_after_eval=success_callback)
    
    model.learn(2_000_000, callback=eval_callback)
    #model.save(f'./model-{train_date}-{m}-{reward}-{seed}')

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 model with specified thresholds and action type.')
    parser.add_argument('--threshold_pos', type=float, default=0.005, help='Position threshold for the environment.')
    parser.add_argument('--threshold_ori', type=float, default=0.05, help='Orientation threshold for the environment.')
    parser.add_argument('--action_type', type=str, default='fouractions', help='Type of action to use in the environment.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--reward', type=str, default='sparse', help='Reward type to use in training.')
    parser.add_argument('--model', type=str, default='TD3', help='Model type to use for training (TD3, SAC, PPO).')
    parser.add_argument('--ran', type=int, default=1, help='Random seed for the run, used in logging and model naming.')
    args = parser.parse_args()
    train(threshold_pos=args.threshold_pos,
          threshold_ori=args.threshold_ori,
          action_type=args.action_type,
          seed=args.seed,
          reward=args.reward,
          model=args.model,
          ran=args.ran)
