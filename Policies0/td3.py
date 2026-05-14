import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
import tensorboard
#from gym_fracture.envs.fracuresurgery import fracturesurgery_env
from stable_baselines3.common.monitor import Monitor
import wandb
import numpy as np
from typing import Callable
import datetime
from git import Repo, InvalidGitRepositoryError
import argparse
#import log_callback

#repo_path = "/home/catherine/FractureGym/fracturesurgeryenv"
#repo_path="/users/cop21cma/FracSurg-Gym/fracturesurgeryenv"
repo_path = "/users/cop21cma/FracSoftGym/fracturesurgeryenv"
def get_git_commit_hash(repo_path):
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        print(f"Invalid Git repository at {repo_path}")
    except Exception as e:
        print(f"An error occurred while getting the commit hash: {e}")
        return None

def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func


def train(threshold_pos=0.001, threshold_ori=np.deg2rad(6), action_type='pos_only',seed=42):
    commit = get_git_commit_hash(repo_path)
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type
    threshold_pos = threshold_pos
    threshold_ori = np.deg2rad(threshold_ori)
    wandb.init(project="Chp1-Test", name = (f'{train_date}-{action_type}-{threshold_pos}-{np.round(threshold_ori,2)}-{seed}'),notes= (f"Git Commit: {commit}, Ori:{threshold_ori}"),sync_tensorboard=True, save_code=True)  # Initialize W&B
    env_kwargs = {
        'reward_type': 'sparse',
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'distance_threshold_ori' : threshold_ori,
        'dv': 0.001,
        'action_type': action_type,
        'softtissue': False,
        'start_pos' : 'home',
        'render_mode':'direct'}
        
    
    #vec_env=make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=SubprocVecEnv)
    vec_env = gym.make('gym_fracture:softsurg-v0', **env_kwargs)
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(vec_env.action_space.shape[0]), 
                                              sigma=0.02 * np.ones(vec_env.action_space.shape[0]))

    policy_kwargs = dict(net_arch=[256, 256,256])#, activation_fn='relu')

    model = TD3(policy="MultiInputPolicy", 
                env=vec_env,verbose=0,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=4),
                learning_rate=linear_schedule(0.0003),
                train_freq=1,
                buffer_size=1000000,
                learning_starts=1000,
                batch_size=256,
                tau= 0.005,
                gamma=0.93,
                policy_kwargs=policy_kwargs,
                gradient_steps=-1,
                seed=seed, action_noise=action_noise,
                tensorboard_log='./logs')

    
    # Separate evaluation env
    #log_callback1 = log_callback.CustomCallback()
#    early_stop = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10,min_evals=15, verbose=1)
    eval_env = Monitor(gym.make('gym_fracture:softsurg-v0', **env_kwargs))
    #(make_vec_env(lambda: gym.make('gym_fracture:softsurg-v0', **env_kwargs), n_envs=1))

    eval_callback = EvalCallback(eval_env,  eval_freq=10000, 
                                deterministic=True, n_eval_episodes=20)

    model.learn(3_000_000, callback=eval_callback)
    model_name = f'model-{train_date}-{action_type}-{threshold_pos}-{seed}'
    #save model name in log file
    with open('./logs/model_log.txt', 'w') as f:
        f.write(f'{model_name}\n')
    model.save(f'./models/{model_name}')

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 model with specified thresholds and action type.')
    parser.add_argument('--threshold_pos', type=float, default=0.005, help='Position threshold for the environment.')
    parser.add_argument('--threshold_ori', type=float, default=0.05, help='Orientation threshold for the environment.')
    parser.add_argument('--action_type', type=str, default='fouractions', help='Type of action to use in the environment.')
    parser.add_argument('--seed', type=int, default='42', help='Seed for algorithm')
    args = parser.parse_args()
    train(threshold_pos=args.threshold_pos, threshold_ori=args.threshold_ori, action_type=args.action_type, seed = args.seed)