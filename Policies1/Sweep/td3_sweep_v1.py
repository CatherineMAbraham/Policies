import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import wandb
import numpy as np
from typing import Callable
import datetime
from git import Repo, InvalidGitRepositoryError
import argparse
from success_callback import StopTrainingOnSuccessRate
#repo_path = "/home/catherine/FractureGym/fracturesurgeryenv"
repo_paths = ["/users/cop21cma/FracSoftGym/", "/home/catherine/FractureGym/",'/home/catherine/FractureSoftGym/']
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


def train(threshold_pos=0.001, 
          threshold_ori=np.deg2rad(6), 
          action_type='euler', 
          render_mode='human',
          maxforce=4, 
          softtissue='spring',
          num_springs=3,
          contact_type="None",
          ran='1',
          seed=1,
          youngs_modulus=1e6,
          log=True):

    for repo_path in repo_paths:
        try:
            commit = get_git_commit_hash(repo_path)
            if commit is not None:
                print(f"Git commit hash for repository at {repo_path}: {commit}")
                break
        except Exception as e: print(f"Could not get commit hash for repository at {repo_path}: {e}")
    
    wandb.init(project="Chapter2-Sweep", sync_tensorboard=True, save_code=True)  # Initialize W&B
    config = wandb.config
    
    action_type = 'euler'# 'fouractions'#'pos_only' #action_type
    threshold_pos = 0.0005
    threshold_ori = np.deg2rad(0.5)
    maxforce = 4
    softtissue = 'spring'
    num_springs = 3
    contact_type = 0
    youngs_modulus = 1e7
    eval_seed = 42
    
    learning_rate = config.learning_rate
    gamma = config.gamma
    tau = config.tau
    batch_size = config.batch_size
    train_freq = config.train_freq
    net_arch = config.net_arch
    learning_starts = config.learning_starts
    her_sampled_goal = config.her_sampled_goal 
    action_noise = config.action_noise

    env_kwargs = {
        'reward_type': 'sparse',
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'dt': 0.001,
        'dr':0.01,
        'distance_threshold_ori': threshold_ori,
        'action_type': action_type,
        'start_pos' : 'home',
        'maxforce': maxforce,
        'contact_type' :contact_type,
        'number_of_springs':num_springs,
        'softtissue':softtissue,
        'test': False,
        'youngs_modulus': youngs_modulus,
        'render_mode': None}
        #"0.025 -0.04 0" rpy="0 1.57 0"
   
    env = make_vec_env('gym_fracture:anklesurg-v1', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=DummyVecEnv, seed= seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    if action_noise == "normal":
        action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), 
                                              sigma=0.1 * np.ones(env.action_space.shape[0]))
    elif action_noise == "OU":
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]), 
                                              sigma=0.02 * np.ones(env.action_space.shape[0]))

    policy_kwargs = dict(net_arch=net_arch)#, activation_fn='relu')

    model = TD3(policy="MultiInputPolicy",
                env=env,verbose=0,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=her_sampled_goal),
                learning_rate=linear_schedule(learning_rate),
                train_freq=train_freq,
                buffer_size=1_000_000,
                learning_starts= learning_starts,
                batch_size=batch_size,
                tau= tau,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                gradient_steps=-1,
                seed=42, action_noise=action_noise,tensorboard_log='./logs/{ran}')


    eval_env_kwargs = {
            'reward_type': 'sparse',
            'max_steps': 100,
            'horizon': 'variable',
            'obs_type': 'dict',
            'distance_threshold_pos': threshold_pos,
            'dt': 0.001,
            'dr':0.01,
            'distance_threshold_ori': threshold_ori,
            'action_type': action_type,
            'start_pos' : 'home',
            'render_mode': None,
            'maxforce': maxforce,
            'contact_type' :contact_type,
            'number_of_springs':num_springs,
            'softtissue':softtissue,
            'test': False,
            'youngs_modulus': youngs_modulus,
            'render_mode': None}
    eval_env=make_vec_env('gym_fracture:anklesurg-v1', env_kwargs=eval_env_kwargs,vec_env_cls=SubprocVecEnv, seed = eval_seed)
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, max_no_improvement_evals=1,
                                                                success_threshold=1)
    
    eval_callback = EvalCallback(eval_env,  eval_freq=10000,
                                 deterministic=True, n_eval_episodes=20,callback_after_eval=success_callback)
                                
    model.learn(500_000, callback=eval_callback)
    #save model name in log file
    



if __name__ == "__main__":
    sweep_id = "1bayndf7"
    wandb.agent(sweep_id, function=train, count=4)