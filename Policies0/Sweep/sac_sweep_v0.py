import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from success_callback import StopTrainingOnSuccessRate
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


def train(threshold_pos=0.001, threshold_ori=np.deg2rad(6), action_type='pos_only',seed=42,ran="1"):
    commit = get_git_commit_hash(repo_path)
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type
    threshold_pos = 0.0005
    threshold_ori = np.deg2rad(0.5)
    wandb.init(project="Chp1-Sweep-Sac", sync_tensorboard=True, save_code=True)  # Initialize W&B
    config = wandb.config
    
    
    
    learning_rate = config.learning_rate
    gamma = config.gamma
    tau = config.tau
    batch_size = config.batch_size
    train_freq = config.train_freq
    net_arch = config.net_arch
    learning_starts = config.learning_starts
    her_sampled_goal = config.her_sampled_goal 
    ent_coef = config.ent_coef
    use_sde = config.use_sde
    sde_sample_freq = config.sde_sample_freq
    if use_sde == False:
        sde_sample_freq=None
    env_kwargs = {
        'reward_type': 'sparse',
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'distance_threshold_ori' : threshold_ori,
        'dv': 0.001,
        'softtissue': False,
        'action_type': action_type,
        'start_pos' : 'home',
        'render_mode':'direct'}
        
    
    #vec_env=make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=SubprocVecEnv)
    vec_env = make_vec_env('gym_fracture:anklesurg-v0', env_kwargs=env_kwargs, n_envs=1, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(vec_env.action_space.shape[0]), 
                                              sigma=0.02 * np.ones(vec_env.action_space.shape[0]))

    policy_kwargs = dict(net_arch=net_arch)#, activation_fn='relu')

    model = SAC(policy="MultiInputPolicy", 
                env=vec_env,verbose=0,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=her_sampled_goal),
                learning_rate=linear_schedule(learning_rate),
                train_freq=train_freq,
                buffer_size=1000000,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau= tau,
                gamma=gamma,
                ent_coef=ent_coef,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                policy_kwargs=policy_kwargs,
                gradient_steps=-1,
                seed=seed, action_noise=action_noise,
                tensorboard_log=f'./logs/{ran}')

    
    # Separate evaluation env
    #log_callback1 = log_callback.CustomCallback()
#    early_stop = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10,min_evals=15, verbose=1)
    eval_env = make_vec_env('gym_fracture:anklesurg-v0', env_kwargs=env_kwargs, n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    #(make_vec_env(lambda: gym.make('gym_fracture:softsurg-v0', **env_kwargs), n_envs=1))
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, max_no_improvement_evals=1,
                                                                success_threshold=1)
    eval_callback = EvalCallback(eval_env,  eval_freq=10000, 
                                deterministic=True, n_eval_episodes=20, callback_after_eval=success_callback)

    model.learn(500_000, callback=eval_callback)
    #model_name = f'model-{train_date}-{action_type}-{threshold_pos}-{seed}'
    #save model name in log file
    # with open('./logs/model_log.txt', 'w') as f:
    #     f.write(f'{model_name}\n')
    # model.save(f'./models/{model_name}')

            


if __name__ == "__main__":
    sweep_id = "idyxesfp"
    wandb.agent(sweep_id, function=train, count=10)