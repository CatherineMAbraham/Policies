import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common import vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import log_callback
from success_callback import StopTrainingOnSuccessRate
import os
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


def train(threshold_pos=0.001, 
          threshold_ori=np.deg2rad(6), 
          action_type='euler', 
          render_mode='human',
          maxforce=4, 
          softtissue='spring',
          num_springs=3,
          contact_type="None",
          ran='1',
          youngs_modulus=1e6,
          youngs_modulus_type="None",
          model = 'model',
          log=1,
          seed=42):
    commit = get_git_commit_hash(repo_path)
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type
    threshold_pos = threshold_pos
    threshold_ori = np.deg2rad(threshold_ori)
   
    #model_path = os.path.join(os.getcwd(), model)
    model_name = f'model-{train_date}-{softtissue}-{maxforce}'
    print(f'Training with model {model}')
    if log ==1:
        wandb.init(project="ForceCL", name = model_name,notes= (f"Git Commit: {commit}, model {model}"),sync_tensorboard=True, save_code=True)  # Initialize W&B
    #print(f'Training with model {model}')

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
        'youngs_modulus_type': youngs_modulus_type,
        'render_mode': None}
    model_path = model

    #vec_env=make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=SubprocVecEnv)
    env = make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=DummyVecEnv, seed=seed)
    #env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecNormalize.load(f"{model_path}/vec_normalize.pkl", env) # Register the environment
    env.training = True
    env.norm_reward = False
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]),
    #                                           sigma=0.02 * np.ones(env.action_space.shape[0]))

    policy_kwargs = dict(net_arch=[256, 256,256])#, activation_fn='relu')
    model_dir = Path(model_path)
    ##find file that starts with model but does not end in rb 
    model_candidates = sorted(
            [p for p in model_dir.glob("model*") if p.is_file()],# and not p.name.endswith('-rb.zip')],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
    )
    if not model_candidates:
            raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")
    # model that doesnt end in rb 
    for candidate in model_candidates:
        if not candidate.name.endswith('-rb.zip'):
            selected_model = candidate
            

        elif candidate.name.endswith('-rb.zip'):
            replay_buffer_model = candidate
            
    print(f"Selected model: {selected_model}")
    #print(f"Selected replay buffer model: {replay_buffer_model}")
    model = TD3.load(str(selected_model), env=env)
    model.load_replay_buffer(replay_buffer_model)
    print(f"Loaded replay buffer from {replay_buffer_model}")
    # if os.path.exists(f'{model_path}-rb.zip'):
    #     model.load_replay_buffer(f'{model_path}_replay')
    #     print(f"Loaded replay buffer from {model_path}_replay")
    #model.seed(seed_value)
    model.seed= 42
  
    
    # Separate evaluation env
    eval_env=make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs,vec_env_cls=SubprocVecEnv, seed= seed)
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    log_callback1 = log_callback.CustomCallback()
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, 
                                                    max_no_improvement_evals=5, 
                                                    success_threshold=1,  
                                                    min_evals=1, verbose=1, 
                                                    model_name = model_name,
                                                    model_save_path=f'./best_models/{ran}')
    eval_callback = EvalCallback(eval_env,  eval_freq=10000,
                                deterministic=True, n_eval_episodes=50,
                                callback_after_eval=success_callback)

    model.learn(2_500_000, callback=[eval_callback,log_callback1])
    model.save(f'./{model_name}')
    model.save_replay_buffer(f'./{model_name}_replay')
    #save model name in log file
    with open(f'./logs/model_log{42}.txt', 'w') as f:
        f.write(f'{model_name}\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 model with specified thresholds and action type.')
    parser.add_argument('--threshold_pos', type=float, default=0.005, help='Position threshold for the environment.')
    parser.add_argument('--threshold_ori', type=float, default=0.05, help='Orientation threshold for the environment.')
    parser.add_argument('--action_type', type=str, default='euler', help='Type of action to use in the environment.')
    parser.add_argument('--render_mode', type=str, default="human", help='Render mode for the environment.')
    parser.add_argument('--maxforce', type=float, default=4, help='Force threshold for the environment.')
    parser.add_argument('--softtissue', type=str, default="spring", help='Soft Tissue Type.')
    parser.add_argument('--num_springs', type=int, default=3, help='Number of springs for the soft tissue.')
    parser.add_argument('--contact_type', type=int, default=0, help='Type of contact for the environment.')
    parser.add_argument('--youngs_modulus', type=float, default=1e6, help='Young\'s modulus for the soft tissue.')
    parser.add_argument('--youngs_modulus_type', type=str, default="None", help='Type of Young\'s modulus for the soft tissue.')
    parser.add_argument('--ran', type=str, default="1", help='Random seed for the run.')
    parser.add_argument('--log', type=int, default=1, help='Whether to log the training run to W&B.')
    parser.add_argument('--model', type=str, default="model", help='Model name.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    train(threshold_pos=args.threshold_pos, 
          threshold_ori=args.threshold_ori, 
          action_type=args.action_type, 
          render_mode=args.render_mode,
          maxforce=args.maxforce, 
          num_springs=args.num_springs,
          contact_type=args.contact_type,
          softtissue=args.softtissue, 
          model=args.model,
          ran=args.ran,
          log=args.log,
          youngs_modulus=args.youngs_modulus,
          youngs_modulus_type=args.youngs_modulus_type,
          seed=args.seed)


