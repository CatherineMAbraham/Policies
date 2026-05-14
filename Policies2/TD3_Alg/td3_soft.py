import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import wandb
import numpy as np
from typing import Callable
import datetime
from git import Repo, InvalidGitRepositoryError
import argparse
import log_callback
from success_callback import StopTrainingOnSuccessRate
#repo_path = "/home/catherine/FractureGym/fracturesurgeryenv"
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
          log=True):

    commit = get_git_commit_hash(repo_path)
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type# 'fouractions'#'pos_only' #action_type
    threshold_pos = threshold_pos
    threshold_ori = np.deg2rad(threshold_ori)
    maxforce = maxforce
    softtissue = softtissue
    num_springs = num_springs
    contact_type = contact_type
    model_name = f'model-{train_date}-{action_type}-{threshold_pos}-{threshold_ori}-{ran}'
    if log:
        wandb.init(project="Tissue", name = (f'{softtissue}-{train_date}'),notes= (f"Git Commit: {commit}"),sync_tensorboard=True, save_code=True)  # Initialize W&B

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
        'test': True,
        'render_mode': None}
        #"0.025 -0.04 0" rpy="0 1.57 0"
    
    env = make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]),
                                              sigma=0.02 * np.ones(env.action_space.shape[0]))

    policy_kwargs = dict(net_arch=[256, 256,256])#, activation_fn='relu')

    model = TD3(policy="MultiInputPolicy",
                env=env,verbose=0,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=4),
                learning_rate=linear_schedule(0.0003),
                train_freq=1,
                buffer_size=1000000,
                learning_starts=500,
                batch_size=256,
                tau= 0.005,
                gamma=0.93,
                policy_kwargs=policy_kwargs,
                gradient_steps=-1,
                seed=42, action_noise=action_noise,tensorboard_log=f'./logs/{ran}')


   
    eval_env=make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=SubprocVecEnv)
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    log_callback1 = log_callback.CustomCallback()
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, 
                                                 max_no_improvement_evals=10, 
                                                 success_threshold=0.4,  
                                                 min_evals=1, verbose=1, 
                                                 model_name = model_name,
                                                 model_save_path=f'./best_models/{ran}')
    eval_callback = EvalCallback(eval_env,  eval_freq=10000,
                                deterministic=True, n_eval_episodes=20,
                                callback_after_eval=success_callback)

    model.learn(100_000, callback=[eval_callback,log_callback1])
    #save model name in log file
    with open('./logs/model_log.txt', 'w') as f:
        f.write(f'{model_name}\n')
    model.save(f'./models/{model_name}')
    model.save_replay_buffer(f'./models/{model_name}-rb')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 model with specified thresholds and action type.')
    parser.add_argument('--threshold_pos', type=float, default=0.005, help='Position threshold for the environment.')
    parser.add_argument('--threshold_ori', type=float, default=0.05, help='Orientation threshold for the environment.')
    parser.add_argument('--action_type', type=str, default='euler', help='Type of action to use in the environment.')
    parser.add_argument('--render_mode', type=str, default="human", help='Render mode for the environment.')
    parser.add_argument('--maxforce', type=float, default=4, help='Force threshold for the environment.')
    parser.add_argument('--softtissue', type=str, default="spring", help='Soft Tissue Type.')
    parser.add_argument('--num_springs', type=int, default=3, help='Number of springs for the soft tissue.')
    parser.add_argument('--contact_type', type=str, default="None", help='Type of contact for the environment.')
    parser.add_argument('--ran', type=str, default="1", help='Random seed for the run.')
    parser.add_argument('--log', type=bool, default=True, help='Whether to log the training run to W&B.')
    args = parser.parse_args()
    train(threshold_pos=args.threshold_pos, 
          threshold_ori=args.threshold_ori, 
          action_type=args.action_type, 
          render_mode=args.render_mode,
          maxforce=args.maxforce, 
          num_springs=args.num_springs,
          contact_type=args.contact_type,
          softtissue=args.softtissue, 
          ran=args.ran,
          log=args.log)
