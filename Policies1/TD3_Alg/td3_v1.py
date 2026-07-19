import os

import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import wandb
import numpy as np
from typing import Callable
import datetime
from git import Repo, InvalidGitRepositoryError
import argparse
import log_callback
from success_callback import StopTrainingOnSuccessRate
import shutil
import gc
#from env_test2 import multiple_envs 
#repo_path = "/home/catherine/FractureGym/fracturesurgeryenv"
repo_paths = ["/users/cop21cma/FracSoftGym/", "/home/catherine/FractureGym/",'/home/catherine/FractureSoftGym/']


def int_or_none(value: str):
    """argparse type: parse an int or the literal 'None'."""
    if value is None:
        return None
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "youngs_modulus must be an integer or 'None'"
        ) from exc


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

def get_youngs_modulus():
    young_modulus_options = [1e6 ,1e7,5e6, 1e8]
    ## Select a youngs modulus for the eval, making sure to use a different one each time 
    youngs_modulus = np.random.choice(young_modulus_options)
    print(f"Selected Young's Modulus for evaluation: {youngs_modulus}")
    return youngs_modulus
def get_width():
    width_options = np.arange(0.001, 0.01, 0.001)
    width = np.random.choice(width_options)
    print(f"Selected width for evaluation: {width}")
    return width

def train(threshold_pos=0.001, 
          threshold_ori=np.deg2rad(6), 
          action_type='euler', 
          render_mode='human',
          maxforce=4, 
          softtissue='spring',
          num_springs=3,
          contact_type="None",
          ran='1',
          youngs_modulus=1000000,
          vtkfile='rect0009.vtk',
          log=True,
          seed=0):
    render_mode = render_mode
    for repo_path in repo_paths:
        try:
            commit = get_git_commit_hash(repo_path)
            if commit is not None:
                print(f"Git commit hash for repository at {repo_path}: {commit}")
                if repo_path == "/users/cop21cma/FracSoftGym/":
                    render_mode = None
                    log =1 
                break
        except Exception as e: print(f"Could not get commit hash for repository at {repo_path}: {e}")
        
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type# 'fouractions'#'pos_only' #action_type
    threshold_pos = threshold_pos
    threshold_ori = np.deg2rad(threshold_ori)
    maxforce = maxforce
    softtissue = softtissue
    num_springs = num_springs
    contact_type = contact_type
    eval_seed = 42
    youngs_modulus_name = "None" if youngs_modulus is None else "{:.1E}".format(youngs_modulus)
    #print(youngs_modulus)
    #print(contact_type)
    name = f'{softtissue}_{num_springs}_{youngs_modulus_name}_{seed}_{train_date}'
    model_name = f'model-{name}'
    if log==1:
        wandb.init(project="Chapter2-Results", name = (name),notes= (f"Git Commit: {commit}"),sync_tensorboard=True, save_code=True)  # Initialize W&B
    #print((f'{softtissue}-{train_date}-{num_springs}-{youngs_modulus}-{ran}'))
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
        'vtk_file': vtkfile,
        'youngs_modulus': youngs_modulus,
        'test': True,
        'render_mode': render_mode,}
    
    td3_kwargs = {"tau": 0.1,
                "gamma": 0.9,
                "batch_size":  128,
                "train_freq":  2,
                "buffer_size": 500_000,
                "learning_rate": linear_schedule(0.001),
                "learning_starts":2000,
                "gradient_steps": -1,
                "policy": "MultiInputPolicy",
                "replay_buffer_class": HerReplayBuffer,
                "replay_buffer_kwargs": dict(n_sampled_goal=8,goal_selection_strategy='future'),
                "policy_kwargs": dict(net_arch=[400, 300]),
                "tensorboard_log": f'./logs/{ran}',
                "seed": seed}
   
    env = make_vec_env('gym_fracture:anklesurg-v1', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=DummyVecEnv, seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))


    model = TD3(**td3_kwargs, env=env, action_noise=action_noise)


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
            'maxforce': maxforce,
            'contact_type' :contact_type,
            'number_of_springs':num_springs,
            'youngs_modulus': youngs_modulus,
            'softtissue':softtissue,
            'test': False,
            'render_mode': 'direct'}
   
    eval_env=make_vec_env('gym_fracture:anklesurg-v1', env_kwargs=eval_env_kwargs,vec_env_cls=SubprocVecEnv, seed = eval_seed)
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    log_callback1 = log_callback.CustomCallback()
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, 
                                                    max_no_improvement_evals=1, 
                                                    success_threshold=1,  
                                                    min_evals=1, verbose=1, 
                                                    model_name = model_name,
                                                    model_save_path=f'./best_models/{ran}')
    eval_callback = EvalCallback(eval_env,  eval_freq=10000,
                                deterministic=True, n_eval_episodes=50,
                                callback_after_eval=success_callback)
    if log == 1:
        callback = [eval_callback, log_callback1]
    else:
        callback = [eval_callback]
    model.learn(1_500_000, callback=callback)
    #save model name in log file
    with open('./logs/model_log.txt', 'w') as f:
        f.write(f'{model_name}\n')
    #model.save(f'./models/{model_name}')
    #model.save_replay_buffer(f'./models/{model_name}-rb')
   
    ## Evalulate the model using the FEM model 
    
    vtk_file = 'rect0009.vtk'
    soft_eval_env_kwargs = {
                'reward_type': 'sparse',
                'max_steps': 100,
                'horizon': 'variable',
                'obs_type': 'dict',
                'distance_threshold_pos': threshold_pos,
                'dt': 0.001,
                'dr':0.01,
                'distance_threshold_ori': threshold_ori,
                'softtissue': 'soft',
                'number_of_springs': num_springs,
                'youngs_modulus': 1.5e6,
                'vtk_file': vtk_file,
                'action_type': 'euler',
                'maxforce': maxforce,
                'contact_type' : 0,
                'start_pos' : 'home',
                'render_mode': 'direct',
                'test': True,}
    soft_eval_env = make_vec_env('gym_fracture:anklesurg-v1', n_envs=20, env_kwargs=soft_eval_env_kwargs,vec_env_cls=SubprocVecEnv, seed=eval_seed)
    stats_path = f'./best_models/{ran}/{model_name}/vec_normalize.pkl'
    soft_eval_env = VecNormalize.load(stats_path, soft_eval_env)

    #soft_eval_env.obs_rms = env.obs_rms  # Direct reference copy of the running means
    soft_eval_env.training = False       # FREEZE STATS: Essential so eval steps don't corrupt them
    soft_eval_env.norm_reward = False

    # 4. Create an identical, blank TD3 architecture hooked up to the new environment
    #eval_model = TD3(**td3_kwargs, env=soft_eval_env, action_noise=action_noise)
    model_path = f'./best_models/{ran}/{model_name}/{model_name}'
    eval_model = TD3.load(model_path, env=soft_eval_env)#, action_noise=action_noise)



    dones = []
    contacts = []
    num = 1000
    episodes_collected = 0
    obs = soft_eval_env.reset()
    max_forces = []
    
    eps = 0
    while episodes_collected < num:
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, dones_array, info_list = soft_eval_env.step(action)
            
            for i in range(soft_eval_env.num_envs):
                    if dones_array[i]:
                            info = info_list[i]
                            
                            # 1. Get the actual final observation (before the auto-reset)
                            # This is critical if you want to calculate metrics manually
                            final_obs = info.get("terminal_observation")
                            
                            # 2. Get the success flag provided by the environment/Monitor
                            is_success = info.get("is_success", False)
                            
                            # 3. Get your custom 'contact' metric
                            # Note: Ensure your env puts 'contact' in the info dict even on the final step!
                            has_contact = info.get("contact", False)
                            
                            dones.append(is_success)
                            contacts.append(has_contact)
                            
                            episodes_collected += 1
                            print(f"[{episodes_collected}/{num}] Env {i} Success: {is_success} Force: {info.get('force')} Pos: {info.get('pos_distance')} Angle: {info.get('angle')} Contact: {has_contact}, Success Rate: {sum(dones) / len(dones)}")
                            ## If force >50 do not log to wandb as it is an outlier and can skew the results
                            # remove number of episodes collected from the success rate calculation in the log as well
                            if info.get('force', 0) <= 50:
                                    eps +=1
                            if log==1 :
                                #table = wandb.Table(data = is_success,columns=["Episode", "Success"])
                                #histogram = wandb.plot.Histogram(table,value='Success', title="Success Distribution")
                                wandb.log({"Episode": episodes_collected,  "Contact": has_contact, "force": info.get('force', 0), "Position Distance": info.get('pos_distance', 0), "Angle Distance": info.get('angle', 0), "Success": is_success, "Success Rate": sum(dones) / len(dones)})
                                if info.get('force', 0) <= 50:      
                                    wandb.run.summary["final_success_rate"] = sum(dones) / eps
                                    if info.get('force', 0) <= maxforce:
                                        max_force = info.get('force', 0)
                                        max_forces.append(max_force)
                                        wandb.run.summary["max_force"] = max_force
                                        wandb.run.summary["Average baselines Force"] = sum(max_forces) / len(max_forces) ## want to see what the average max force is 
                                        wandb.run.summary['Fail With Contact'] = sum(1 for d, c in zip(dones, contacts) if not d and c)
                                        wandb.run.summary['Fail Without Contact'] = sum(1 for d, c in zip(dones, contacts) if not d and not c)
                                        wandb.run.summary['Success With Contact'] = sum(1 for d, c in zip(dones, contacts) if d and c)
                                        wandb.run.summary['Success Without Contact'] = sum(1 for d, c in zip(dones, contacts) if d and not c)
                            if episodes_collected >= num:
                                    break
    
    print("\nEvaluation complete. Cleaning up resources to save memory...")

    # 1. Close the evaluation environments to free up system/subprocess RAM
    soft_eval_env.close()

    # 2. Delete model and environment variables from Python memory, then force GC
    del eval_model
    del soft_eval_env
    gc.collect()

    # 3. Delete the physical model files from your disk to free up storage
    model_folder_path = f'./best_models/{ran}/{model_name}'
    if os.path.exists(model_folder_path):
        try:
            shutil.rmtree(model_folder_path)
            print(f"Successfully deleted model directory: {model_folder_path}")
        except Exception as e:
            print(f"Error while deleting directory {model_folder_path}: {e}")

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
    parser.add_argument('--youngs_modulus', type=float, default=1e7, help='Young\'s modulus for the soft tissue. Use an integer or None')
    parser.add_argument('--vtkfile', type=str, default='rect0009.vtk', help='VTK file for the soft tissue model.')
    parser.add_argument('--ran', type=str, default="1", help='Random seed for the run.')
    parser.add_argument('--log', type=int, default=0, help='Whether to log the training run to W&B.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
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
          log=args.log,
          youngs_modulus=args.youngs_modulus,
          vtkfile=args.vtkfile,
          seed=args.seed)
