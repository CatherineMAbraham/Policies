import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
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
import os 
import gc
import shutil
#repo_path = "/home/catherine/FractureGym/fracturesurgeryenv"
repo_paths = ["/users/cop21cma/FracSoftGym/fracturesurgeryenv", "/home/catherine/FractureGym/fracturesurgeryenv",'/home/catherine/FractureSoftGym/fracturesurgeryenv/']


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
          youngs_modulus_type = 'testing',
          randomise_ligs=False,
          randomise_start=False,
          randomise_num_springs=False,
          log=True,
          seed=0):
    render_mode = render_mode
    for repo_path in repo_paths:
        try:
            commit = get_git_commit_hash(repo_path)
            if commit is not None:
                print(f"Git commit hash for repository at {repo_path}: {commit}")
                if repo_path == "/users/cop21cma/FracSoftGym/fracturesurgeryenv":
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
    youngs_modulus = 5e5
    num_springs = 3
    contact_type = contact_type
    eval_seed = 42
    youngs_modulus_name = "None" if youngs_modulus is None else "{:.1E}".format(youngs_modulus)
    if youngs_modulus_type == 'testing':
         name = f'{softtissue}_randomYM_{train_date}_{seed}'
    elif contact_type == 1:
            name = f'{softtissue}_contact_{train_date}_{seed}'
    elif randomise_ligs == 1:
            name = f'{softtissue}_randomligs_{train_date}_{seed}'
    elif randomise_start == 1:
            name = f'{softtissue}_randomstart_{train_date}_{seed}'
    else:
         name = f'{softtissue}-{train_date}-{num_springs}-{youngs_modulus}-{ran}'
    randomise_ligs = True if randomise_ligs == 1 else False
    randomise_start = True if randomise_start == 1 else False
    randomise_num_springs = True if randomise_num_springs == 1 else False
    #print(youngs_modulus)  |
    #print(contact_type)
    #name = f'{softtissue}_{randomise_start}_{randomise_ligs}-{seed}'
    model_name = f'model-{name}'
    if log==1:
        wandb.init(project="Chapter3-Test", name = (name),notes= (f"Git Commit: {commit}"),sync_tensorboard=True, save_code=True)  # Initialize W&B
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
        'patient':110,
        'test': False,
        'youngs_modulus_type': youngs_modulus_type,
        'randomise_ligs':randomise_ligs,
        'randomise_num_springs':randomise_num_springs,
        'randomise_start':randomise_start,
        'render_mode': render_mode}
        #"0.025 -0.04 0" rpy="0 1.57 0"
   
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
      
    env = make_vec_env('gym_fracture:anklesurg-v2', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=DummyVecEnv, seed=seed)
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
                    'softtissue':softtissue,
                    'patient':110,
                    'test': False,
                    'youngs_modulus_type': youngs_modulus_type,
                    'randomise_start':randomise_start,
                    'randomise_ligs':randomise_ligs,
                    'randomise_num_springs': randomise_num_springs,
                    'render_mode': 'direct'}
    
    eval_env=make_vec_env('gym_fracture:anklesurg-v2', n_envs=20, env_kwargs=eval_env_kwargs, vec_env_cls=SubprocVecEnv, seed = eval_seed)
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    log_callback1 = log_callback.CustomCallback()
    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, 
                                                    max_no_improvement_evals=10, 
                                                    success_threshold=0.95,  
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
    model.learn(2_000_000, callback=callback)
    # #save model name in log file
    # with open('./logs/model_log.txt', 'w') as f:
    #     f.write(f'{model_name}\n')
    # model.save(f'./models/{model_name}')
    # model.save_replay_buffer(f'./models/{model_name}-rb')
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
                'youngs_modulus_type': youngs_modulus_type,
                'vtk_file': vtk_file,
                'patient': 110,
                'action_type': 'euler',
                'maxforce': maxforce,
                'contact_type' : contact_type,
                'start_pos' : 'home',
                'render_mode': 'direct',
                'test': True,}
    #ran = 18
    #model_name = 'model-spring_0_testing-7'
    #ran = 1
    #model_name = 'model-spring_randomYM_08161518_1'#'model-spring_contact_08162136_1'
    soft_eval_env = make_vec_env('gym_fracture:anklesurg-v2', n_envs=10, env_kwargs=soft_eval_env_kwargs,vec_env_cls=SubprocVecEnv, seed=eval_seed)
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
    forces = []
    max_forces = []
    success_contact_forces = []
    fail_contact_forces = []
    eps = 0
    all_step_contact_forces = []  # Logs step environment contact force
    all_step_agent_forces = []    # Logs step applied agent force

    # Episode Contact Force summaries split by outcome
    succ_with_contact_forces = []    # Max contact force for Success + Contact
    succ_without_contact_forces = [] # Max contact force for Success + No Contact
    fail_with_contact_forces = []    # Max contact force for Fail + Contact
    fail_without_contact_forces = [] # Max contact force for Fail + No Contact

    # Episode Agent Force summaries split by outcome
    succ_with_contact_agent_forces = []
    succ_without_contact_agent_forces = []
    fail_with_contact_agent_forces = []
    fail_without_contact_agent_forces = []

    # --- PER-ENV BUFFERS (Outside while loop) ---
    env_step_contact_forces = [[] for _ in range(soft_eval_env.num_envs)]
    env_step_agent_forces = [[] for _ in range(soft_eval_env.num_envs)]

    while episodes_collected < num:
        action, _ = eval_model.predict(obs, deterministic=True)
        obs, reward, dones_array, info_list = soft_eval_env.step(action)
        
        # 1. STEP-LEVEL TRACKING (Both Contact & Agent Force)
        for i in range(soft_eval_env.num_envs):
            info = info_list[i]
            
            step_contact_force = info.get('contact_force', 0.0)
            step_agent_force = info.get('force', 0.0)
            
            env_step_contact_forces[i].append(step_contact_force)
            env_step_agent_forces[i].append(step_agent_force)
            
            all_step_contact_forces.append(step_contact_force)
            all_step_agent_forces.append(step_agent_force)
            
            if log == 1:
                wandb.log({
                    'Contact Force at Step': step_contact_force,
                    'Agent Applied Force at Step': step_agent_force
                })

        # 2. EPISODE TERMINATION & CATEGORIZATION
        for i in range(soft_eval_env.num_envs):
            if dones_array[i]:
                info = info_list[i]
                is_success = info.get("is_success", False)
                has_contact = info.get("contact", False)
                
                # Contact Forces
                ep_c_forces = env_step_contact_forces[i]
                ep_max_contact_force = max(ep_c_forces) if ep_c_forces else 0.0
                ep_avg_contact_force = sum(ep_c_forces) / len(ep_c_forces) if ep_c_forces else 0.0

                # Agent Forces
                ep_a_forces = env_step_agent_forces[i]
                ep_max_agent_force = max(ep_a_forces) if ep_a_forces else 0.0
                ep_avg_agent_force = sum(ep_a_forces) / len(ep_a_forces) if ep_a_forces else 0.0
                
                dones.append(is_success)
                contacts.append(has_contact)
                
                # Categorize Max Contact & Agent Forces by Episode Outcome
                if is_success and has_contact:
                    succ_with_contact_forces.append(ep_max_contact_force)
                    succ_with_contact_agent_forces.append(ep_max_agent_force)
                elif is_success and not has_contact:
                    succ_without_contact_forces.append(ep_max_contact_force)
                    succ_without_contact_agent_forces.append(ep_max_agent_force)
                elif not is_success and has_contact:
                    fail_with_contact_forces.append(ep_max_contact_force)
                    fail_with_contact_agent_forces.append(ep_max_agent_force)
                else:
                    fail_without_contact_forces.append(ep_max_contact_force)
                    fail_without_contact_agent_forces.append(ep_max_agent_force)

                episodes_collected += 1
                print(f"[{episodes_collected}/{num}] Env {i} | Success: {is_success} | Contact: {has_contact} | "
                    f"Max Contact Force: {ep_max_contact_force:.2f}N | Max Agent Force: {ep_max_agent_force:.2f}N")

                # Reset local step buffers for this environment
                env_step_contact_forces[i] = []
                env_step_agent_forces[i] = []

                # 3. WANDB LOGGING & SUMMARY UPDATES
                if log == 1:
                    wandb.log({
                        "Episode": episodes_collected,
                        "Success": is_success,
                        "Contact": has_contact,
                        "Episode Max Contact Force": ep_max_contact_force,
                        "Episode Avg Contact Force": ep_avg_contact_force,
                        "Episode Max Agent Force": ep_max_agent_force,
                        "Episode Avg Agent Force": ep_avg_agent_force,
                        "Success Rate": sum(dones) / len(dones)
                    })

                    # Counts
                    wandb.run.summary['Count / Success With Contact'] = len(succ_with_contact_forces)
                    wandb.run.summary['Count / Success Without Contact'] = len(succ_without_contact_forces)
                    wandb.run.summary['Count / Fail With Contact'] = len(fail_with_contact_forces)
                    wandb.run.summary['Count / Fail Without Contact'] = len(fail_without_contact_forces)
                    
                    # Global Mean Step Forces across all runs
                    wandb.run.summary['Force / Overall Mean Contact Force'] = sum(all_step_contact_forces) / len(all_step_contact_forces) if all_step_contact_forces else 0.0
                    wandb.run.summary['Force / Overall Mean Agent Force'] = sum(all_step_agent_forces) / len(all_step_agent_forces) if all_step_agent_forces else 0.0
                    
                    # Contact Force Summaries (Avg Max)
                    if succ_with_contact_forces:
                        wandb.run.summary['Contact Force / Success With Contact (Avg Max)'] = sum(succ_with_contact_forces) / len(succ_with_contact_forces)
                    if succ_without_contact_forces:
                        wandb.run.summary['Contact Force / Success Without Contact (Avg Max)'] = sum(succ_without_contact_forces) / len(succ_without_contact_forces)
                    if fail_with_contact_forces:
                        wandb.run.summary['Contact Force / Fail With Contact (Avg Max)'] = sum(fail_with_contact_forces) / len(fail_with_contact_forces)
                    if fail_without_contact_forces:
                        wandb.run.summary['Contact Force / Fail Without Contact (Avg Max)'] = sum(fail_without_contact_forces) / len(fail_without_contact_forces)

                    # Agent Force Summaries (Avg Max)
                    if succ_with_contact_agent_forces:
                        wandb.run.summary['Agent Force / Success With Contact (Avg Max)'] = sum(succ_with_contact_agent_forces) / len(succ_with_contact_agent_forces)
                    if succ_without_contact_agent_forces:
                        wandb.run.summary['Agent Force / Success Without Contact (Avg Max)'] = sum(succ_without_contact_agent_forces) / len(succ_without_contact_agent_forces)
                    if fail_with_contact_agent_forces:
                        wandb.run.summary['Agent Force / Fail With Contact (Avg Max)'] = sum(fail_with_contact_agent_forces) / len(fail_with_contact_agent_forces)
                    if fail_without_contact_agent_forces:
                        wandb.run.summary['Agent Force / Fail Without Contact (Avg Max)'] = sum(fail_without_contact_agent_forces) / len(fail_without_contact_agent_forces)

                if episodes_collected >= num:
                    break
            
    
    print("\nEvaluation complete. Cleaning up resources to save memory...")

    # 1. Close the evaluation environments to free up system/subprocess RAM
    soft_eval_env.close()

    # 2. Delete model and environment variables from Python memory, then force GC
    # del eval_model
    # del soft_eval_env
    # gc.collect()

    # # 3. Delete the physical model files from your disk to free up storage
    # model_folder_path = f'./best_models/{ran}/{model_name}'
    # if os.path.exists(model_folder_path):
    #     try:
    #         shutil.rmtree(model_folder_path)
    #         print(f"Successfully deleted model directory: {model_folder_path}")
    #     except Exception as e:
    #         print(f"Error while deleting directory {model_folder_path}: {e}")



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
    parser.add_argument('--youngs_modulus_type', type=str, default='eval_mode', help='Type of Young\'s modulus for the soft tissue.')
    parser.add_argument('--randomise_ligs', type=int, default=0, help='Whether to randomise ligaments for the environment.')
    parser.add_argument('--randomise_start', type=int, default=0, help='Whether to randomise the starting position for the environment.')
    parser.add_argument('--randomise_num_springs', type=int, default=0, help='Whether to randomise the number of springs for the environment.')
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
          youngs_modulus_type=args.youngs_modulus_type,
          randomise_ligs=args.randomise_ligs,
          randomise_start=args.randomise_start,
          seed=args.seed)
