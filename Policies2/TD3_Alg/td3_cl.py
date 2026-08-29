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

repo_paths = [
    "/users/cop21cma/FracSoftGym/fracturesurgeryenv", 
    "/home/catherine/FractureGym/fracturesurgeryenv",
    "/home/catherine/FractureSoftGym/fracturesurgeryenv/"
]

def int_or_none(value: str):
    if value is None or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("youngs_modulus must be an integer or 'None'") from exc

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
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def run_threshold_evaluation(model, eval_env, n_episodes=30):
    """Evaluates the model over n_episodes and records success rate and peak contact forces."""
    successes = []
    peak_contact_forces = []
    
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_forces = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            done = dones[0]
            info = infos[0]
            ep_forces.append(info.get('contact_force', 0.0))
            if done:
                successes.append(info.get('is_success', False))
                peak_contact_forces.append(max(ep_forces) if ep_forces else 0.0)
                
    avg_success = sum(successes) / len(successes) if successes else 0.0
    mean_peak_force = sum(peak_contact_forces) / len(peak_contact_forces) if peak_contact_forces else 0.0
    max_peak_force = max(peak_contact_forces) if peak_contact_forces else 0.0
    
    return avg_success, mean_peak_force, max_peak_force


def train(threshold_pos=0.001, 
          threshold_ori=np.deg2rad(6), 
          max_contact_force_threshold=0.5,
          action_type='euler', 
          render_mode='human',
          maxforce=4, 
          softtissue='spring',
          num_springs=3,
          contact_type="None",
          ran='1',
          youngs_modulus=1000000,
          youngs_modulus_type='testing',
          randomise_ligs=False,
          randomise_start=False,
          randomise_num_springs=False,
          log=True,
          seed=0,
          # --- Iterative Force Tuning Arguments ---
          run_iterative_search=True,
          decay_factor=0.8,
          target_success_rate=0.85,
          max_tuning_iters=4,
          fine_tune_timesteps=50_000):
    
    render_mode = render_mode
    commit = None
    for repo_path in repo_paths:
        try:
            commit = get_git_commit_hash(repo_path)
            if commit is not None:
                print(f"Git commit hash for repository at {repo_path}: {commit}")
                if repo_path == "/users/cop21cma/FracSoftGym/fracturesurgeryenv":
                    render_mode = None
                    log = 1 
                break
        except Exception as e: 
            print(f"Could not get commit hash for repository at {repo_path}: {e}")
        
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    threshold_pos = threshold_pos
    threshold_ori = np.deg2rad(threshold_ori)
    maxforce = maxforce
    softtissue = softtissue
    num_springs = 3
    contact_type = contact_type
    eval_seed = 42

    if youngs_modulus_type == 'testing':
        name = f'{softtissue}_randomYM_{train_date}_{seed}'
    elif contact_type == 1:
        name = f'{softtissue}_contact_{threshold_pos}_{train_date}_{seed}'
    elif randomise_ligs == 1:
        name = f'{softtissue}_randomligs_{train_date}_{seed}'
    elif randomise_start == 1:
        name = f'{softtissue}_randomstart_{train_date}_{seed}'
    else:
        name = f'{softtissue}-{train_date}-{num_springs}-{youngs_modulus}-{ran}'

    randomise_ligs = True if randomise_ligs == 1 else False
    randomise_start = True if randomise_start == 1 else False
    randomise_num_springs = True if randomise_num_springs == 1 else False

    tags = [f"{contact_type}", f"{max_contact_force_threshold}", "baseline"]
    model_name = f'model-{name}'
    
    if log == 1:
        wandb.init(project="Chapter3-Results-2", tags=tags, name=name, notes=f"Git Commit: {commit}", sync_tensorboard=True, save_code=True)

    current_force_threshold = max_contact_force_threshold

    env_kwargs = {
        'reward_type': 'sparse',
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'dt': 0.001,
        'dr': 0.01,
        'distance_threshold_ori': threshold_ori,
        'action_type': action_type,
        'start_pos': 'home',
        'maxforce': maxforce,
        'contact_type': contact_type,
        'number_of_springs': num_springs,
        'softtissue': softtissue,
        'patient': 110,
        'test': False,
        'maximum_contact_force_threshold': current_force_threshold,
        'youngs_modulus_type': youngs_modulus_type,
        'randomise_ligs': randomise_ligs,
        'randomise_num_springs': randomise_num_springs,
        'randomise_start': randomise_start,
        'render_mode': render_mode
    }

    td3_kwargs = {
        "tau": 0.1,
        "gamma": 0.9,
        "batch_size": 256,
        "train_freq": 1,
        "buffer_size": 500_000,
        "learning_rate": linear_schedule(0.001),
        "learning_starts": 5000,
        "gradient_steps": -1,
        "policy": "MultiInputPolicy",
        "replay_buffer_class": HerReplayBuffer,
        "replay_buffer_kwargs": dict(n_sampled_goal=4, goal_selection_strategy='future'),
        "policy_kwargs": dict(net_arch=[256, 256, 256]),
        "tensorboard_log": f'./logs/{ran}',
        "seed": seed
    }
      
    env = make_vec_env('gym_fracture:anklesurg-v2', env_kwargs=env_kwargs, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
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
        'dr': 0.01,
        'distance_threshold_ori': threshold_ori,
        'action_type': action_type,
        'start_pos': 'home',
        'maxforce': maxforce,
        'contact_type': contact_type,
        'number_of_springs': num_springs,
        'softtissue': softtissue,
        'maximum_contact_force_threshold': current_force_threshold,
        'patient': 110,
        'test': False,
        'youngs_modulus_type': youngs_modulus_type,
        'randomise_start': randomise_start,
        'randomise_ligs': randomise_ligs,
        'randomise_num_springs': randomise_num_springs,
        'render_mode': 'direct'
    }
    
    eval_env = make_vec_env('gym_fracture:anklesurg-v2', n_envs=1, env_kwargs=eval_env_kwargs, vec_env_cls=SubprocVecEnv, seed=eval_seed)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False

    log_callback1 = log_callback.CustomCallback()
    success_callback = StopTrainingOnSuccessRate(
        vec_env=eval_env, 
        max_no_improvement_evals=10, 
        success_threshold=0.90,  
        min_evals=1, verbose=1, 
        model_name=model_name,
        model_save_path=f'./best_models/{ran}'
    )
    eval_callback = EvalCallback(
        eval_env, eval_freq=10000,
        deterministic=True, n_eval_episodes=50,
        callback_after_eval=success_callback
    )

    callback = [eval_callback, log_callback1] if log == 1 else [eval_callback]
    
    # --- STAGE 1: BASELINE TRAINING ---
    print("\n=== Stage 1: Base TD3 Training ===")
    model.learn(1_000_000, callback=callback)

    # --- STAGE 2: ITERATIVE FORCE THRESHOLD SEARCH ---
    best_threshold = current_force_threshold
    best_model_save_path = f'./best_models/{ran}/{model_name}/{model_name}.zip'

    if run_iterative_search:
        print("\n=== Stage 2: Iterative Threshold Decay Search ===")
        
        # Evaluate baseline performance
        base_succ, mean_p_force, max_p_force = run_threshold_evaluation(model, eval_env, n_episodes=30)
        print(f"[Iter 0] Base Threshold: {current_force_threshold:.3f}N | Success: {base_succ*100:.1f}% | Mean Peak Force: {mean_p_force:.3f}N")

        # Set initial decay boundary relative to observed policy behavior
        if max_p_force > 0 and max_p_force < current_force_threshold:
            current_force_threshold = max_p_force * decay_factor
        else:
            current_force_threshold = current_force_threshold * decay_factor

        for iteration in range(1, max_tuning_iters + 1):
            print(f"\n--- Search Iteration {iteration}/{max_tuning_iters} | Target Threshold: {current_force_threshold:.4f}N ---")

            # Update threshold in environment specs
            env_kwargs['maximum_contact_force_threshold'] = current_force_threshold
            eval_env_kwargs['maximum_contact_force_threshold'] = current_force_threshold

            # Re-create environments with updated threshold
            env.close()
            eval_env.close()
            
            env = make_vec_env('gym_fracture:anklesurg-v2', env_kwargs=env_kwargs, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
            env = VecNormalize(env, norm_obs=True, norm_reward=False)
            
            eval_env = make_vec_env('gym_fracture:anklesurg-v2', n_envs=1, env_kwargs=eval_env_kwargs, vec_env_cls=SubprocVecEnv, seed=eval_seed)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
            eval_env.training = False

            # Attach updated environments to current model & fine-tune
            model.set_env(env)
            model.learn(total_timesteps=fine_tune_timesteps, log_interval=10)

            # Evaluate performance at new threshold
            iter_succ, iter_mean_force, _ = run_threshold_evaluation(model, eval_env, n_episodes=30)
            print(f"[Iter {iteration}] Success Rate: {iter_succ*100:.1f}% | Mean Peak Force: {iter_mean_force:.3f}N")

            if log == 1:
                wandb.log({
                    "Tuning/Iteration": iteration,
                    "Tuning/Threshold": current_force_threshold,
                    "Tuning/Success Rate": iter_succ,
                    "Tuning/Mean Peak Force": iter_mean_force
                })

            # Check termination criteria based on policy success drop
            if iter_succ >= target_success_rate:
                best_threshold = current_force_threshold
                # Save checkpoint of best tuned model
                tuned_path = f'./best_models/{ran}/{model_name}_tuned_iter{iteration}'
                model.save(tuned_path)
                best_model_save_path = f"{tuned_path}.zip"
                current_force_threshold *= decay_factor
            else:
                print(f"Success rate drop detected ({iter_succ*100:.1f}% < {target_success_rate*100:.1f}%). Halting search.")
                break

        print(f"\nOptimal Contact Force Threshold Determined: {best_threshold:.4f}N")

    # --- STAGE 3: SOFT EVALUATION BENCHMARK ---
    print("\n=== Stage 3: Benchmark on Soft Evaluation Environment ===")
    vtk_file = 'rect0009.vtk'
    soft_eval_env_kwargs = {
        'reward_type': 'sparse',
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'dt': 0.001,
        'dr': 0.01,
        'distance_threshold_ori': threshold_ori,
        'maximum_contact_force_threshold': best_threshold, # Using optimal threshold found
        'softtissue': 'soft',
        'number_of_springs': num_springs,
        'youngs_modulus': 1.5e6,
        'youngs_modulus_type': youngs_modulus_type,
        'vtk_file': vtk_file,
        'patient': 110,
        'action_type': 'euler',
        'maxforce': maxforce,
        'contact_type': contact_type,
        'start_pos': 'home',
        'render_mode': 'direct',
        'test': True,
    }

    soft_eval_env = make_vec_env('gym_fracture:anklesurg-v2', n_envs=10, env_kwargs=soft_eval_env_kwargs, vec_env_cls=SubprocVecEnv, seed=eval_seed)
    stats_path = f'./best_models/{ran}/{model_name}/vec_normalize.pkl'
    
    if os.path.exists(stats_path):
        soft_eval_env = VecNormalize.load(stats_path, soft_eval_env)
    else:
        soft_eval_env = VecNormalize(soft_eval_env, norm_obs=True, norm_reward=False)
        
    soft_eval_env.training = False       
    soft_eval_env.norm_reward = False

    # Load model from the optimal iterative checkpoint
    eval_model = TD3.load(best_model_save_path, env=soft_eval_env)

    dones, contacts, explosions = [], [], []
    num = 1000
    episodes_collected = 0
    obs = soft_eval_env.reset()

    # Step-level and per-env logs
    all_step_contact_forces, all_step_agent_forces = [], []
    overall_succ_with_contact_forces, overall_succ_without_contact_forces = [], []
    overall_fail_with_contact_forces, overall_fail_without_contact_forces = [], []
    overall_succ_with_contact_agent_forces, overall_succ_without_contact_agent_forces = [], []
    overall_fail_with_contact_agent_forces, overall_fail_without_contact_agent_forces = [], []

    clean_succ_with_contact_forces, clean_succ_without_contact_forces = [], []
    clean_fail_with_contact_forces, clean_fail_without_contact_forces = [], []
    clean_succ_with_contact_agent_forces, clean_succ_without_contact_agent_forces = [], []
    clean_fail_with_contact_agent_forces, clean_fail_without_contact_agent_forces = [], []

    exp_succ_with_contact_forces, exp_succ_without_contact_forces = [], []
    exp_fail_with_contact_forces, exp_fail_without_contact_forces = [], []
    exp_succ_with_contact_agent_forces, exp_succ_without_contact_agent_forces = [], []
    exp_fail_with_contact_agent_forces, exp_fail_without_contact_agent_forces = [], []

    env_step_contact_forces = [[] for _ in range(soft_eval_env.num_envs)]
    env_step_agent_forces = [[] for _ in range(soft_eval_env.num_envs)]

    while episodes_collected < num:
        action, _ = eval_model.predict(obs, deterministic=True)
        obs, reward, dones_array, info_list = soft_eval_env.step(action)
        
        # 1. STEP-LEVEL TRACKING
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
                has_exploded = info.get("exploded", False)

                ep_c_forces = env_step_contact_forces[i]
                ep_max_contact_force = max(ep_c_forces) if ep_c_forces else 0.0
                ep_avg_contact_force = sum(ep_c_forces) / len(ep_c_forces) if ep_c_forces else 0.0

                ep_a_forces = env_step_agent_forces[i]
                ep_max_agent_force = max(ep_a_forces) if ep_a_forces else 0.0
                ep_avg_agent_force = sum(ep_a_forces) / len(ep_a_forces) if ep_a_forces else 0.0

                dones.append(is_success)
                contacts.append(has_contact)
                explosions.append(has_exploded)

                # Populate overall buffers
                if is_success and has_contact:
                    overall_succ_with_contact_forces.append(ep_max_contact_force)
                    overall_succ_with_contact_agent_forces.append(ep_max_agent_force)
                elif is_success and not has_contact:
                    overall_succ_without_contact_forces.append(ep_max_contact_force)
                    overall_succ_without_contact_agent_forces.append(ep_max_agent_force)
                elif not is_success and has_contact:
                    overall_fail_with_contact_forces.append(ep_max_contact_force)
                    overall_fail_with_contact_agent_forces.append(ep_max_agent_force)
                else:
                    overall_fail_without_contact_forces.append(ep_max_contact_force)
                    overall_fail_without_contact_agent_forces.append(ep_max_agent_force)

                # Populate clean vs exploded buffers
                if has_exploded:
                    if is_success and has_contact:
                        exp_succ_with_contact_forces.append(ep_max_contact_force)
                        exp_succ_with_contact_agent_forces.append(ep_max_agent_force)
                    elif is_success and not has_contact:
                        exp_succ_without_contact_forces.append(ep_max_contact_force)
                        exp_succ_without_contact_agent_forces.append(ep_max_agent_force)
                    elif not is_success and has_contact:
                        exp_fail_with_contact_forces.append(ep_max_contact_force)
                        exp_fail_with_contact_agent_forces.append(ep_max_agent_force)
                    else:
                        exp_fail_without_contact_forces.append(ep_max_contact_force)
                        exp_fail_without_contact_agent_forces.append(ep_max_agent_force)
                else:
                    if is_success and has_contact:
                        clean_succ_with_contact_forces.append(ep_max_contact_force)
                        clean_succ_with_contact_agent_forces.append(ep_max_agent_force)
                    elif is_success and not has_contact:
                        clean_succ_without_contact_forces.append(ep_max_contact_force)
                        clean_succ_without_contact_agent_forces.append(ep_max_agent_force)
                    elif not is_success and has_contact:
                        clean_fail_with_contact_forces.append(ep_max_contact_force)
                        clean_fail_with_contact_agent_forces.append(ep_max_agent_force)
                    else:
                        clean_fail_without_contact_forces.append(ep_max_contact_force)
                        clean_fail_without_contact_agent_forces.append(ep_max_agent_force)

                episodes_collected += 1
                print(f"[{episodes_collected}/{num}] Env {i} | Success: {is_success} | Contact: {has_contact} | Exploded: {has_exploded} | "
                      f"Max Contact Force: {ep_max_contact_force:.2f}N | Max Agent Force: {ep_max_agent_force:.2f}N")

                env_step_contact_forces[i] = []
                env_step_agent_forces[i] = []

                valid_dones = [d for d, e in zip(dones, explosions) if not e]
                not_exploded_success_rate = (sum(valid_dones) / len(valid_dones)) if len(valid_dones) > 0 else 0.0

                if log == 1:
                    wandb.log({
                        "Episode": episodes_collected,
                        "Success": is_success,
                        "Contact": has_contact,
                        "Exploded": has_exploded,
                        "Episode Max Contact Force": ep_max_contact_force,
                        "Episode Avg Contact Force": ep_avg_contact_force,
                        "Episode Max Agent Force": ep_max_agent_force,
                        "Episode Avg Agent Force": ep_avg_agent_force,
                        "Overall Success Rate": sum(dones) / len(dones),
                        "Clean Success Rate": not_exploded_success_rate
                    })

                    wandb.run.summary['Count Overall/Success With Contact'] = len(overall_succ_with_contact_forces)
                    wandb.run.summary['Count Overall/Success Without Contact'] = len(overall_succ_without_contact_forces)
                    wandb.run.summary['Count Overall/Fail With Contact'] = len(overall_fail_with_contact_forces)
                    wandb.run.summary['Count Overall/Fail Without Contact'] = len(overall_fail_without_contact_forces)

                    wandb.run.summary['Count Clean/Success With Contact'] = len(clean_succ_with_contact_forces)
                    wandb.run.summary['Count Clean/Success Without Contact'] = len(clean_succ_without_contact_forces)
                    wandb.run.summary['Count Clean/Fail With Contact'] = len(clean_fail_with_contact_forces)
                    wandb.run.summary['Count Clean/Fail Without Contact'] = len(clean_fail_without_contact_forces)

                    if overall_succ_with_contact_forces:
                        wandb.run.summary['Contact Force Overall/Success With Contact (Avg Max)'] = sum(overall_succ_with_contact_forces) / len(overall_succ_with_contact_forces)
                    if overall_succ_without_contact_forces:
                        wandb.run.summary['Contact Force Overall/Success Without Contact (Avg Max)'] = sum(overall_succ_without_contact_forces) / len(overall_succ_without_contact_forces)
                    if overall_fail_with_contact_forces:
                        wandb.run.summary['Contact Force Overall/Fail With Contact (Avg Max)'] = sum(overall_fail_with_contact_forces) / len(overall_fail_with_contact_forces)
                    if overall_fail_without_contact_forces:
                        wandb.run.summary['Contact Force Overall/Fail Without Contact (Avg Max)'] = sum(overall_fail_without_contact_forces) / len(overall_fail_without_contact_forces)

                    if clean_succ_with_contact_forces:
                        wandb.run.summary['Contact Force Clean/Success With Contact (Avg Max)'] = sum(clean_succ_with_contact_forces) / len(clean_succ_with_contact_forces)
                    if clean_succ_without_contact_forces:
                        wandb.run.summary['Contact Force Clean/Success Without Contact (Avg Max)'] = sum(clean_succ_without_contact_forces) / len(clean_succ_without_contact_forces)
                    if clean_fail_with_contact_forces:
                        wandb.run.summary['Contact Force Clean/Fail With Contact (Avg Max)'] = sum(clean_fail_with_contact_forces) / len(clean_fail_with_contact_forces)
                    if clean_fail_without_contact_forces:
                        wandb.run.summary['Contact Force Clean/Fail Without Contact (Avg Max)'] = sum(clean_fail_without_contact_forces) / len(clean_fail_without_contact_forces)

                if episodes_collected >= num:
                    break

    print("\nEvaluation complete. Cleaning up resources...")
    soft_eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 model with iterative force threshold tuning.')
    parser.add_argument('--threshold_pos', type=float, default=0.005)
    parser.add_argument('--threshold_ori', type=float, default=0.05)
    parser.add_argument('--maximum_contact_force_threshold', type=float, default=0.5)
    parser.add_argument('--action_type', type=str, default='euler')
    parser.add_argument('--render_mode', type=str, default="human")
    parser.add_argument('--maxforce', type=float, default=4)
    parser.add_argument('--softtissue', type=str, default="spring")
    parser.add_argument('--num_springs', type=int, default=3)
    parser.add_argument('--contact_type', type=int, default=0)
    parser.add_argument('--youngs_modulus', type=float, default=1e7)
    parser.add_argument('--youngs_modulus_type', type=str, default='eval_mode')
    parser.add_argument('--randomise_ligs', type=int, default=0)
    parser.add_argument('--randomise_start', type=int, default=0)
    parser.add_argument('--randomise_num_springs', type=int, default=0)
    parser.add_argument('--ran', type=str, default="1")
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    
    # New CLI Arguments for Iterative Search
    parser.add_argument('--run_iterative_search', type=int, default=1, help='Set 1 to enable iterative force threshold decay.')
    parser.add_argument('--decay_factor', type=float, default=0.8, help='Decay multiplier for force threshold (e.g., 0.8).')
    parser.add_argument('--target_success_rate', type=float, default=0.85, help='Minimum acceptable success rate before stopping decay.')
    parser.add_argument('--max_tuning_iters', type=int, default=10, help='Number of threshold reduction iterations.')
    parser.add_argument('--fine_tune_timesteps', type=int, default=50000, help='Timesteps to fine-tune model at each threshold step.')

    args = parser.parse_args()
    
    train(
        threshold_pos=args.threshold_pos, 
        threshold_ori=args.threshold_ori, 
        action_type=args.action_type, 
        render_mode=args.render_mode,
        maxforce=args.maxforce, 
        num_springs=args.num_springs,
        contact_type=args.contact_type,
        softtissue=args.softtissue, 
        max_contact_force_threshold=args.maximum_contact_force_threshold,
        ran=args.ran,
        log=args.log,
        youngs_modulus=args.youngs_modulus,
        youngs_modulus_type=args.youngs_modulus_type,
        randomise_ligs=args.randomise_ligs,
        randomise_start=args.randomise_start,
        seed=args.seed,
        run_iterative_search=bool(args.run_iterative_search),
        decay_factor=args.decay_factor,
        target_success_rate=args.target_success_rate,
        max_tuning_iters=args.max_tuning_iters,
        fine_tune_timesteps=args.fine_tune_timesteps
    )