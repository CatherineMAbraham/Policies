import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import wandb


def multiple_envs(
    model_path,
    patient=110,
    threshold_pos=0.0005,
    threshold_ori=0.00872665,  # np.deg2rad(0.5)
    maxforce=3.3,
    softtissue="spring",
    num_springs=3,
    youngs_modulus=1e7,
    vtk_file="rect0009.vtk",
    n_envs=1,
    num_eps=50,
    log=0,
    seed=42,
    force_limit=0.5,
):
    # Construct env_kwargs dynamically from passed function parameters
    goal_type = [0, 0, 0]
    env_kwargs = {
        "reward_type": "sparse",
        "max_steps": 100,
        "goal_type": goal_type,
        "horizon": "variable",
        "obs_type": "dict",
        "distance_threshold_pos": threshold_pos,
        "dt": 0.001,
        "dr": 0.01,
        "distance_threshold_ori": threshold_ori,
        "action_type": "euler",
        "start_pos": "home",
        "maxforce": maxforce,
        "contact_type": 1,
        "maximum_contact_force_threshold": force_limit,
        "number_of_springs": num_springs,
        "youngs_modulus_type": "testing",
        "randomise_num_springs": 1,
        "randomise_foot_dynamics": 1,
        "randomise_sensor_noise": 1,
        "randomise_start": 1,
        "softtissue": softtissue,
        "patient": patient,
        "vtk_file": vtk_file,
        "test": True,
        "render_mode": "direct",
    }

    # Initialize vectorized environment
    env = make_vec_env(
        "gym_fracture:anklesurg-v2",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        seed=seed,
    )
    #model_path= '/media/catherine/Data/Best Models 30926/model-spring_randomYM_09031956_1'
    # Load observation normalization statistics
    vec_norm_path = os.path.join(model_path, "vec_normalize.pkl")
    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found at: {vec_norm_path}")

    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False

    # Locate and load the latest TD3 model checkpoint
    model_dir = Path(model_path)
    model_candidates = sorted(
        [p for p in model_dir.glob("model*") if p.is_file() and not p.name.endswith("-rb.pkl")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not model_candidates:
        raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")

    selected_model = model_candidates[0]
    model = TD3.load(str(selected_model), env=env)

    # Secondary metrics collection arrays
    dones = []
    contacts = []
    position_error = []
    angle_error = []
    agent_force = []
    contact_forces = []
    interlock_triggers = []
    episode_steps = []
    force_breached = []

    episodes_collected = 0
    ep_contact_forces = [[] for _ in range(env.num_envs)]
    ep_step_counters = [0 for _ in range(env.num_envs)]

    obs = env.reset()

    while episodes_collected < num_eps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones_array, info_list = env.step(action)

        for i in range(env.num_envs):
            ep_step_counters[i] += 1
            current_step_force = info_list[i].get("contact_force", 0.0)
            ep_contact_forces[i].append(current_step_force)

            if dones_array[i]:
                info = info_list[i]

                is_success = info.get("is_success", False)
                has_contact = info.get("contact", False)
                max_force_val = info.get("maximum_force", info.get("force", 0.0))
                pos_dist = info.get("pos_distance", 0.0)
                angle_dist = info.get("angle", 0.0)
                interlocks = info.get("interlock_count", 0)
                steps_taken = ep_step_counters[i]

                # Peak episode contact force (excluding initial step)
                peak_ep_contact = (
                    max(ep_contact_forces[i][1:]) if len(ep_contact_forces[i]) > 1 else 0.0
                )

                is_breached = 1 if max_force_val > force_limit else 0

                # Append metrics
                dones.append(is_success)
                contacts.append(has_contact)
                position_error.append(pos_dist)
                angle_error.append(angle_dist)
                agent_force.append(max_force_val)
                contact_forces.append(peak_ep_contact)
                interlock_triggers.append(interlocks)
                episode_steps.append(steps_taken)
                force_breached.append(is_breached)

                episodes_collected += 1

                print(
                    f"[{episodes_collected}/{num_eps}] Patient {patient} | Env {i} "
                    f"Success: {is_success} | Peak Force: {max_force_val:.3f}N | "
                    f"Peak Contact: {peak_ep_contact:.3f}N  "
                    f"Interlocks: {interlocks} | Steps: {steps_taken} | "
                    f"Pos Err: {pos_dist:.5f}m | Angle Err: {np.rad2deg(angle_dist):.2f}° | "
                    f"Success Rate: {sum(dones) / len(dones):.2%}"
                )

                # Save raw step contact forces for detailed plotting
                np.save(
                    f"ep_contact_forces_patient{patient}_ep{episodes_collected}.npy",
                    np.array(ep_contact_forces[i]),
                )

                # Log episode to WandB
                if log == 1 and max_force_val <= 50:
                    wandb.log(
                        {
                            "Patient_ID": patient,
                            "Episode": episodes_collected,
                            "Success": is_success,
                            "Contact": has_contact,
                            "Peak_Agent_Force": max_force_val,
                            "Force_Violation": is_breached,
                            "Interlock_Triggers": interlocks,
                            "Episode_Steps": steps_taken,
                            "Position_Distance": pos_dist,
                            "Angle_Distance": angle_dist,
                            "Success_Rate": sum(dones) / len(dones),
                        }
                    )

                # Reset per-episode temporary arrays
                ep_contact_forces[i] = []
                ep_step_counters[i] = 0

                if episodes_collected >= num_eps:
                    break

    # Save structured CSV for paper figures and p-value statistical tests
    model_name = Path(model_path).name
    df = pd.DataFrame(
        {
            "patient": patient,
            "success": dones,
            "contact": contacts,
            "pos_error_m": position_error,
            "angle_error_rad": angle_error,
            "peak_force_N": agent_force,
            "peak_contact_force_N": contact_forces,
            "force_violation": force_breached,
            "interlock_triggers": interlock_triggers,
            "episode_steps": episode_steps,
        }
    )
    csv_filename = f"eval_patient_{patient}_{model_name}.csv"
    df.to_csv(csv_filename, index=False)

    # Print formal dissertation summary table
    success_no_contact = sum(1 for d, c in zip(dones, contacts) if d and not c)
    failure_no_contact = sum(1 for d, c in zip(dones, contacts) if not d and not c)
    success_contact = sum(1 for d, c in zip(dones, contacts) if d and c)
    failure_contact = sum(1 for d, c in zip(dones, contacts) if not d and c)

    print(f"\n================ Summary Table (Patient {patient}) ================")
    print(f"{'Metric':<35} {'Value':<15}")
    print("-" * 50)
    print(f"{'Overall Success Rate':<35} {np.mean(dones):.2%}")
    print(f"{'Force Violation Rate (>0.5N)':<35} {np.mean(force_breached):.2%}")
    print(f"{'Avg Interlock Triggers / Ep':<35} {np.mean(interlock_triggers):.2f}")
    print(f"{'Avg Steps to Completion':<35} {np.mean(episode_steps):.2f}")
    print(f"{'Avg Position Error (m)':<35} {np.mean(position_error):.6f}")
    print(f"{'Avg Angle Error (rad)':<35} {np.mean(angle_error):.6f}")
    print(f"{'Avg Peak Agent Force (N)':<35} {np.mean(agent_force):.4f}")
    print(f"{'Avg Contact Force (N)':<35} {np.mean(contact_forces):.4f}")
    print("-" * 50)
    print(f"{'Success, No Contact':<35} {success_no_contact:<15}")
    print(f"{'Failure, No Contact':<35} {failure_no_contact:<15}")
    print(f"{'Success, Contact':<35} {success_contact:<15}")
    print(f"{'Failure, Contact':<35} {failure_contact:<15}")
    print("===================================================================\n")

    if log == 1:
        wandb.run.summary[f"patient_{patient}_success_rate"] = np.mean(dones)
        wandb.run.summary[f"patient_{patient}_violation_rate"] = np.mean(force_breached)
        wandb.run.summary[f"patient_{patient}_avg_interlocks"] = np.mean(interlock_triggers)
        wandb.run.summary[f"patient_{patient}_avg_steps"] = np.mean(episode_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model across randomized patients")
    parser.add_argument("--model_path", type=str, required=False, help="Path to trained model folder")
    parser.add_argument("--maxforce", type=float, default=3.3, help="Max motor command force.")
    parser.add_argument("--youngs_modulus", type=float, default=1e7, help="Tissue Young's modulus.")
    parser.add_argument("--num_springs", type=int, default=3, help="Number of ligament springs.")
    parser.add_argument("--softtissue", type=str, default="spring", help="Soft Tissue Type.")
    parser.add_argument("--vtk_file", type=str, default="rect0009.vtk", help="VTK geometry file")
    parser.add_argument("--threshold_pos", type=float, default=0.0005, help="Position error limit (m)")
    parser.add_argument("--threshold_ori", type=float, default=0.5, help="Angle error limit (deg)")
    parser.add_argument("--n_envs", type=int, default=1, help="Parallel environment count")
    parser.add_argument("--num_eps", type=int, default=50, help="Episodes to evaluate per patient")
    parser.add_argument("--log", type=int, default=0, help="Log to Weights & Biases (0 or 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.log == 1:
        model_name_clean = args.model_path.split("/")[-1].split(".")[0]
        wandb.init(project="softsurg", name=f"Eval_{model_name_clean}")

    patients = [252, 102, 132, 198]
    for patient in patients:
        multiple_envs(
            model_path=args.model_path,
            patient=patient,
            threshold_pos=args.threshold_pos,
            threshold_ori=np.deg2rad(args.threshold_ori),
            maxforce=args.maxforce,
            softtissue=args.softtissue,
            num_springs=args.num_springs,
            youngs_modulus=args.youngs_modulus,
            vtk_file=args.vtk_file,
            n_envs=args.n_envs,
            num_eps=args.num_eps,
            log=args.log,
            seed=args.seed,
        )