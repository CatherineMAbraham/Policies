import wandb

sweep_config = {
    "method": "random",
    "metric": {"name": "rollout/success_rate", "goal": "maximize"},
    "parameters": {
        # Core Shared Hyperparameters
        "learning_rate": {"values": [3e-5, 1e-4, 3e-4, 1e-3], "distribution": "categorical"}, # SAC handles higher LRs better than TD3
        "gamma": {"values": [0.95, 0.98, 0.99, 0.995], "distribution": "categorical"},       # Robotics often favors higher gamma
        "tau": {"values": [0.005, 0.01, 0.02], "distribution": "categorical"},               # SAC usually prefers smaller Polyak smoothing (0.005 is standard)
        "batch_size": {"values": [128, 256, 512], "distribution": "categorical"},            # SAC thrives on larger batches for stable gradient steps
        "train_freq": {"values": [1, 2, 4], "distribution": "categorical"},  
        "learning_starts": {"values": [1000, 5000, 10000], "distribution": "categorical"},  # SAC benefits from a larger initial random buffer to map entropy
        "net_arch": {
            "values": [[256, 256], [256, 256, 256], [400, 300]], "distribution": "categorical"
        },
        "her_sampled_goal": {"values": [4, 8], "distribution": "categorical"},

        # --- SAC EXCLUSIVE PARAMETERS ---
        "ent_coef": {
            "values": ["auto", 0.1, 0.01], "distribution": "categorical"
        },
        # If ent_coef is 'auto', you can sweep the learning rate of the entropy coefficient itself
        "target_kl": {
            "values": [0.01, 0.05, 0.2], "distribution": "categorical" 
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 3,
        "max_iter": 81
    }
}

# Environment config
# sweep_id = wandb.sweep(sweep_config, project="Chp1-SAC-Sweep", entity="cmabraham1-university-of-sheffield")
# print(f"Initialized SAC Sweep ID: {sweep_id}")

#Environment config


sweep_id = wandb.sweep(sweep_config, project="Chp1-Sweep-SAC", entity="cmabraham1-university-of-sheffield")
print(f"Initialized Sweep ID: {sweep_id}")