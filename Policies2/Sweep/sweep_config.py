import wandb
#TD3 config
sweep_config = {"method": "random",
    "metric": {"name": "rollout/success_rate", "goal": "maximize"},
    "parameters": {
    "learning_rate": {"values": [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3], "distribution": "categorical"},
    "gamma": {"values": [0.9,0.93,0.95,0.97,0.99], "distribution": "categorical"},
    "tau": {"values": [0.1,0.07,0.05,0.02,0.01,0.005], "distribution": "categorical"},
    "batch_size": {"values": [64, 128, 256, 512], "distribution": "categorical"},
    "train_freq": {"values": [1,2,4,5], "distribution": "categorical"},  
    "learning_starts": {"values": [500, 1000, 2000], "distribution": "categorical"},
    "net_arch": {
        "values": [[256, 256, 256], [400,300]],"distribution": "categorical"},
    "her_sampled_goal": {"values": [4,8,16], "distribution": "categorical"},
}

    ,
    "early_terminate": {
        "type": "hyperband",
        "s": 2,
        "eta": 3,
        "max_iter": 81
    }
}

#Environment config


sweep_id = wandb.sweep(sweep_config, project="Chp2-Sweep", entity="cmabraham1-university-of-sheffield")
print(f"Initialized Sweep ID: {sweep_id}")