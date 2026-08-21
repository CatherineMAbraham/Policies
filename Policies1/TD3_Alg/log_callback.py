from stable_baselines3.common.callbacks import BaseCallback
import wandb


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            infos = self.locals['infos']
            log_dict = {}

            for j in range(len(infos)):
                # Logged EVERY step
                log_dict[f'step_force{j}'] = infos[j].get('force', 0)

                # Logged ONLY when an episode ends
                if self.locals['dones'][j]:
                    log_dict[f'Holding{j}'] = infos[j].get('isHolding', 0)
                    log_dict[f'Contact{j}'] = infos[j].get('contact', 0)
                    log_dict[f'Position Distance{j}'] = infos[j].get('pos_distance', 0)
                    log_dict[f'Angle Distance{j}'] = infos[j].get('angle', 0)
                    log_dict[f'Youngs Modulus{j}'] = infos[j].get('young_modulus', 0)
                    log_dict[f'Width{j}'] = infos[j].get('width', 0)
                    log_dict[f'maximum_force{j}'] = infos[j].get('maximum_force', 0)

            # Single log call anchors all metrics to the exact environment timestep
            wandb.log(log_dict, step=self.num_timesteps)

        return True

    