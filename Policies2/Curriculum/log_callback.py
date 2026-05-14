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
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if 'infos' in self.locals:
            infos = self.locals['infos']
            for j in range(len(infos)):
                if self.locals['dones'][j]:
                    wandb.log({f'max_force{j}': infos[j].get('force', 0), 
                               f'Holding{j}': infos[j].get('isHolding', 0),
                               f'Contact{j}': infos[j].get('contact', 0),
                               f'Position Distance{j}': infos[j].get('pos_distance', 0),
                               f'Angle Distance{j}': infos[j].get('angle', 0)})
                    #print(f"Max Force{j}: {infos[j].get('force')}, Contact{j}: {infos[j].get('contact')}")
        return True

    