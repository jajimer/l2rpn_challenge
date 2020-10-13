from stable_baselines3.common.callbacks import BaseCallback, EventCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log grads
        for name, param in self.model.policy.named_parameters():
            self.logger.record('grads/'+name, param.grad)
        return True