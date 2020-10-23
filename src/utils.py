import numpy as np
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


def benchmark_do_nothing(env, n_episodes = 10, percentile = 75):
    """"""
    rewards = []
    no_act = env.actuator.do_nothing
    for i in range(n_episodes):
        _ = env.reset()
        R = 0
        done = False
        while not done:
            _, reward, done, info = env.step(no_act)
            R += reward
        rewards.append(R)
    return np.percentile(rewards, percentile)