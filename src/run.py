"""
Execute the model.
"""


from torch import nn as nn

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.a2c import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env

from environment import GridEnv
from network import GridCNN
from agent import CustomGridPolicy


# Execution params
EVAL_FREQ = 1000
SEED = 42
DISCRETE_ENV = False
NUM_ENVS = 4
TB_LOGS = './tb_logs/'
MODEL_PATHS = './logs/'
NUM_TIMESTEPS = 1e5
LR = 0.0009
GAMMA = 0.9
LAMBDA = 0.9
STEPS_PER_UPDATE = 10
RUN_NAME = 'A2C_multibinary'

# Environment arguments
env_kwargs = dict(
    discrete = DISCRETE_ENV, 
    seed = SEED
)

# Environments for training and evaluation
env = make_vec_env(GridEnv, n_envs = NUM_ENVS, env_kwargs = env_kwargs, seed = SEED)
eval_env = GridEnv(discrete = DISCRETE_ENV, seed = SEED // 2)
policy_kwargs = dict(mask=eval_env.actuator.mask)

# Callback for eval
eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_PATHS,
                            log_path=MODEL_PATHS, eval_freq=EVAL_FREQ,
                            deterministic=True, render=False)


# A2C model
model = A2C(CustomGridPolicy, 
            env,
            learning_rate=LR,
            n_steps=STEPS_PER_UPDATE,
            gamma=GAMMA,
            gae_lambda=LAMBDA,
            policy_kwargs=policy_kwargs,
            tensorboard_log = TB_LOGS, 
            seed = SEED, 
            verbose = 2)

# Train model
model.learn(total_timesteps = NUM_TIMESTEPS, 
            tb_log_name=RUN_NAME, 
            callback=eval_callback
)


## Benchmark do nothing: 
    # Episode Reward: 585.26 +/- 285.352
    # Episode length: 810.6 +/- 408.576


# https://discuss.pytorch.org/t/implement-selected-sparse-connected-neural-network/45517/2
# https://discuss.pytorch.org/t/network-custom-connections-paired-connections-performance-issue/11713
# https://discuss.pytorch.org/t/how-to-build-a-custom-connections/64368/3 <<--