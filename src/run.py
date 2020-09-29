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


# Execution params
EVAL_FREQ = 1000
SEED = 42
DISCRETE_ENV = True
NUM_ENVS = 4
TB_LOGS = './tb_logs/'
MODEL_PATHS = './logs/'
NUM_TIMESTEPS = 1e5
LR = 0.001
STEPS_PER_UPDATE = 5
RUN_NAME = 'A2C_discrete'

# Policy arguments
policy_kwargs = dict(
    features_extractor_class=GridCNN,
    normalize_images=False,
    features_extractor_kwargs=dict(features_dim=512),
    net_arch = [128, dict(vf=[64], pi=[64])],
#    activation_fn = nn.ReLU
)

# Environment arguments
env_kwargs = dict(
    discrete = DISCRETE_ENV, 
    seed = SEED
)

# Environments for training and evaluation
env = make_vec_env(GridEnv, n_envs = NUM_ENVS, env_kwargs = env_kwargs, seed = SEED)
eval_env = GridEnv(discrete = DISCRETE_ENV, seed = SEED // 2)

# Callback for eval
eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_PATHS,
                            log_path=MODEL_PATHS, eval_freq=EVAL_FREQ,
                            deterministic=True, render=False)

# A2C model
model = A2C('CnnPolicy', 
            env,
            learning_rate=LR,
            n_steps=STEPS_PER_UPDATE,
            policy_kwargs=policy_kwargs,
            tensorboard_log = TB_LOGS, 
            seed = SEED, 
            verbose = 2)

# Train model
model.learn(total_timesteps = NUM_TIMESTEPS, 
            tb_log_name=RUN_NAME, 
            callback=eval_callback)


#TODO
# MultiBinary() but limit actions to belong to same stations

## Benchmark do nothing: 
    # Episode Reward: 585.26 +/- 285.352
    # Episode length: 810.6 +/- 408.576
