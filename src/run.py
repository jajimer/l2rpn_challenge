"""
Execute the model.
"""

import logging
import sys
import os
import argparse
from datetime import datetime

from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EveryNTimesteps
from stable_baselines3.a2c import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env

from environment import GridEnv
from network import GridCNN
from agent import CustomGridPolicy
from utils import TensorboardCallback


def main():
    """"""
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Algorithm", type=str, default='A2C')
    parser.add_argument("--n", help="Number of environments", type=int, default=4)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
    parser.add_argument("--update_steps", help="Number of steps per update", type=int, default=5)
    parser.add_argument("--gamma", help="Discount factor", type=float, default=0.99)
    parser.add_argument("--gae", help="GAE factor", type=float, default=1.0)
    parser.add_argument("--coef_ent", help="Entropy coefficient", type=float, default=0.01)
    parser.add_argument("--coef_vf", help="V(s) coefficient", type=float, default=0.5)
    parser.add_argument('--use_adam', help="Use Adam in A2C (RMSPROP otherwise)", action='store_true')
    parser.add_argument('--norm_adv', help="Normalize advantage", action='store_true')
    parser.add_argument("--total_steps", help="Number of total steps", type=float, default=1e6)
    parser.add_argument('--use_backend', action='store_true')
    args = parser.parse_args()

    # Execution params
    SEED = 42
    EVAL_FREQ = 1000
    TB_LOGS = './tb_logs/'
    MODEL_PATHS = './logs/'

    # Log
    exp_id = str(args.algo) + str(args.n) + datetime.now().strftime('%Y%m%d%H%M')
    log_path = MODEL_PATHS + exp_id
    log_file = '%s/params.log' % log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
#    sys.stdout = Logger(LOGFILE)
    log = logging.getLogger(exp_id)
    log.setLevel(logging.INFO)
    hdlr = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr)
    log.info('Experiment %s created.' % exp_id)
    log.info(str(args)[10:-1])

    # Environment arguments
    env_kwargs = dict(seed = SEED, use_backend = args.use_backend)

    # Environments for training and evaluation
    env = make_vec_env(GridEnv, n_envs = args.n, env_kwargs = env_kwargs, seed = SEED)
    eval_env = GridEnv(seed = SEED // 2, use_backend = args.use_backend)
    log.info('Number of available actions: %d' % eval_env.action_space.n)

    # Callbacks for eval and logging
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                log_path=log_path, eval_freq=EVAL_FREQ,
                                deterministic=True, render=False)

    event_callback = EveryNTimesteps(n_steps=100, callback=TensorboardCallback())
    callbacks = CallbackList([eval_callback, event_callback])

    # Policy arguments
    policy_kwargs = {} #dict(optimizer_kwargs = dict(weight_decay = 0.5))
    policy = CustomGridPolicy

    # Model
    if args.algo == 'A2C':
        model = A2C(policy, 
                    env,
                    learning_rate=args.lr,
                    n_steps=args.update_steps,
                    gamma=args.gamma,
                    gae_lambda=args.gae,
                    ent_coef=args.coef_ent,
                    vf_coef=args.coef_vf,
                    use_rms_prop=not args.use_adam,
                    normalize_advantage=args.norm_adv,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log = TB_LOGS, 
                    seed = SEED, 
                    verbose = 2)
    else:
        model = PPO(policy, env,
                    n_steps=args.update_steps,
                    learning_rate=args.lr,
                    gamma=args.gamma,
                    gae_lambda=args.gae,
                    ent_coef=args.coef_ent,
                    vf_coef=args.coef_vf,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log = TB_LOGS, 
                    seed = SEED, 
                    verbose = 2)

    # Train model
    model.learn(total_timesteps = args.total_steps, 
        tb_log_name=exp_id,
        callback=callbacks)
    log.info('Done!')
    return True


if __name__ == "__main__":
    main()

