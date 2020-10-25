"""
Execute the model.
"""

import logging
import sys
import os
import argparse
from datetime import datetime

from stable_baselines3.common.callbacks import EvalCallback, CallbackList, EveryNTimesteps, StopTrainingOnRewardThreshold
from stable_baselines3.a2c import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env

from environment import GridEnv
from network import GridCNN
from agent import CustomGridPolicy
from utils import TensorboardCallback, benchmark_do_nothing


# Execution params
SEED = 42
EVAL_FREQ = 500
EVAL_EPISODES = 10
TB_LOGS = './tb_logs/'
MODEL_PATHS = './logs'


def curriculum_learning(exp_id, args, log, levels = ['1', '2', 'competition']):
    """"""
    for i, level in enumerate(levels):
        log.info('#########')
        # Environment arguments
        env_kwargs = dict(seed = SEED, use_backend = args.use_backend, difficulty = level)
        # Environments for training and evaluation
        env = make_vec_env(GridEnv, n_envs = args.n, env_kwargs = env_kwargs, seed = SEED)
        eval_env = GridEnv(seed = SEED // 2, use_backend = args.use_backend, difficulty = level)
        log.info('Difficulty: %s' % level)
        # Benchmark score
        log.info('Determining reward threshold...')
        score_benchmark = benchmark_do_nothing(eval_env, EVAL_EPISODES)
        log.info('Benchmark score: %.3f' % score_benchmark)
        # Callbacks for eval and logging
        path_evals = '%s/%s/difficulty_%s' % (MODEL_PATHS, exp_id, level)
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=score_benchmark, verbose=1)
        eval_callback = EvalCallback(eval_env, best_model_save_path=path_evals,
                                    callback_on_new_best=callback_on_best,
                                    log_path=path_evals, eval_freq=EVAL_FREQ,
                                    n_eval_episodes=EVAL_EPISODES,
                                    deterministic=True, render=False)

        event_callback = EveryNTimesteps(n_steps=1000, callback=TensorboardCallback())
        callbacks = CallbackList([eval_callback, event_callback])
        # Policy arguments
        policy_kwargs = {} #dict(optimizer_kwargs = dict(weight_decay = 0.5))
        policy = CustomGridPolicy
        # Reset environments
        _ = env.reset()
        _ = eval_env.reset()
        # Model
        if args.algo == 'A2C':
            if i == 0:
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
                log.info('Loading from level %s...' % levels[i-1])
                path_last_model = '%s/%s/difficulty_%s' % (MODEL_PATHS, exp_id, levels[i-1]) 
                model = A2C.load('%s/best_model' % path_last_model, env)
        else:
            if i == 0:
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
            else:
                log.info('Loading from level %s...' % levels[i-1])
                path_last_model = '%s/%s/difficulty_%s' % (MODEL_PATHS, exp_id, levels[i-1])
                model = PPO.load('%s/best_model' % path_last_model, env)
        
        # Train model
        model.learn(total_timesteps = args.total_steps, 
            tb_log_name=exp_id,
            callback=callbacks)
        log.info('Done!')
    return model
        

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
    parser.add_argument("--custom_name", help="Name of the run", type=str, default='')
    args = parser.parse_args()

    # Log
    name = '_%s_' % args.custom_name
    exp_id = str(args.algo) + str(args.n) + name + datetime.now().strftime('%Y%m%d%H%M')
    log_path = '%s/%s' % (MODEL_PATHS, exp_id)
    log_file = '%s/%s/params.log' % (MODEL_PATHS, exp_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = logging.getLogger(exp_id)
    log.setLevel(logging.INFO)
    hdlr = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    hdlr.setFormatter(formatter)
    log.addHandler(hdlr)
    log.info('Experiment %s created.' % exp_id)
    log.info(str(args)[10:-1])

    final_model = curriculum_learning(exp_id, args, log)

    return True


if __name__ == "__main__":
    main()

