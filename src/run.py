import grid2op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import PGAgent
from sensors import *
from grid2op.Reward import L2RPNReward
plt.style.use('seaborn')


def run(run_name, num_episodes, benchmark = False):
    env = grid2op.make("l2rpn_neurips_2020_track1_small", reward_class=L2RPNReward, difficulty = '0') # ['0', '1', '2', None for default]
    env.seed(42)
    if not benchmark:
        agent = PGAgent((100,), 4)

    scores = []
    steps = []
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        score = 0
        step = 0
        action_space = env.action_space
        x = np.zeros(177, dtype=np.bool)
        while not done:
            if benchmark:
                action = env.action_space()
            else:
                action = agent.get_action(obs.to_vect()[:100]).astype('bool')
                x[:4] = action
                a = action_space({'change_bus': x})
                if a.is_ambiguous():
                    print('Ambiguous action')
                if action_space._is_legal(a, env):
                    print('Illegal action')
            obs, reward, done, info = env.step(a)
            score += reward
            step += 1
        scores.append(score)
        steps.append(step)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)


    window = 20
    mean_score = [elem for elem in pd.Series.rolling(pd.Series(scores), window).mean()]

    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Avg. score')
    plt.title(run_name)
    plt.savefig('%s_score.png' % run_name)
    plt.close()

    plt.plot(steps)
    plt.xlabel('Episode')
    plt.ylabel('Episode duration')
    plt.title(run_name)
    plt.savefig('%s_steps.png' % run_name)
    plt.close()

    return scores, steps



if __name__ == '__main__':

    run_name = 'benchmark_donothing_lvl0'
    score, steps = run(run_name, 10, False)
