import grid2op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import PGAgent
from sensors import *

if __name__ == '__main__':
    env = grid2op.make('rte_case14_realistic', test = False)
    agent = PGAgent()
    n_games = 1000


    scores = []
    for i in range(n_games):
        done = False
        observation = obs_to_state(env.reset())
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = obs_to_state(observation_)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)

    window = 20
    mean_score_norm = [elem for elem in pd.Series.rolling(pd.Series(scores), window).mean()]
        
    plt.plot(mean_score_norm)
    plt.xlabel('Episode')
    plt.ylabel('Avg. score')
    plt.savefig('pg_rte-case14-false.png')
    plt.close()
