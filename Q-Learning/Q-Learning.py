import gym
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys


from collections import defaultdict

from torch import ne
from windy_gridworld import WindyGridworldEnv
import plotting

matplotlib.style.use('ggplot')

env = WindyGridworldEnv()

# Creates epsilon-greedy policy w/given Q func, epsilon
# Returns func. that returns probs. for ea action in array size of action len
def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,
                                dtype = float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities
    
    return policyFunction

# Q-learning model
def QLearning(env, num_episodes, discount_fac = 1.0, alpha = 0.6, epsilon = 0.1):
    # QLearning alg: finds optimal greedy policy
    
    # Actiion value func
    # Nested dict. that maps state -> (action -> action_value)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths = np.zeros(num_episodes),
        episode_rewards = np.zeros(num_episodes))
    
    # Create e-greedy pol. func. for env. act. space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)
    # Iterate ea episode
    for ith_ep in range(num_episodes):
        # Reset env. & pick first act.
        state = env.reset()
        
        for t in itertools.count():
            # Get probs. of all acts. from curr state
            action_probabilities = policy(state)
            # Choose act. according to prob. distrib.
            action = np.random_choice(np.arrange(
                        len(action_probabilities)),
                        p = action_probabilities)
            # Take act. get reward. next state
            next_state, reward, done, _ = env.step(action)
            # Update stats.
            stats.episode_rewards[ith_ep] += reward
            stats.episode_lengths[ith_ep] = t
            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_fac * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            if done:
                break
            state = next_state
    return Q, state

# Train model              # Num episodes
Q, stats = QLearning(env, 1000)
# Plot important stats.
plotting.plot_episode_stats(stats)

