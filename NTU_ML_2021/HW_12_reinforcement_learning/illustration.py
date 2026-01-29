'''
Illustration of the trained Lunar Lander policy gradient agent

Author: Dr. Ka Hou Leong
Date: 28/1/2026
Version: 0.1
ML library: PyTorch
'''
import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from PolicyAgent_Class import PolicyGradientAgent, PolicyNetwork


# Define environment
env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("LunarLander-v3")

# Create the agent
network = PolicyNetwork()
agent = PolicyGradientAgent(network, allow_auto_device=False)
agent.load("./lunar_lander_policy_gradient_agent_num_batch_800_episodes_10_gamma_0.999.pth")

#Reset the environment to generate the first observation
observation, info = env.reset()

# Set the agent to evaluation mode
agent.network.eval()
# Set num_episodes for evaluation
num_episodes = 5

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    while not done:
        action, _ = agent.sample(observation)
        next_state, reward, terminated, truncated, info = env.step(action)
        observation = next_state
        done = terminated or truncated
        if done:
            print("Final reward: {}".format(reward))
            # time.sleep(1)
            break
