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
from PolicyAgent_Class import *


# Define environment
env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("LunarLander-v3")

# Create the agent
actor_network = PolicyNetwork()
value_network = ValueNetwork()
# agent = PolicyGradientAgent(actor_network, allow_auto_device=False)
agent = ActorCriticAgent(actor_network, value_network, allow_auto_device=False)
agent.load("./lunar_lander_actor_critic_mc_method_num_batch_800_episodes_10_gamma_0.995.pth")

#Reset the environment to generate the first observation
observation, info = env.reset()

# Set the agent to evaluation mode
agent.actor_network.eval()
agent.value_network.eval()

# Set num_episodes for evaluation
num_episodes = 5

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    value = agent.value_network(torch.as_tensor(observation, dtype=torch.float32, device=agent.device))
    print("Initial state value: {:.2f}".format(value.item()))
    while not done:
        action, _ = agent.sample(observation)
        next_state, reward, terminated, truncated, info = env.step(action)
        observation = next_state
        done = terminated or truncated
        value = agent.value_network(torch.as_tensor(observation, dtype=torch.float32, device=agent.device))
        # print("Current predicted value: {:.2f}, real reward: {:.2f}".format(value.item(), reward))
        if done:
            print("Final reward: {}".format(reward))
            # time.sleep(1)
            break
