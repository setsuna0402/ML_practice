'''
Define the Policy Gradient Agent class and the network architecture
Author: Dr. Ka Hou Leong
Date: 28/1/2026
Version: 0.1
ML library: PyTorch
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Define the agent network
class PolicyNetwork(nn.Module):
    # In the Lunar Lander environment, the state space has 8 dimensions and action space has 4 discrete actions
    def __init__(self, state_space=8, action_space=4):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        # dropout layer with p=0.25
        self.dropout = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
    
# Define the Policy Gradient Agent
# allow_auto_device = True: try to use GPU if available False: use CPU
class PolicyGradientAgent():
    def __init__(self, network, learning_rate=0.001, gamma=0.99, allow_auto_device=False):
        self.network = network
        # self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        if allow_auto_device:
            # Move the network to the appropriate device
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("Using GPU")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS")
            else:
                device = torch.device("cpu")
                print("Using CPU")
        else:
            device = torch.device("cpu")
        self.device = device
        self.network.to(self.device) # Move the network to the appropriate device
        self.log_probs = []
        self.rewards = []

    def forward(self, state):
        state_d = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.network(state_d)

    def learn_trivial(self, log_probs, rewards):
        ### parameters of action don't affect the prob(s_t+1|s_t, a_t) ###
        ### So, we only include prob(a_t|s_t) in the calculation of loss ###
        # Here, we consider the total reward for each trajectory: 
        # expectation value of reward = sum(trajectory's total reward * probability of the trajectory) / N
        # This is the worst method, because it treats all actions in the trajectory equally.

        # log_probs: 2D list of log probabilities of actions taken [i][j]: i-th trajectory, j-th action
        # rewards: 2D list of rewards for each action taken [i][j]: i-th trajectory, j-th action
        '''
        if self.device.type != "cpu":
            print("learn_trivial supports only cpu! Going to terminate!")
            exit(1)
        '''
        #Warning: No normalisation
        losses = []
        for ep_logps, ep_rewards in zip(log_probs, rewards):
            ep_logps = torch.stack(ep_logps)  # list [tensor_scale_a, tensor_scale_b, ..., tensor_scale_c] -> 1D Tensor [a, b, c, ..., ]
            ep_total_reward = sum(ep_rewards)  # scalar
            ep_returns = torch.ones_like(ep_logps, device=self.device) * ep_total_reward  # 1D Tensor [R, R, R, ..., R]
            losses.append(-(ep_logps * ep_returns).sum())
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, log_probs, rewards):
        # This is the default learn function
        # log_probs and rewards are both 2D lists (from all trajectories concatenated)
        # Convert lists to flat tensors
        N_trajectories = len(log_probs)
        log_probs_flat = torch.cat([torch.stack(ep) for ep in log_probs]).to(self.device)
        rewards_flat   = torch.tensor([r for ep in rewards for r in ep], dtype=torch.float32, device=self.device)
        # Normalise rewards (for the default learn function)
        rewards_flat = (rewards_flat - rewards_flat.mean()) / (rewards_flat.std() + 1e-9)
        loss = -torch.sum(log_probs_flat * rewards_flat) / N_trajectories
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        state_d = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_prob = self.network(state_d)
        action_dist = Categorical(action_prob)  # Create a categorical distribution over the action probabilities
        action = action_dist.sample()  # Sample an action from the distribution
        log_prob = action_dist.log_prob(action) # Get the log probability of the sampled action
        return action.item(), log_prob

    def save(self, PATH): # You should not revise this
        # Move the network back to the CPU before saving
        self.network.to(torch.device("cpu"))
        Agent_Dict = {
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH): # You should not revise this
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Move the network back to the appropriate device after loading
        self.network.to(self.device)