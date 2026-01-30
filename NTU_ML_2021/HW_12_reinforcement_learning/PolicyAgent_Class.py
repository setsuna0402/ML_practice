'''
Define the PolicyGradientAgent class and the ActorCriticAgent class
Define the PolicyNetwork class and ValueNetwork class
Author: Dr. Ka Hou Leong
Date: 31/1/2026
Version: 0.2
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
        x = self.fc3(x) # shape [batch, 4]
        return F.softmax(x, dim=-1) #dim = -1: softmax over the last dimension

# Define the value function network
class ValueNetwork(nn.Module):
    def __init__(self, state_space=8):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        # dropout layer with p=0.25
        self.dropout = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)          # shape [batch, 1] or [1]
        return x.squeeze(-1)     # shape [batch]

# Define the Policy Gradient Agent
# allow_auto_device = True: try to use GPU if available False: use CPU
class PolicyGradientAgent():
    def __init__(self, actor_network, learning_rate=0.001, gamma=0.99, allow_auto_device=False):
        self.actor_network = actor_network
        # self.actor_optimizer = optim.SGD(self.actor_network.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
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
        self.actor_network.to(self.device) # Move the network to the appropriate device
        self.log_probs = []
        self.rewards = []

    def forward(self, state):
        state_d = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.actor_network(state_d)

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
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

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
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def sample(self, state):
        state_d = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action_prob = self.actor_network(state_d)
        action_dist = Categorical(action_prob)  # Create a categorical distribution over the action probabilities
        action = action_dist.sample()  # Sample an action from the distribution
        log_prob = action_dist.log_prob(action) # Get the log probability of the sampled action
        return action.item(), log_prob

    def save(self, PATH): # You should not revise this
        # Move the network back to the CPU before saving
        self.actor_network.to(torch.device("cpu"))
        Agent_Dict = {
            "network" : self.actor_network.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH): # You should not revise this
        checkpoint = torch.load(PATH)
        self.actor_network.load_state_dict(checkpoint["network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        # Move the network back to the appropriate device after loading
        self.actor_network.to(self.device)

# Policy Gradient Agent with Value Function (actor-critic method)
class ActorCriticAgent(PolicyGradientAgent):
    def __init__(self, actor_network, value_network, learning_rate=0.001, gamma=0.99, allow_auto_device=False):
        super().__init__(actor_network, learning_rate, gamma, allow_auto_device)
        self.value_network = value_network
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

    def get_value(self, state):
        state_d = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.value_network(state_d)

    # learn with value function (MC method)
    def learn_actor_critic_mc(self, log_probs, rewards, states):
        # This is learn function involving with value function
        # The estimation of the value function uses Monte Carlo method  
        # log_probs, rewards, states are all 2D lists (from all trajectories concatenated)
        # Convert lists to flat tensors
        N_trajectories = len(log_probs)
        # flatten all steps from all episodes
        log_probs_flat = torch.cat([torch.stack(ep) for ep in log_probs])  # [T]
        # torch tensor detach the object from the graph and prevent gradients from flowing back
        states_flat = torch.tensor([st for ep in states for st in ep], dtype=torch.float32, device=self.device)  # [T,8]
        rewards_flat = torch.tensor([g for ep in rewards for g in ep], dtype=torch.float32, device=self.device)  # [T]

        # critic prediction
        values_flat = self.value_network(states_flat)   # [T]

        # advantage (detach values_flat so actor doesn't update critic)
        advantages_flat = rewards_flat - values_flat.detach()

        # losses
        # expectation value of rewards over trajectories
        actor_loss  = -(log_probs_flat * advantages_flat).sum() / N_trajectories
        # minimize the difference between predicted values and actual rewards
        critic_loss = (values_flat - rewards_flat).pow(2).sum() / N_trajectories

        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update critic
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()

    # Override save() from parent class PolicyGradientAgent
    def save(self, PATH): # You should not revise this
        # Move the network back to the CPU before saving
        # super().save(PATH)
        self.actor_network.to(torch.device("cpu"))
        self.value_network.to(torch.device("cpu"))
        Agent_Dict = {
            "actor_network" : self.actor_network.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "value_network" : self.value_network.state_dict(),
            "value_optimizer" : self.value_optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    # Override load() from parent class PolicyGradientAgent
    def load(self, PATH): # You should not revise this
        checkpoint = torch.load(PATH)
        self.actor_network.load_state_dict(checkpoint["actor_network"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        # Move the network back to the appropriate device after loading
        self.actor_network.to(self.device)
        self.value_network.to(self.device)