# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:00:56 2025

@author: Orhan
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython import display

env = gym.make("CartPole-v1", render_mode = "human")
device = "cpu"

Transition = namedtuple("Transition", 
                        ("state", "action", "next_state", "reward"))  

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
# %% Create DQL Model    

class DQN(nn.Module):
    
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# %% Hyperparameters

batch_size = 128
gamma = 0.99 # discount factor
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000 
tau = 0.005 # update rate of target network
lr = 1e-4


n_actions = env.action_space.n

state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
   
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations(show_result = False):
    
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype = torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training ...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    
    if len(durations_t) > 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        
    plt.pause(0.001)
    
    display.display(plt.gcf())
    display.clear_output(wait = True)

def optimize_model():
    
    
    
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype = torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_state_action_values = (next_state_values*gamma) + reward_batch
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# %% Training Model and Evaluation

num_episodes = 400

for i_episode in range(num_episodes):
    
    # reset env
    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        action = select_action(state) # action secme
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated # keybettik yada yandik        
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype = torch.float32, device=device).unsqueeze(0)
            
      
        memory.push(state, action, next_state, reward)
            
        # Update State
        state = next_state
        
        # training 
        optimize_model()
        
        # update network parameters
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
        
print("Done")
plot_durations(show_result=True)
plt.ioff()
plt.show()






