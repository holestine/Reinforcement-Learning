from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent

from buffer import ReplayBuffer, BUFFER_SIZE, BATCH_SIZE

# Get the Unity environment for vector observations
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# Initialize the environment and get all necessary parameters
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=False)[brain_name] 
num_agents = len(env_info.agents)
state = env_info.vector_observations 
state_size = len(state[0])
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

agent = []
memory = []
for i in range(num_agents):
    agent.append(Agent(state_size=state_size, action_size=action_size, random_seed=i))
              
def run(actors, critics):
    global agent, env, state

    for j in range(5000):
        action = []
        for i in range(num_agents):
            action.append(agent[i].act(np.reshape(state[i], ((1,state_size)))))              # Get action
            
        env_info = env.step(np.reshape(action, (action_size*num_agents, 1)))[brain_name]     # send the action to the environment
        state = env_info.vector_observations

        #if any(env_info.local_done):                      # see if episode finished
        #    break

#    agent.actor_local.load_state_dict(torch.load(actor))  # load checkpoints
#    agent.critic_local.load_state_dict(torch.load(critic))
#
#    for i in range(10):
#        env_info = env.reset(train_mode=False)[brain_name]
#        state = env_info.vector_observations
#        for j in range(100):
#            action = agent.act(state, add_noise=False)     # get the action from the agent
#            env_info = env.step(action)[brain_name]        # send the action to the environment
#            next_state = env_info.vector_observations      # get the next state
#            reward = env_info.rewards[0]                   # get the reward
#            done = env_info.local_done[0]                  # see if episode has finished
#            score += reward                                # update the score
#            state = next_state                             # roll over the state to next time step
#            if done:                                       # Exit loop if episode finished                
#
#                break

run(['actor0.pth', 'actor1.pth'], ['critic0.pth', 'critic1.pth'])

# Close the environment
env.close()
