from unityagents import UnityEnvironment
import numpy as np
from ddpg_agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt

# Get the Unity environment for vector observations
env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations               # Initialize the state
agent = Agent(state_size=len(state[0]), action_size=brain.vector_action_space_size, random_seed=0)

def run(actor, critic):
    global agent, env, state
    score = 0                                          # Initialize the score

    for j in range(100):
        action = agent.act(state, add_noise=False)     # get the action from the agent
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations      # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # Exit loop if episode finished
            break

    agent.actor_local.load_state_dict(torch.load(actor))  # load checkpoints
    agent.critic_local.load_state_dict(torch.load(critic))

    for i in range(10):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        for j in range(100):
            action = agent.act(state, add_noise=False)     # get the action from the agent
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations      # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # Exit loop if episode finished                

                break
    
    print("Score: {}".format(score))

run('actor.pth', 'critic.pth')

# Close the environment
env.close()
