from unityagents import UnityEnvironment
import numpy as np
from time import sleep
from dqn_agent import Agent
from collections import deque
import torch

# Get the Unity environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

score = 0                                          # initialize the score
state = env_info.vector_observations[0]            # initialize the state
agent = Agent(state_size=len(state), action_size=brain.vector_action_space_size, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('best.pth'))
while True:
    action = int(agent.act(state))
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    sleep(0.2)
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))

# Close the environment.
env.close()
