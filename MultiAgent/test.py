from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# Get the Unity environment
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# Initialize the environment and get all necessary parameters
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=False)[brain_name] 
num_agents = len(env_info.agents)
state_size = len(env_info.vector_observations[0])
action_size = env.brains[brain_name].vector_action_space_size

agent = []

def run(time=200):
    global env
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations

    for t in range(time):
        action = []

        # Get action(s)
        for i in range(num_agents):
            action.append(agent[i].act(np.reshape(state[i], ((1,state_size))), add_noise=False))              
        
        # Send the action(s) to the environment
        env_info = env.step(np.reshape(action, (action_size*num_agents, 1)))[brain_name]     
        
        # Update state
        state = env_info.vector_observations

        #if any(env_info.local_done):                      # see if episode finished
        #    break

def start_demo(actors, critics):
    # Run with random agents
    for i in range(num_agents):
        agent.append(Agent(state_size=state_size, action_size=action_size, random_seed=i))
    run()

    # Run with saved checkpoints
    for i in range(num_agents):
        agent[i].actor_local.load_state_dict(torch.load(actors[i]))  
        agent[i].critic_local.load_state_dict(torch.load(critics[i]))
    run()

start_demo(['actor0.pth', 'actor1.pth'], ['critic0.pth', 'critic1.pth'])

# Close the environment
env.close()
