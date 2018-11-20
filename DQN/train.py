from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt

# Get the Unity environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# Set the brain we control from Python to the first one available
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
print('Number of agents:', len(env_info.agents))

# Number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# Examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# Set up agent with appropriate sizes for the state and action spaces
agent = Agent(state_size=state_size, action_size=action_size, seed=0)


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    max_score = 13;                    # Track max score in order to save best performing network
    scores = []                        # List containing scores from each episode
    scores_window = deque(maxlen=100)  # Last 100 scores
    eps = eps_start                    # Initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = int(agent.act(state, eps))            # Get the next action
            env_info = env.step(action)[brain_name]        # Send the action to the environment
            next_state = env_info.vector_observations[0]   # Get the next state
            reward = env_info.rewards[0]                   # Get the reward
            done = env_info.local_done[0]                  # See if the episode has finished
            agent.step(state, action, reward, next_state, done) # Send training information to DQN algorithm
            state = next_state          # Transition to next state
            score += reward             # Track the score
            if done:
                break 
        scores_window.append(score)       # save most recent score to moving window
        scores.append(score)              # save most recent score to complete list of scores
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if score >= max_score:
            max_score = score
            torch.save(agent.qnetwork_local.state_dict(), 'agent.pth')
        if np.mean(scores_window) > 13:
            break
    return scores

scores = dqn(1000, 1000, 1, .01, .99)

# Plot and save the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig("images/training.png")
plt.show()

# Close the environment.
env.close()
