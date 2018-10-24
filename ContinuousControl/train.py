from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# Get the Unity environment for vector observations
env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

# Set the brain we control from Python to the first one available
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# Number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# Examine the state space 
state = env_info.vector_observations
print('States look like:', state)

# Get state size
state_size = len(state[0])
print('States have length:', state_size)

# Set up agent with appropriate sizes for the state and action spaces
agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

def ddpg(n_episodes=300, max_t=5000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)                         # Get noisy action
            env_info = env.step(action)[brain_name]           # send the action to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            reward = env_info.rewards                         # get reward (for each agent)
            agent.step(state, action, reward, next_state, env_info.local_done)
            score += reward[0]                                # update the score (for each agent)
            state = next_state                                # roll over to next time step
            if env_info.local_done[0]:                        # see if episode finished
                break
        if len(scores_deque) > 0 and score > np.max(scores_deque):
            torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_target.pth')
            torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_target.pth') 
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_local.pth') 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'actor.pth')
            torch.save(agent.critic_local.state_dict(), 'critic.pth')
        
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Close the environment.
env.close()
