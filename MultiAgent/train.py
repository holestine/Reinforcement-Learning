from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent

from buffer import ReplayBuffer, BUFFER_SIZE, BATCH_SIZE

# Get the Unity environment for vector observations
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
#env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

# Get the Unity brain
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

# State properties
state = env_info.vector_observations
state_size = len(state[0])
print('States have length:', state_size)
print('States look like:', state[0])

# Set up agent with appropriate sizes for the state and action spaces
agent = []
memory = []
for i in range(num_agents):
    agent.append(Agent(state_size=state_size, action_size=action_size, random_seed=i))
    memory.append(ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, i))

def ddpg(n_episodes=1000, max_t=1000, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = []
        for i in range(num_agents):
            agent[i].reset()
            score.append(0)

        for t in range(max_t):
            action = []
            for i in range(num_agents):
                action.append(agent[i].act(np.reshape(state[i], ((1,state_size)))))              # Get noisy action
            
            env_info = env.step(np.reshape(action, (action_size*num_agents, 1)))[brain_name]     # send the action to the environment
            next_state = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                        # get reward (for each agent)

            for i in range(num_agents):
                memory[i].add(state[i], action[i], rewards[i], next_state[i], env_info.local_done[i])
                agent[i].step(memory[i])
                score[i] += rewards[i]                        # update the score (for each agent)

            state = next_state                                # roll over to next time step

            #if any(env_info.local_done):                      # see if episode finished
            #    break

        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        
        if scores[len(scores)-1] == np.max(scores):
            torch.save(agent[0].actor_target.state_dict(), 'actor0.pth')
            torch.save(agent[0].critic_target.state_dict(), 'critic0.pth')
            torch.save(agent[1].actor_target.state_dict(), 'actor1.pth')
            torch.save(agent[1].critic_target.state_dict(), 'critic1.pth')

        print('\rEpisode {} \tAverage Score: {:.2f} \tMax Score: {:.2f} \tLast Score: {:.2f}'.format(i_episode, np.mean(scores_deque), np.max(scores), scores[-1]), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {} \tAverage Score: {:.2f} \tMax Score: {:.2f} \tLast Score: {:.2f}'.format(i_episode, np.mean(scores_deque), np.max(scores), scores[-1]))

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
