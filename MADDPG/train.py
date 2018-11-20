from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent

from buffer import ReplayBuffer
BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 1024       # Minibatch size

# Get the Unity environment for vector observations
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

# Get the Unity brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
num_agents = len(env_info.agents)
print('\nNumber of agents:', num_agents)

# Number of actions
action_size = brain.vector_action_space_size
print('\nNumber of actions:', action_size)

# State properties
state = env_info.vector_observations
state_size = len(state[0])
print('\nStates have length:', state_size)
print('\nStates look like:', state[0], '\n')

actors, critics = ['actor0.pth', 'actor1.pth'], ['critic0.pth', 'critic1.pth']

def maddpg(n_episodes=50000, max_t=500, print_every=100):
    training_phase = 1
    scores = []
    add_noise=True

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = []
        for i in range(num_agents):   # Initialize scores
            score.append(0)

        for t in range(max_t):
            action = []
            for i in range(num_agents):
                action.append(agent[i].act(np.reshape(state[i], ((1,state_size))), add_noise=add_noise))   # Get noisy action
            
            env_info = env.step(np.reshape(action, (action_size*num_agents, 1)))[brain_name]          # Perform a step in the environment
            next_state = env_info.vector_observations                                                 # Get the new state (for each agent)
            rewards = env_info.rewards                                                                # Get the reward (for each agent)

            for i in range(num_agents):
                memory[i].add(state[i], action[i], rewards[i], next_state[i], env_info.local_done[i]) # Save to replay buffer
                agent[i].step(memory[i])                                                              # Perform a step in the neural network
                score[i] += rewards[i]                                                                # Update the score (for each agent)

            state = next_state                                                                        # Update state

            if any(env_info.local_done):                                                              # Break if episode complete
                break

        scores.append(np.max(score))
        
        # Save weights when score improves
        try:
            if scores[len(scores)-1] == np.max(scores):
                torch.save(agent[0].actor_target.state_dict(), actors[0])
                torch.save(agent[0].critic_target.state_dict(), critics[0])
                torch.save(agent[1].actor_target.state_dict(), actors[1])
                torch.save(agent[1].critic_target.state_dict(), critics[1])
            if training_phase == 1 and np.max(scores) > 1:
                training_phase = 2
                for i in range(num_agents):
                    memory[i] = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, i)
                    agent[i].enable_major_update(False)
                    add_noise=False
        except:
            print("Failed to save weights on episode {}", i_episode)

        # Send status to display
        print('\r Episode {} \tAverage Score: {:.2f} \tMax Score: {:.2f} \tLast Score: {:.2f}'.format(i_episode, np.mean(scores[-100:]), np.max(scores), scores[-1]), end="")
        if i_episode % print_every == 0:
            print('\r Episode {} \tAverage Score: {:.2f} \tMax Score: {:.2f} \tLast Score: {:.2f}'.format(i_episode, np.mean(scores[-100:]), np.max(scores), scores[-1]))

        if np.mean(scores[-100:]) > 1:
            return scores

    return scores

# Set up agent with appropriate sizes for the state and action spaces
agent = []
memory = []

for i in range(num_agents):
    agent.append(Agent(state_size=state_size, action_size=action_size, batch_size=BATCH_SIZE, random_seed=i))
    memory.append(ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, i))

scores = maddpg()

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
fig.savefig("images/training.png")
plt.show()

# Close the environment.
env.close()
