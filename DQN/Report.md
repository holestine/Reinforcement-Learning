# Project Details
This project uses a Deep Q-Network (DQN) to create a controller that collects bananas in a virtual environment. 

# Environment
 A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

# Learning Algorithm
This project uses a Deep Q-Network to navigate an entity through the Unity Banana environment. There are 4 possible actions for the entity (left, right, forward and backward) and 37 observations for the state. The underlying neural network has a fully connected architecture with 37 inputs, 4 outputs and two hidden layers each with 64 neurons. The ReLU activation function is used in all layers with a learning rate of .0005. The DQN algorithm runs 1000 episodes for a maximum of 1000 time steps and uses an epsilon greedy policy with an initial value of 1 a minimum value of .01 and a decay rate of .99.

# Reward History
The total rewards for each episode are shown in the diagram below where the maximum score was 27 and the last 100 had an average of approximatelly 16. The average score exceeded 13 after about 400 time steps but was allowed to run until all episodes completed or an average score of 20 was obtained (this never happened).
![Training Profile](images/training.png)


# Improvements
The design could benift by using the pixels as input to a convolutional neural network rather than using the 37 state variables available from the agent. Could also use additional frames to allow the network to learn from changes in position as well as experience replay to reduce the temporal dependence. 