# Introduction
I've gone through the DDPG paper a few times and tried a variety of things but haven't been able to get the required score. So here's what I've done so far, hopefully I can get some feedback on what I'm overlooking.

# Project Details
I started this project with the Bipedal DDPG code from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal. The Actor NN is fully connected with 33 inputs describing state information, a hidden layer with 256 neurons and an output layer with 4 neurons, the rectified linear unit is used internally and hyperbolic tangent is used on the output (I have used larger networks but this size seem sufficient if it solves bipel environments). The Critic Network has three hidden layers of size 256, 256 adn 128, it takes the state information as input but concatenates that with the actions at the second hidden layer, leaky rectified linear units are used throughout. I added batch normalization to the input of each network and added a variable to keep track of the number of stpes so training could be performed every 20 steps with a batch size of 128. I've also done some expimenting with the noise using different combinations of Ornstein-Uhlenbeck and Gaussian, right now I alternate between the two every step. 
![Hyperparameters](images/hyper.png)

# Reward History
This is representative of what I'm currently seeing.
![Training Profile](images/training.png)

# Improvements
I'll update this when I get better results.
