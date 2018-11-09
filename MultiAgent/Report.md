# Notes to Instructor
At one point I was getting good results on this project but after adding the code to save the weights and doing some clean up I haven't been able to get the same behavior and am not sure what I'm overlooking. Here's the current state, any idea what I should try?

# Project Details
I began this project with the DDPG code I submitted for the continuous control project posted at https://github.com/holestine/Reinforcement-Learning/tree/master/ContinuousControl. The Actor NN is fully connected with 24 inputs describing state information, a hidden layer with 256 neurons and an output layer with 2 neurons, the rectified linear unit is used internally and hyperbolic tangent is used on the output. The Critic Network has three hidden layers of size 256, 256 and 128, it takes state information as input and concatenates that with the actions at the second hidden layer, leaky rectified linear units are used throughout. I kept the batch normalization from the previous project on the input of each network and performed a minor update every 100 steps with a batch size of 1024 and have tried variations of the major update. I've experimented with the Ornstein-Uhlenbeck noise functions but have seen better results using Gaussian with a distribution centered at 0 with a standard deviation of 1. The additional hyperparamters are shown below and are the ones specified in the MADDPG paper. This environment was solved at roughly the tbd episode.

![Hyperparameters](images/hyper.png)

# Reward History

![Training Profile](images/training.png)

# Improvements
I'll update this when I get good results. 
