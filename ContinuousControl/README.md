# Project Details
This project uses a Deep Deterministic Policy Gradient (DDPG) Network to create a control for a virtual robotic reacher. There are 33 observations for the state used by the actor network to provide 4 actions that correspond to rotations in each of the two joints but unlike DQN they represent a continuous space. The code here was developed natively with Python 3.6.3 on a 64-bit Windows 8.1 system however the Unity environments for 32-bit Windows, Linux and Mac are included in the repository. Pytorch was used for the neural network components, unityagents was used to integrate with the Unity model and the standard libraries numpy, collections, matplotlib and random were used as well. 

# Train the model
In order to train the agent execute the file train.py. The environment is considered solved when scores greater than 30 are obtained for 100 consecutive episodes.

# Run the model
In order to view the behavior of the target networks execute the file test.py

