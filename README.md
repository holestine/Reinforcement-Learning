# Description 
This repository contains sample code for the reinforcement learning algorithms Deep Q-Network (DQN), Deep Deterministic Policy Gradients (DDPG) and Multi-Agent DDPG (MADDPG). The learning is performed using data collected from various Unity environments. All code was developed natively with Python 3.6.3 on a 64-bit Windows 8.1 system although the environments for 32-bit Windows, Linux and Mac are included.

# Dependencies
Install Unity using the instructions at https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md 
Install the Python libraries: pytorch, unityagents, numpy, matplotlib, copy and random (e.g., pip install pytorch).

# Train the model
For each project train the agent(s) by running the file train.py.

# Run the model
To view the behavior of the trained agent(s) run the file test.py which first executes with randomly initialized networks and then loads the saved PyTorch checkpoints to show the results of the training.

