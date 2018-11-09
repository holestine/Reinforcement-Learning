# Project Details
This project uses a Multi Agent Deep Deterministic Policy Gradient (MADDPG) Network to create a controllers for virtual tennis players. There are 24 observations for the state used by the actor network to provide 2 actions that correspond to position from the net and the ground. The code here was developed natively with Python 3.6.3 on a 64-bit Windows 8.1 system. Pytorch was used for the neural network components, unityagents was used to integrate with the Unity environment and the standard libraries numpy, collections, matplotlib and random were used as well. 

# Train the model
In order to train the agent execute the file train.py. The environment is considered solved when scores greater than 0.5 are obtained for 100 consecutive episodes.

# Run the model
In order to view the behavior of the target networks execute the file test.py

