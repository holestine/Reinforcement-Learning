# Project Details
This project uses a Deep Q-Network to navigate an entity through the Unity Banana environment. There are 4 possible actions for the entity (left, right, forward and backward) and 37 observations for the state. The code here was developed natively with Python 3.6.3 on a 64-bit Windows 8.1 system however the Unity environments for 32-bit Windows, Linux and Mac are included in the repository. Pytorch was used for the neural network components, unityagents was used to integrate with the Unity model and the standard libraries numpy, collections, matplotlib and random were used as well. 

# Train the model
In order to train the agent execute the file train.py

# Run the model
In order to view the behavior of the most fit network execute the file test.py
