# Project Details
This project uses a Multi Agent Deep Deterministic Policy Gradient (MADDPG) Network to create controllers for virtual tennis players. There are 24 observations for the state used by the actor network which provides 2 actions that correspond to position from the net and the ground. The code here was developed natively with Python 3.6.3 on a 64-bit Windows 8.1 system. 

# Dependencies
Install these libraries pytorch, unityagents, numpy, collections, matplotlib, copy and random. If you want more info than this in the write up can you point me to an example? I do have a full time job complete with customers and would appreciate a little more information on what is expected. Thanks

# Rewards
The agent gets a reward of +.1 when hitting the ball over the net and -.1 otherwise. The environment is considered solved when one of the agents maintains an average score of at least 0.5 over 100 episodes.

# Train the model
In order to train the agent execute the file train.py.

# Run the model
In order to view the behavior of the target networks execute the file test.py

