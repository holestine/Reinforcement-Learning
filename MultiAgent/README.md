# Project Details
This project uses a Multi Agent Deep Deterministic Policy Gradient (MADDPG) Network to create controllers for virtual tennis players. There are 24 observations for the state used by the actor network which provides 2 actions that correspond to position from the net and the ground. The code here was developed natively with Python 3.6.3 on a 64-bit Windows 8.1 system. 

# Environment
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

# Dependencies
All files needed to run train.py and test.py are in the repository however you will need to install the Python libraries: pytorch, unityagents, numpy, collections, matplotlib, copy and random (e.g., pip install pytorch).

# Train the model
In order to train the agent execute the file train.py.

# Run the model
In order to view the behavior of the target networks execute the file test.py

