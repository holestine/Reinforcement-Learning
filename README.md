# Project Details
This project uses a Deep Q-Network to navigate an entity through the Unity Banana environment. There are 4 possible actions for the entity (left, right, forward and backward) and 37 observations for the state which seem to correspond to velocity and the presence of a banana within 10 degree rotations. The code here is intended to run on a Windows environment with the same libraries used thus far in the course, any other dependencies are in this repository. Training is run until an average score of 20 is reached or a total of 2000 episodes have been completed. 

The reward system could be modified to account for the number of time steps between positive rewards for better path planning or perhaps penalizing more for collecting blue bananas to learn avoidance. 

Learning from the pixels because the state information is a simplification, may also include a third person view to expose more of the environment.
