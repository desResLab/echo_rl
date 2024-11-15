# Project Overview

This project consists of three main Python files:

1. **echo_gym.py**  
   This file is used to create the Echo environment within the Gymnasium package. It sets up the simulation environment where reinforcement learning can be applied.

2. **agent_DQN.py**  
   This file contains the DQN class, which defines the deep neural network used for optimization tasks. It includes methods for acting, training, and optimizing the policy.

2. **Memory.py**  
   This file contains the memory, which is used to store the (state, action, reward, state_next, terminated) pair for batch learning.
   
4. **interface.py**  
   This script is used to run and observe the environment's behavior. It integrates the Echo environment with the DQN agent to test and visualize the agent's performance.

> Note: Other scripts used for running on the CRC are not included here and can be ignored.

