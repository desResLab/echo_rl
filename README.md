## 📁 Repository Structure

### [`model_plots/`](./model_plots/)
This folder contains scripts to generate visualizations of the echo environment, including:
- Distribution of echo test durations  
- Patient arrival and scheduling patterns

### [`Pre-defined policies/`](./Pre-defined%20policies/)
This folder includes:
- The environment for evaluating **pre-defined policies**
- Dynamic plots of patient flow and resource usage over time

### [`RL_codes/`](./RL_codes/)
This folder includes:
- Reinforcement learning agent training and evaluation code
- Implementation of the custom RL environment
- Support for two settings:
  - **Resource-limited case**
  - **Abundant-resource case**

### [`gymnasium/`](./gymnasium/)
This folder provides a demo of how to integrate the echo environment into the **Gymnasium** package, including:
- Custom environment registration
- Gym-compatible wrappers

### [`policy_comparison/`](./policy_comparison/)
This folder contains code to generate comparative plots for different policies, including:
- Penalty plots of different policies  
- Violin plots of patients’ average waiting times  
- Violin plots of sonographers’ average  
- State-by-state analysis plots  
- Running average reward comparison of policy 1, policy 2, and the RL-based policy

---



