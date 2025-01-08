import gymnasium as gym
import random
import numpy as np
from echo_gym_trivial import echo_register
from gymnasium.envs.registration import register, registry

print(gym.envs.registration.registry)
env_id = 'EchoEnv-v0'
echo_register()
print(gym.envs.registration.registry)
env = gym.make('EchoEnv-v0')

# Simulation parameters
EPISODES = 1
run = 0
all_steps = 0
Continue = True
while Continue:
    run += 1
    random.seed(run)
    np.random.seed(run)

    # Reset environment
    state, info = env.reset(seed=run)
    terminated = False
    step = 0

    while not terminated:
        step += 1

        # Sample random action
        action = env.action_space.sample()

        # Take action
        state, reward, terminated, truncated, info = env.step(action)

        if terminated:
            if run == EPISODES:
                Continue = False
