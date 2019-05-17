import gym
import numpy as np
# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')

v_reward = []
for i in range(10000):
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    is_done = False
    tot_reward = 0
    print(f"ep:{i}")
    while not is_done:

        frame, reward, is_done, _ = env.step(env.action_space.sample())
        tot_reward += reward
        #env.render()
    v_reward.append(tot_reward)
    #print(tot_reward)

env.close()
print(np.mean(v_reward), np.var(v_reward))
