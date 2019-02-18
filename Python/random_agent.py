import gym
import numpy as np
# Create a breakout environment
env = gym.make('MsPacman-v0')

v_reward = []
for i in range(100):
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    #env.render()
    is_done = False
    tot_reward = 0
    while not is_done:

      # Perform a random action, returns the new frame, reward and whether the game is over
      frame, reward, is_done, _ = env.step(env.action_space.sample())
      tot_reward += reward
      # Render
      #env.render()
    v_reward.append(tot_reward)
    #print(tot_reward)

env.close()
print(np.mean(v_reward))
