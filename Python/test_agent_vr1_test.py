import gym
import matplotlib.pyplot as plt

# Create a breakout environment
env = gym.make('MsPacman-v0')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False
data = []
while not is_done:
    # Perform a random action, returns the new frame, reward and whether the game is over

    frame, reward, is_done, _ = env.step(env.action_space.sample())
    print(env.action_space.sample())
    data.append(reward)
  # Render
    env.render()

plt.plot(data)
plt.show()