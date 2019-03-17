# Breakout-v4

Learning to play Breakout through reinforcement learning, implementing the DQN algorithm described in the article '[Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)'.  *This is still very much a work in progress* 



## Changes and todo lists

Here is a little list of things we plan to implement 

#### Code improvements

- [x] Renamed files to remove versioning + todo list. 

- [x] preprocess fn: remove the walls

- [x] only learn every 4th frame

- [x] change game-mode to *breakout_v4*?

- [ ] investigate game-mode  *deterministic* version?

- [ ] Replay memory class implemented in numpy (ints and zipping)

- [ ] numba.jit use investigated

- [ ] slower epsilon greedy decay

- [x] check if files to be loaded exist

- [ ] track learning iterations over runs 

  

#### Data management & plots

- [x] data folder with checks in place 

- [ ] restructure what we save: sample 2000 states and document the average of the $q_{max}$ over all those states. save into csv 

- [ ] box plots added

- [ ] average q plotted

- [ ] data gathered indexed by learning iterations not episodes 

  

## File structure (outdated)

The main algorithm is started from **DQN_algorithm.py** which in turn loads a few helper files.

* DQN_agent 
* NN_v1
* preprocess_BO

## Dependencies

To be filled in soon.

## Authors

Martti Yap

Gabriel Andersson



