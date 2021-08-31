import time

import gym

env = gym.make('Farm_Env-v0')
env.reset()
env.render()

time.sleep(5)