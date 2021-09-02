import time
import random
from math import pi

import gym

env = gym.make('Farm_Env-v0')
action = random.uniform(0, pi)

env.reset()

for i in range(500):
    env.step(env.action_space.sample())
    env.render()
    time.sleep(0.01)

env.close()
