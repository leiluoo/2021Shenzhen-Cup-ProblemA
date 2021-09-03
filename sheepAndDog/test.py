import time
import random
from math import pi

import gym

env = gym.make('Farm_Env-v0')


env.reset()

for i in range(3000):
    env.step(0)

    env.render()
#    time.sleep(0.1)

env.close()
