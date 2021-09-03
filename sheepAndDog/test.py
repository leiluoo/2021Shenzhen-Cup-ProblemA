import time
import random
from math import pi

import gym

env = gym.make('Farm_Env-v0')


env.reset()
cnt = 0
for i in range(3000):
    s, r, d, s_ = env.step(env.action_space.sample())
    env.render()

    print(cnt)
    print(d)
    if d:
        break

    cnt += 1
#    time.sleep(0.1)

env.close()
