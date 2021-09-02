'''
This is a environment for RL
'''
import time

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
from math import cos, pi, sqrt, fabs, pow, acos, sin
from gym.envs.classic_control import rendering

####################################hyper parameters#################################
INIT_V = 30      # velocity of dog
INIT_v = 20    # velocity of sheep
RADIUS = 200     # radius of farmland
DELTA_T = 0.1   # interval
#####################################################################################
class Dog(object):
    def __init__(self):
        self.V = INIT_V
        self.pos = np.array([RADIUS, random.uniform(0,2) * pi])

    def updatePos(self, action, pre_sheep, cur_sheep):
        # do some math
        b = -2 * cos(action + pi / 2) * pre_sheep[0]
        c = pow(pre_sheep[0], 2) - pow(RADIUS, 2)
        distance = (-b - sqrt(pow(b, 2) - 4 * c)) / (4 * c)
        cos_gamma = (pow(pre_sheep[0], 2) + pow(RADIUS, 2) - pow(distance, 2)) / (2 * RADIUS * pre_sheep[0])
        if (cos_gamma > 1):
            cos_gamma = 1
        gamma = acos(cos_gamma)

        pre_angle = fabs(pre_sheep[1] - self.pos[1])
        if (pre_angle > pi):
            pre_angle = 2 * pi - pre_angle
        cur_angle = fabs(cur_sheep[1] - self.pos[1])
        if (cur_angle > pi):
            cur_angle = 2 * pi - cur_angle
        # dog choose its action
        print(cur_sheep[0], end='\t')

        if (self.pos[1] < pi):
            if (pre_sheep[1] > self.pos[1] and pre_sheep[1] <= self.pos[1] + pi):
                if (pre_angle > cur_angle):
                    angle_pos = fabs(pre_sheep[1] - gamma)
                else:
                    angle_pos = pre_sheep[1] + gamma
                    if (angle_pos > 2 * pi):
                        angle_pos = 2 * pi - angle_pos
                if (angle_pos > self.pos[1] and angle_pos <= self.pos[1] + pi):
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                    print('1')
                else:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                    print('2')
            elif (pre_sheep[1] <= self.pos[1] and pre_sheep[1] > 0 or pre_sheep[1] > self.pos[
                1] + pi and pre_sheep[1] <= 2 * pi):
                if (pre_angle > cur_angle):
                    angle_pos = fabs(pre_sheep[1] - gamma)
                else:
                    angle_pos = pre_sheep[1] + gamma
                    if (angle_pos > 2 * pi):
                        angle_pos = 2 * pi - angle_pos
                if (pre_sheep[1] <= self.pos[1] and pre_sheep[1] > 0 or pre_sheep[1] > self.pos[
                    1] + pi and pre_sheep[1] <= 2 * pi):
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                    print('3')
                else:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                    print('4')
            else:
                print('error')
        else:
            if (pre_sheep[1] > self.pos[1] and pre_sheep[1] <= 2 * pi or pre_sheep[1] <= self.pos[
                1] - pi and pre_sheep[1] > 0):
                if (pre_angle > cur_angle):
                    angle_pos = fabs(pre_sheep[1] - gamma)
                else:
                    angle_pos = pre_sheep[1] + gamma
                    if (angle_pos > 2 * pi):
                        angle_pos = 2 * pi - angle_pos
                if (pre_sheep[1] > self.pos[1] and pre_sheep[1] <= 2 * pi or pre_sheep[1] <= self.pos[
                    1] - pi and pre_sheep[1] > 0):
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                    print('5')
                else:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                    print('6')
            elif (pre_sheep[1] > self.pos[1] - pi and pre_sheep[1] <= self.pos[1]):
                if (pre_angle > cur_angle):
                    angle_pos = pre_sheep[1] + gamma
                    if (angle_pos > 2 * pi):
                        angle_pos = 2 * pi - angle_pos
                else:
                    angle_pos = fabs(pre_sheep[1] - gamma)
                if (pre_sheep[1] > self.pos[1] - pi and pre_sheep[1] <= self.pos[1]):
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                    print('7')
                else:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                    print('8')
            else:
                print('error')
        print(self.pos[1])
        # Judge whether the angle is out of bounds
        if (self.pos[1] > 2 * pi):
            self.pos[1] = self.pos[1] - 2 * pi

        if ((fabs(cur_sheep[1] - self.pos[1])) >= pi):
            included_angle = fabs(cur_sheep[1] - self.pos[1])
        else:
            included_angle = 2 * pi - fabs(cur_sheep[1] - self.pos[1])
        # return next state
        return np.array([cur_sheep[0], included_angle])

    def resetPos(self):
        self.pos = np.array([RADIUS, random.uniform(0, 2) * pi])

class Sheep(object):
    def __init__(self):
        self.v = INIT_v
        self.pos = np.array([random.uniform(0.001,1) * RADIUS, random.uniform(0,2) * pi])

    def updatePos(self,action):
        self.pre_rho = self.pos[0]
        self.pre_theta = self.pos[1]
        if (action < pi / 2):
            self.pos[0] = sqrt(
                pow(self.v * cos(action) * DELTA_T, 2) + pow(self.pos[0], 2) - 2 * self.pos[0] * cos(
                    action) * DELTA_T * cos((pi / 2) + action))
            self.pos[1] = self.pos[1] - acos(
                (pow(self.pos[0], 2) + pow(self.pre_rho, 2) - pow(self.v * cos(action) * DELTA_T, 2)) / (
                            2 * self.pos[0] * self.pre_rho))
        elif (action > pi / 2):
            self.pos[0] = sqrt(
                pow(self.v * cos(pi - action) * DELTA_T, 2) + pow(self.pos[0], 2) - 2 * self.pos[0] * cos(
                    pi - action) * DELTA_T * cos((pi / 2) + pi - action))
            self.pos[1] = self.pos[1] + acos(
                (pow(self.pos[0], 2) + pow(self.pre_rho, 2) - pow(self.v * cos(pi - action) * DELTA_T, 2)) / (
                            2 * self.pos[0] * self.pre_rho))
        else:
            self.pos[0] += self.v * DELTA_T

        if (self.pos[1] > 2 * pi):
            self.pos[1] = self.pos[1] - 2 * pi

    def resetPos(self):
        self.pos = np.array([random.uniform(0.001,1) * RADIUS, random.uniform(0,2) * pi])

class Farm_Env(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = rendering.Viewer(600, 600)
        # dog initialization
        self.dog = Dog()
        # sheep initilaization
        self.sheep = Sheep()
        # action the sheep can take, the degree it turns
        self.action_space = Box(low = np.array([0]), high = np.array([pi]), dtype = np.float32) # todo:check the method
        # the distance between sheep and the boarder of farmland , the angle bwtween sheep and dog
        self.observation_space = Box(low = np.array([0, 0]), high = np.array([RADIUS, pi]), dtype = np.float32)
        ####
        self.seed()

    def step(self, action):
        # apply action
        ## update sheep's position
        self.pre_sheep = self.sheep.pos
        self.sheep.updatePos(action)
        self.cur_sheep = self.sheep.pos
        ## update dog's position and get next state
        state = self.dog.updatePos(action, self.pre_sheep, self.cur_sheep)
        # Calculate Reward
        reward = 0.5 * (state[0] - RADIUS/2)/RADIUS + 0.5 * (state[1] - pi/2)/pi
        # check if game is done
        if(self.sheep.pos[0] > RADIUS):
            done = True
        else:
            done = False
        # define empty info
        info = {}
        return state, reward, done, info

    def reset(self):
        # reset dog
        self.dog.resetPos()
        # reset sheep
        self.sheep.resetPos()
        # reset state
        if ((fabs(self.sheep.pos[1] - self.dog.pos[1])) >= pi):
            included_angle = fabs(self.sheep.pos[1] - self.dog.pos[1])
        else:
            included_angle = 2 * pi - fabs(self.sheep.pos[1] - self.dog.pos[1])
        # wrap state into array
        state = np.array([self.sheep.pos[0], included_angle])
        # return the new initial state
        return state

    def render(self, mode='human'):
        self.viewer.geoms.clear()
        # 方式一
        ring = rendering.make_circle(radius=200,
                                     res=50,
                                     filled=False)
        # radius=10 半径
        # res=30    说是画圆，其实是画正多边形，res指定多边形的边数
        # filled=True   是否填充
        ring.set_color(0, 0, 0)
        ring.set_linewidth(5)  # 设置线宽
        sheep = rendering.make_circle(radius=5,
                                     res=50,
                                     filled=True)
        dog =  rendering.make_circle(radius=5,
                                      res = 50,
                                      filled = True)

        sheep.set_color(0, 0, 255)
        dog.set_color(255, 0, 0)

        # 添加一个平移操作
        transform1 = rendering.Transform(translation=(300, 300))  # 相对偏移
        transform_s = rendering.Transform(translation=(300 + self.sheep.pos[0] * cos(self.sheep.pos[1]), 300 + self.sheep.pos[0] * sin(self.sheep.pos[1])))
        transform_d = rendering.Transform(translation=(300 + self.dog.pos[0] * cos(self.dog.pos[1]), 300 + self.dog.pos[0] * sin(self.dog.pos[1])))
        # 让圆添加平移这个属性
        ring.add_attr(transform1)
        sheep.add_attr(transform_s)
        dog.add_attr(transform_d)

        self.viewer.add_geom(ring)
        self.viewer.add_geom(sheep)
        self.viewer.add_geom(dog)

        return self.viewer.render(return_rgb_array= mode=='rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None