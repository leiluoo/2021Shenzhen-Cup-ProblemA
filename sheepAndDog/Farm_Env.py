'''
This is a environment for RL, which includes a circle acts as a farmland, a point as a dog, a point as a sheep
On this farmland, the dog will chase the sheep to prevent it escape from the farmland, but the dog can only move
on the circumference，while the sheep can only constantly move away from center of the circle.Meanwhile, dog will make the
optimal movement
'''
import time

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
from math import cos, pi, sqrt, fabs, acos, sin
from gym.envs.classic_control import rendering

####################################hyper parameters#################################
INIT_V = 30  # velocity of dog
INIT_v = 20  # velocity of sheep
RADIUS = 200  # radius of farmland
DELTA_T = 0.1  # interval
#####################################################################################
class Dog(object):
    def __init__(self):
        self.V = INIT_V
        self.pos = np.array([RADIUS, random.uniform(0, 2) * pi])

    def updatepos(self, action, pre_sheep, cur_sheep):
        # do some math
        if action > pi/2:
            action = pi - action
        b = -2 * cos(action + pi / 2) * pre_sheep[0]
        c = pre_sheep[0]**2 - RADIUS**2
        distance = (-b + sqrt(b**2 - 4 * c)) / 2
        cos_gamma = (pre_sheep[0]**2 + RADIUS**2 - distance**2) / (2 * RADIUS * pre_sheep[0])
        if cos_gamma > 1:
            cos_gamma = 1
        gamma = acos(cos_gamma)

        pre_angle = fabs(pre_sheep[1] - self.pos[1])
        if pre_angle > pi:
            pre_angle = 2 * pi - pre_angle
        cur_angle = fabs(cur_sheep[1] - self.pos[1])
        if cur_angle > pi:
            cur_angle = 2 * pi - cur_angle
        # dog choose its action
        if self.pos[1] < pi:
            if self.pos[1] < pre_sheep[1] <= self.pos[1] + pi:
                if action <= pi/2:
                    angle_pos = pre_sheep[1] - gamma
                    if angle_pos < 0:
                        angle_pos = 2 * pi + angle_pos
                else:
                    angle_pos = pre_sheep[1] + gamma
                    if angle_pos > 2 * pi:
                        angle_pos = angle_pos - 2 * pi
                if self.pos[1] < angle_pos <= self.pos[1] + pi:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                else:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
            elif self.pos[1] >= pre_sheep[1] >= 0 or self.pos[1] + pi < pre_sheep[1] <= 2 * pi:
                if action <= pi/2:
                    angle_pos = pre_sheep[1] - gamma
                    if angle_pos < 0:
                        angle_pos = 2 * pi + angle_pos
                else:
                    angle_pos = pre_sheep[1] + gamma
                    if angle_pos > 2 * pi:
                        angle_pos = angle_pos - 2 * pi
                if self.pos[1] >= angle_pos >= 0 or self.pos[1] + pi < angle_pos <= 2 * pi:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                else:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
            else:
                if random.uniform(0, 1) > 0.5:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                else:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
        else:
            if self.pos[1] < pre_sheep[1] <= 2 * pi or self.pos[1] - pi >= pre_sheep[1] >= 0:
                if action <= pi/2:
                    angle_pos = pre_sheep[1] - gamma
                    if angle_pos < 0:
                        angle_pos = 2 * pi + angle_pos
                else:
                    angle_pos = pre_sheep[1] + gamma
                    if angle_pos > 2 * pi:
                        angle_pos = angle_pos - 2 * pi
                if self.pos[1] < angle_pos <= 2 * pi or self.pos[1] - pi >= angle_pos >= 0:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                else:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
            elif self.pos[1] - pi < pre_sheep[1] <= self.pos[1]:
                if action > pi/2:
                    angle_pos = pre_sheep[1] + gamma
                    if angle_pos > 2 * pi:
                        angle_pos = angle_pos - 2 * pi
                else:
                    angle_pos = pre_sheep[1] - gamma
                    if angle_pos < 0:
                        angle_pos = 2 * pi + angle_pos
                if self.pos[1] - pi < angle_pos <= self.pos[1]:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
                else:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
            else:
                if random.uniform(0, 1) > 0.5:
                    self.pos[1] = self.pos[1] + self.V / RADIUS * DELTA_T
                else:
                    self.pos[1] = self.pos[1] - self.V / RADIUS * DELTA_T
        # Judge whether the angle is out of bounds
        if self.pos[1] > 2 * pi:
            self.pos[1] = self.pos[1] - 2 * pi
        if self.pos[1] < 0:
            self.pos[1] = self.pos[1] + 2 * pi

        if (fabs(cur_sheep[1] - self.pos[1])) >= pi:
            included_angle = fabs(cur_sheep[1] - self.pos[1])
        else:
            included_angle = 2 * pi - fabs(cur_sheep[1] - self.pos[1])
        # 0 represent that dog is on the sheep's left semi-circle, 1 vice versa
        if cur_sheep[1] < pi:
            if cur_sheep[1] <= self.pos[1] < cur_sheep[1] + pi:
                rotation = 0
            elif cur_sheep[1] + pi <= self.pos[1] < 2 * pi or 0 <= self.pos[1] < cur_sheep[1]:
                rotation = 1
        else:
            if cur_sheep[1] <= self.pos[1] <= 2 * pi or 0 <= self.pos[1] < cur_sheep[1] - pi:
                rotation = 0
            elif cur_sheep[1] - pi <= self.pos[1] < cur_sheep[1]:
                rotation = 1
        # find the included angle between the intersection of pro-longed line of sheep's velocity and dog's position
        # self.angle_p = angle_pos todo:...
        dest_angle = fabs(angle_pos - self.pos[1])
        if dest_angle > pi:
            dest_angle = 2 * pi - dest_angle
        # return next state
        return np.array([cur_sheep[0], included_angle, rotation]), distance, dest_angle

    def resetpos(self):
        self.pos = np.array([RADIUS, random.uniform(0, 2) * pi])


class Sheep(object):
    def __init__(self):
        self.v = INIT_v
        self.pos = np.array([random.uniform(0.001, 1) * RADIUS, random.uniform(0, 2) * pi])

    def updatepos(self, action):
        pre_rho = self.pos[0]
        pre_theta = self.pos[1]
        if action < pi / 2:
            self.pos[0] = sqrt(
                (self.v * cos(action) * DELTA_T)**2 + self.pos[0]**2 - 2 * self.pos[0] * cos(
                    action) * DELTA_T * cos((pi / 2) + action))
            self.pos[1] = self.pos[1] - acos(
                (self.pos[0]**2 + pre_rho**2 - (self.v * cos(action) * DELTA_T)**2) / (
                        2 * self.pos[0] * pre_rho))
        elif action > pi / 2:
            self.pos[0] = sqrt(
                (self.v * cos(pi - action) * DELTA_T)**2 + self.pos[0]**2 - 2 * self.pos[0] * cos(
                    pi - action) * DELTA_T * cos((pi / 2) + pi - action))
            self.pos[1] = self.pos[1] + acos(
                (self.pos[0]**2 + pre_rho**2 - (self.v * cos(pi - action) * DELTA_T)**2) / (
                        2 * self.pos[0] * pre_rho))
        else:
            self.pos[0] += self.v * DELTA_T

        if self.pos[1] > 2 * pi:
            self.pos[1] = self.pos[1] - 2 * pi
        if self.pos[1] < 0:
            self.pos[1] += 2 * pi

    def resetpos(self):
        self.pos = np.array([random.uniform(0.001, 1) * RADIUS, random.uniform(0, 2) * pi])


class Farm_Env(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = rendering.Viewer(600, 600)
        # dog initialization
        self.dog = Dog()
        # sheep initialization
        self.sheep = Sheep()
        # action the sheep can take, the degree it turns
        self.action_space = Box(low=np.array([0]), high=np.array([pi]), dtype=np.float32)  # todo:check the method
        # the distance between sheep and the boarder of farmland , the angle bwtween sheep and dog
        self.observation_space = Box(low=np.array([0, 0, 0]), high=np.array([RADIUS, pi, 1.1]), dtype=np.float32)
        ####
        self.seed()

    def step(self, action):
        # apply action
        ## update sheep's position
        pre_sheep = self.sheep.pos
        self.sheep.updatepos(action)
        cur_sheep = self.sheep.pos
        ## update dog's position and get next state
        state, distance, dest_angle = self.dog.updatepos(action, pre_sheep, cur_sheep)
        # Calculate Reward
        reward = 0.5 * (state[0] - RADIUS / 2) / RADIUS + 0.5 * (state[1] - pi / 2) / pi
        # check if game is done
        if self.sheep.pos[0] > RADIUS:
            done = True
            print(self.sheep.pos[0])
            if dest_angle / (self.dog.V / RADIUS) - 0.1 > distance / self.sheep.v:
                print('Winner : sheep')
            else:
                print('Winner : dog')
        else:
            print(self.sheep.pos[0])
            done = False
            print(done)
        # define empty info
        info = {}
        return state, reward, done, info

    def reset(self):
        # reset dog
        self.dog.resetpos()
        # reset sheep
        self.sheep.resetpos()
        # reset state
        if (fabs(self.sheep.pos[1] - self.dog.pos[1])) >= pi:
            included_angle = fabs(self.sheep.pos[1] - self.dog.pos[1])
        else:
            included_angle = 2 * pi - fabs(self.sheep.pos[1] - self.dog.pos[1])
        # wrap state into array
        state = np.array([self.sheep.pos[0], included_angle])
        # return the new initial state
        return state

    def render(self, mode='human'):
        # clear the previous wigets
        self.viewer.geoms.clear()
        # bilud wigets
        ring = rendering.make_circle(radius=RADIUS,res=50,filled=False)
        sheep = rendering.make_circle(radius=5,res=50,filled=True)
        dog = rendering.make_circle(radius=5,res=50,filled=True)
        # chase = rendering.make_circle(radius=10,res=50,filled=True) todo:...
        # set RGB color of wigets
        ring.set_color(0, 0, 0)
        ring.set_linewidth(5)
        sheep.set_color(0, 0, 255)
        dog.set_color(255, 0, 0)
        # chase.set_color(0,255,0) todo:...
        # add translations
        transform1 = rendering.Transform(translation=(300, 300))  # relatively translation
        transform_s = rendering.Transform(translation=(
            300 + self.sheep.pos[0] * cos(self.sheep.pos[1]), 300 + self.sheep.pos[0] * sin(self.sheep.pos[1])))
        transform_d = rendering.Transform(
            translation=(300 + self.dog.pos[0] * cos(self.dog.pos[1]), 300 + self.dog.pos[0] * sin(self.dog.pos[1])))
        #transform_c = rendering.Transform(
        #    translation=(300 + RADIUS * cos(self.dog.angle_p), 300 + RADIUS * sin(self.dog.angle_p))) todo:...
        # add translations to wigets
        ring.add_attr(transform1)
        sheep.add_attr(transform_s)
        dog.add_attr(transform_d)
        # chase.add_attr(transform_c) todo:...
        # add wigets to viewer
        self.viewer.add_geom(ring)
        self.viewer.add_geom(sheep)
        self.viewer.add_geom(dog)
        # self.viewer.add_geom(chase) todo:...

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
