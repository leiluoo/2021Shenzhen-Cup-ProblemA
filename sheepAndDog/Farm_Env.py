'''
This is a environment for RL
'''
import time

from gym import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import random
from math import cos, pi, sqrt, fabs, pow, acos
from gym.envs.classic_control import rendering

####################################hyper parameters#################################
INIT_V = 1      # velocity of dog
INIT_v = 0.5    # velocity of sheep
RADIUS = 20     # radius of farmland
DELTA_T = 0.1   # interval
ALL_T = 20      # total time
#####################################################################################

class Farm_Env(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # dog's position
        self.dogPos = np.array([RADIUS, random.uniform(0,2) * pi])
        # sheep's position
        self.sheepPos = np.array([random.uniform(0,1) * RADIUS, random.uniform(0,2) * pi])
        # action the sheep can take, the degree it turns
        self.action_space = Box(low = np.array([0]), high = np.array([pi])) # todo:check the method
        # the distance between sheep and the boarder of farmland , the angle bwtween sheep and dog
        self.observation_space = Box(low = np.array([0, 0]), high = np.array([RADIUS, pi]))
        # time interval
        self.t = DELTA_T
        self.all_time = ALL_T

        self.seed()

    def step(self, action):
        # apply action
        self.pre_rho = self.sheepPos[0]
        self.pre_theta = self.sheepPos[1]
        if(action < pi/2 ):
            self.sheepPos[0] = sqrt(pow(self.v*cos(action)*self.t,2)+pow(self.sheepPos[0],2)-2*self.sheepPos[0]*cos(action)*self.t*cos((pi/2)+action))
            self.sheepPos[1] = self.sheepPos[1] - acos((pow(self.sheepPos[0], 2) + pow(self.pre_rho, 2) - pow(self.v * cos(action) * self.t, 2)) / (2 * self.sheepPos[0] * self.pre_rho))
        elif(action > pi/2):
            self.sheepPos[0] = sqrt(pow(self.v * cos(pi-action) * self.t, 2) + pow(self.sheepPos[0], 2) - 2 * self.sheepPos[0] * cos(pi-action) * self.t * cos((pi / 2) + pi - action))
            self.sheepPos[1] = self.sheepPos[1] + acos((pow(self.sheepPos[0], 2) + pow(self.pre_rho, 2) - pow(self.v * cos(pi-action) * self.t, 2)) / (2 * self.sheepPos[0] * self.pre_rho))
        else:
            self.sheepPos[0] += self.v * self.t

        if(self.sheepPos[1] > 2*pi):
            self.sheepPos[1] = self.sheepPos[1] - 2*pi

        # update dog's position
        self.b = -2 * cos(action + pi/2) * self.pre_rho
        self.c = pow(self.pre_rho,2) - pow(RADIUS,2)
        self.distance = (-self.b - sqrt(pow(self.b,2) - 4 * self.c)) / (4 * self.c)
        self.gamma = acos((pow(self.pre_rho,2) + pow(RADIUS,2) - pow(self.distance,2)) / (2 * RADIUS * self.pre_rho))

        self.pre_angle = fabs(self.pre_theta - self.dogPos[1])
        if(self.pre_angle > pi):
            self.pre_angle = 2*pi - self.pre_angle
        self.cur_angle = fabs(self.sheepPos[1] - self.dogPos[1])
        if (self.cur_angle > pi):
            self.cur_angle = 2 * pi - self.cur_angle

        if(self.dogPos[1] < pi):
            if(self.pre_theta > self.dogPos[1] and self.pre_theta <= self.dogPos[1] + pi):
                if(self.pre_angle > self.cur_angle):
                    self.angle_pos = self.pre_angle - self.gamma
                else:
                    self.angle_pos = self.pre_angle + self.gamma
                if(self.angle_pos > self.dogPos[1] and self.angle_pos <= self.dogPos[1] + pi):
                    self.dogPos[1] = self.dogPos[1] + self.V/RADIUS * self.t
                else:
                    self.dogPos[1] = self.dogPos[1] - self.V/RADIUS * self.t
            elif(self.pre_theta <= self.dogPos[1] and self.pre_theta > 0 or self.pre_theta > self.dogPos[1] + pi and self.pre_theta <= 0):
                if (self.pre_angle > self.cur_angle):
                    self.angle_pos = self.pre_angle + self.gamma
                else:
                    self.angle_pos = self.pre_angle - self.gamma
                if (self.pre_theta <= self.dogPos[1] and self.pre_theta > 0 or self.pre_theta > self.dogPos[1] + pi and self.pre_theta <= 0):
                    self.dogPos[1] = self.dogPos[1] - self.V / RADIUS * self.t
                else:
                    self.dogPos[1] = self.dogPos[1] + self.V / RADIUS * self.t
        if (self.dogPos[1] >= pi):
            if (self.pre_theta > self.dogPos[1] and self.pre_theta <= 2 * pi or self.pre_theta <= self.dogPos[1] - pi and self.pre_theta > 0):
                if (self.pre_angle > self.cur_angle):
                    self.angle_pos = self.pre_angle - self.gamma
                else:
                    self.angle_pos = self.pre_angle + self.gamma
                if (self.pre_theta > self.dogPos[1] and self.pre_theta <= 2 * pi or self.pre_theta <= self.dogPos[1] - pi and self.pre_theta > 0):
                    self.dogPos[1] = self.dogPos[1] + self.V / RADIUS * self.t
                else:
                    self.dogPos[1] = self.dogPos[1] - self.V / RADIUS * self.t
            elif (self.pre_theta > self.dogPos[1] - pi and self.pre_theta <= self.dogPos[1]):
                if (self.pre_angle > self.cur_angle):
                    self.angle_pos = self.pre_angle + self.gamma
                else:
                    self.angle_pos = self.pre_angle - self.gamma
                if (self.pre_theta > self.dogPos[1] - pi and self.pre_theta <= self.dogPos[1]):
                    self.dogPos[1] = self.dogPos[1] - self.V / RADIUS * self.t
                else:
                    self.dogPos[1] = self.dogPos[1] + self.V / RADIUS * self.t

        self.all_time -= 0.1
        # Calculate Reward
        if((fabs(self.sheepPos[1] - self.dogPos[1])) >= pi):
            self.included_angle = fabs(self.sheepPos[1] - self.dogPos[1])
        else:
            self.included_angle = 2*pi - fabs(self.sheepPos[1] - self.dogPos[1])
        reward = 0.5 * ((self.sheepPos[0]-RADIUS) - RADIUS/2)/RADIUS + 0.5 * (self.included_angle - pi/2)/pi

        # check if game is done
        if(self.sheepPos[1] > RADIUS or self.all_time <= 0):
            done = True
        else:
            done = False

        info = {}

        if ((fabs(self.sheepPos[1] - self.dogPos[1])) >= pi):
            self.included_angle = fabs(self.sheepPos[1] - self.dogPos[1])
        else:
            self.included_angle = 2 * pi - fabs(self.sheepPos[1] - self.dogPos[1])

        self.state = np.array([RADIUS - self.sheepPos[0], self.included_angle])

        return self.state, reward, done, info

    def reset(self):
        # dog's position
        self.dogPos = np.array([RADIUS, random.uniform(0, 2) * pi])
        # sheep's position
        self.sheepPos = np.array([random.uniform(0, 1) * RADIUS, random.uniform(0, 2) * pi])
        self.all_time = ALL_T

        if ((fabs(self.sheepPos[1] - self.dogPos[1])) >= pi):
            self.included_angle = fabs(self.sheepPos[1] - self.dogPos[1])
        else:
            self.included_angle = 2 * pi - fabs(self.sheepPos[1] - self.dogPos[1])

        self.state = np.array([RADIUS - self.sheepPos[0], self.included_angle])

        return self.state

    def render(self, mode='human'):
        self.viewer = rendering.Viewer(600, 600)
        # 方式一
        ring = rendering.make_circle(radius=200,
                                     res=50,
                                     filled=False)
        # radius=10 半径
        # res=30    说是画圆，其实是画正多边形，res指定多边形的边数
        # filled=True   是否填充
        ring.set_color(0, 0, 0)
        ring.set_linewidth(5)  # 设置线宽
        sheep = rendering.make_circle(radius=10,
                                     res=3,
                                     filled=True)
        dog =  rendering.make_circle(radius=10,
                                      res = 4,
                                      filled = True)

        # 添加一个平移操作
        transform1 = rendering.Transform(translation=(300, 300))  # 相对偏移
        transform_s = rendering.Transform(translation=(320, 280))
        transform_d = rendering.Transform(translation=(300, 500))
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