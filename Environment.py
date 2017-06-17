from Config import Config
import gym
import PIL
from collections import deque
import numpy as np

class Environment(object):

    def __init__(self, game="MsPacman-v0"):

        self.screen_h = Config.SCREEN_H
        self.screen_w = Config.SCREEN_W
        self.screen_shape = Config.SCREEN_SHAPE
        self.frame_per_row = Config.FRAME_PER_ROW
        self.frame_buffer = None

        self.action_space = 9

        # meta
        self.total_episode_run = 0
        self.steps_in_episode = 0
        self.max_steps_in_episode = 0

        self.env = gym.make(game)
        self.reset()


    def init_frame_buffer(self):
        # initialize history
        if self.frame_buffer:
            self.frame_buffer.clear()
        else:
            self.frame_buffer = deque()
        for i in range(0, self.frame_per_row):
            self.frame_buffer.append(self.get_screen(reduced=True))    # always full

    def reset(self):
        self.max_steps_in_episode = max(self.max_steps_in_episode, self.steps_in_episode)
        self.current_screen = self.env.reset()     # current_screen always align with ORIGINAL SETTING
        self.init_frame_buffer()
        self.current_reward = 0
        self.done_flag = False
        self.info = None
        self.total_episode_run += 1
        self.steps_in_episode = 0

    def step(self, action):
        self.current_screen, r, self.done_flag, self.info = self.env.step(action)
        self.current_reward = r
        self.frame_buffer.popleft()
        self.frame_buffer.append(self.get_screen(reduced=True))
        self.steps_in_episode += 1

    def render(self):
        self.env.render()


    ### GETs ###
    def get_environment(self):
        return self.env

    def get_screen(self, reduced=True):
        if reduced:
            grayscale = self.rgb2gray(self.current_screen)
            return self.resizeScreen(grayscale, self.screen_shape)
        else:
            return self.current_screen

    def get_reward(self):
        return self.current_reward

    def get_done_flag(self):
        return self.done_flag

    def get_info(self):
        return self.info

    def get_action_space(self):
        return self.action_space

    def get_frame_buffer(self):
        return self.frame_buffer

    # return full list of frame_buffer in {W, H, Channel} shape
    def get_history(self):
        return np.transpose(self.frame_buffer, (1,2,0))

    def get_max_steps(self):
        return self.max_steps_in_episode


    ### Utilities ###
    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def resizeScreen(self, state, shape):
        img = PIL.Image.fromarray(state, mode=None)
        img = img.resize(shape, PIL.Image.LANCZOS)
        arr = list(img.getdata())
        return np.reshape(arr, shape)
