# Discounted Reward, OO based

from Config import Config
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time
from collections import deque

import datetime
from pathlib import Path

from Environment import Environment
from ExperienceReplay import ExperienceReplay
from Model import ActorCriticCNN, ICMN


class ACAgent(object):
    def __init__(self, action_size, presist_learning=True):
        self.action_size = action_size

        self.timestep = 0
        self.frame_per_action = Config.FRAME_PER_ACTION   # 1 = no skipping, not used anymore

        self.learning_rate = Config.LEARNING_RATE
        self.epsilon = Config.EPSILON
        self.epsilon_decay = Config.EPSILON_DECAY
        self.epsilon_floor = Config.EPSILON_FLOOR
        self.gamma = Config.GAMMA

        self.batch_size = Config.TRAINING_BATCH_SIZE

        self.replay_memory = ExperienceReplay()

        self.learning_data_path = Config.LEARNING_DATA_PATH
        self.training_log_path = Config.TRAINING_LOG_PATH
        self.losses_log_path = Config.LOSSES_LOG_PATH

        self. presist_learning = presist_learning

        input_shape = [-1, Config.SCREEN_W, Config.SCREEN_H, Config.FRAME_PER_ROW]
        output_shape = [self.action_size]

        # Networks
        self.ACN = ActorCriticCNN(input_shape, output_shape)
        self.AC_obs_input, self.v_out, self.policy_out = self.ACN.build_model()
        self.R_t, self.optimizer = self.ACN.build_optimizer()

        self.ICM = ICMN(input_shape, output_shape)
        self.ICM_obs_input, self.ICM_feature, self.ICM_s_dash_cap, self.ICM_a_cap = self.ICM.build_model()
        self.ICM_optimizer = self.ICM.build_optimizer()

    # take state and suggest an action with e-greedy
    def get_action(self, sess, state):

        current_e = max(self.epsilon, self.epsilon_floor)
        action, action_p, action_p_dist = self.ACN.predict_action(sess, state)

        if random.random() <= current_e:
            action = random.randint(0,self.action_size-1)
            action_p = action_p_dist[action]
            self.epsilon *= self.epsilon_decay

            if Config.LOG:
                print("S-{0}: Random Action [{1}], Epsilon = {2}".format(self.timestep, action, current_e))
        else:
            if Config.LOG:
                print("S-{0}: Taking Action [{1}]  with p(s,a) = {2}% ".format(self.timestep, action, action_p * 100))

        return action, action_p, action_p_dist


    # start to play the game
    def play(self, sess, env, learning=True):
        self.summaries = tf.summary.merge_all()
        sess.run(init)

        if self.presist_learning:
            self.restoreCheckpoint(sess)
        self.storeGraph(sess)

        reward_sum = 0
        start = datetime.datetime.now()
        print("Run starting... {0}".format(start))
        self.writeLog("Run starting... {0}".format(start))

        self.training_num = 0    # Checkpoint running counter

        for e in range (Config.NUM_EPISODE):
            env.reset()
            obs = env.get_screen(reduced=True)
            running_reward = 0
            running_int_reward = 0

            self.timestep = 0
            self.setTimer()

            done = False
            while not done:
                self.timestep += 1

                history = env.get_history()
                action, ap, ap_dist = self.get_action(sess, history)

                # take an action
                env.step(action)

                r_e = env.get_reward()                      # extrinsic reward
                done = env.get_done_flag()
                info = env.get_info()                       # not used
                history_dash = env.get_history()

                # here we handle the Motivation part
                if Config.AGENT_SELF_MOTIVATED:
                    r = self.ICM.get_intrinsic_reward(sess, history, history_dash, ap_dist)
                    running_int_reward += r
                    if Config.MOTIVATED_BY_HYBRID_MODE:
                        r += r_e
                else:
                    r = r_e

                # just display the env., or not
                if not(Config.BACKGROUND):
                    env.render()

                memory_step = self.replay_memory.add_memory(history, action, ap_dist, r, history_dash, done)

                running_reward += r_e
                reward_sum += r_e

                if done:
                    time_diff = agent.reportTimerDiff()
                    str = "[Episode {0}] Steps: {1} Reward: {2:.5g}, Avg: {3:.5g}, Intrinsic/Step: {4:.5g}, Time: {5}".format(e,agent.timestep,running_reward,reward_sum / (e+1), running_int_reward/self.timestep, time_diff)

                    print(str)
                    self.writeLog(str)

                    # we train when memory is full, or running out of episode
                    if  learning and (self.replay_memory.is_memory_full() or e == Config.NUM_EPISODE-1):
                        self.train_replay(sess)
                        self.saveCheckpoint(sess, self.training_num)
                        self.training_num += 1
                    break

        end = datetime.datetime.now()
        end_msg = "Training ends ... {0} [Total Time: {1}]".format(end, end - start)
        print(end_msg)
        self.writeLog(end_msg)


    # run through memory and update models
    def train_replay(self, sess):
        print("Training replay...")
        start = datetime.datetime.now()
        self.replay_memory.prepare_memory_for_training()
        memory_batch_size, state_batch, action_batch, action_dist_batch, reward_batch, state_dash_batch, done_flag_batch = \
                            self.replay_memory.get_sample_batch(Config.TRAINING_BATCH_SIZE)

        # train ACNetwork
        v_dash_batch = []
        running_v = 0
        _, v_dash_values = self.ACN.forward(sess, state_dash_batch)

        for i in range(0, memory_batch_size):
            if done_flag_batch[i]:
                v_dash_batch.append(reward_batch[i])
            else:
                v_dash_batch.append(reward_batch[i] + self.gamma * v_dash_values[i][0])

        Lv, Lp, Hp, c, optimizer = self.ACN.update_gradients(sess, v_dash_batch, state_batch)

        # train ICM
        Lfwd = Linv = ICM_c = -1
        if Config.AGENT_SELF_MOTIVATED:
            Lfwd, Linv, ICM_c, ICM_optimizer = self.ICM.update_gradients(sess, reward_batch, state_batch, state_dash_batch, action_dist_batch)

        # clear all memory we have (this is different from A3C as they only withdraw trained memory)
        self.replay_memory.reset_memory()

        end = datetime.datetime.now()
        print("Training finished. Spent: {0}".format(end-start))
        print("ACN Losses: [Lv] {0:.5g} / [Lp] {1:.5g} / [Hp] {2:.3f} / [C] {3:.5g}".format(Lv, Lp, Hp, c))

        if Config.AGENT_SELF_MOTIVATED:
            print("ICM Losses: [Lfwd] {0:.5g} / [Linv] {1:.3f} / [C] {2:.5g}".format(Lfwd, Linv, ICM_c))

        self.logLosses([Lv, Lp, Hp, c, Lfwd, Linv, ICM_c])


    def saveCheckpoint(self, sess, step):
        saver = tf.train.Saver()
        save_path = saver.save(sess, self.learning_data_path) #, global_step=step)
        print("Saving Checkpoint #{0}".format(step))

    def restoreCheckpoint(self, sess):
        if Path(self.training_log_path).is_file():
            saver = tf.train.Saver()
            saver.restore(sess, self.learning_data_path)        # suppose to be path+step
            print("Checkpoint loaded")
        else:
            print("Checkpoint not found, creating one instead")
            self.saveCheckpoint(sess, step =-1)


    def writeLog(self, str):
        self.writeLogTo(self.training_log_path, str)

    def logLosses(self, losses):
        str = "{0},{1},{2},{3},{4},{5},{6}".format(losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6])
        self.writeLogTo(self.losses_log_path,str)

    def writeLogTo(self, file_path, str):
        f = open(file_path, 'a')
        f.write(str + "\n")  # python will convert \n to os.linesep
        f.close()


    def storeGraph(self, sess):
        self.summary_writer = tf.summary.FileWriter('graph_log', sess.graph)

    def setTimer(self):
        self.startTimer = datetime.datetime.now()
        return None

    def reportTimerDiff(self):
        diff = datetime.datetime.now() - self.startTimer
        return diff

if __name__ == "__main__":

    # this allows us to loop through different profile setting to play around with settings
    max_batch = 1
    for i in range(0,max_batch):
        # sc_list = list(range(0,len(Config.SCENARIOS)))
        sc_list = list(range(13, 45))                       # running A3C Hybrid only
        sc_len = len(sc_list)
        j = 0
        random.shuffle(sc_list)
        for sc in sc_list:
            j += 1
            print("Running Batch #{0} (Max: {1})  |  Profile #{2} (Max: {3}) | Progress {4:.2f}%".format(\
                                                                    i+1, max_batch, j, sc_len, (((i*sc_len) + j)*100) / (max_batch*sc_len)))
            tf.reset_default_graph()
            Config.load_scenario(sc)
            env = Environment("MsPacman-v0")
            agent = ACAgent(action_size = env.get_action_space(), presist_learning=True)
            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                agent.play(sess, env, learning=True)
                sess.close()
