import random
from Config import Config
from collections import deque

class ExperienceReplay(object):

    def __init__(self):
        self.replay_memory = deque()
        self.memory_step = 0
        self.memory_locked = False          # not used. Was planned to be worked with A3C

    def get_memory_length(self):
        return len(self.replay_memory)

    def reset_memory(self):
        self.replay_memory.clear()
        self.memory_step = 0
        self.memory_locked = False

    def is_memory_full(self):
        return (self.memory_step > Config.MEMORY_SIZE)

    # append episode to memory
    def add_memory(self, state, action, action_dist, reward, state_dash, done_flag):
        if self.memory_locked:
            print("Experience Replay locked. Available for training only.")
            return None
        else:
            self.replay_memory.append((state, action, action_dist, reward, state_dash, done_flag))
            self.memory_step += 1
            return self.memory_step

    # the memory will be only usable for Training but not storage afterwards
    def prepare_memory_for_training(self):
        # pre-calculate the discounted rewards for all state
        # need to run through for current replay before sampling (to preserve sequential info)
        running_r = 0
        reward_memory = [data[3] for data in self.replay_memory] # this is handled by Reference, thus can update Reward drectly
        done_flag_memory = [data[5] for data in self.replay_memory]

        for i in reversed(range(0, self.get_memory_length())):
            if done_flag_memory[i]:                 # as our ExpRply may contain multiple episodes seq.
                running_r = reward_memory[i]        # resetting running_r
            else:
                running_r = reward_memory[i] + Config.GAMMA * running_r
            reward_memory[i] = running_r

        self.memory_locked = True

    # only run this after preparing memory
    def get_sample_batch(self, batch_size = Config.TRAINING_BATCH_SIZE):
        memory_batch_size = min([self.get_memory_length(), batch_size])
        minibatch = random.sample(self.replay_memory, memory_batch_size)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        action_dist_batch = [data[2] for data in minibatch]
        reward_batch = [data[3] for data in minibatch]
        state_dash_batch = [data[4] for data in minibatch]
        done_flag_batch = [data[5] for data in minibatch]

        return memory_batch_size, state_batch, action_batch, action_dist_batch, reward_batch, state_dash_batch, done_flag_batch
