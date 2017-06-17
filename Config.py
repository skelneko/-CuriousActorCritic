
class Config:
    NUM_EPISODE = 100
    MEMORY_SIZE = 512
    TRAINING_BATCH_SIZE = 64    # max size

    LOG = False
    BACKGROUND = True

    SCREEN_W = SCREEN_H = 64
    SCREEN_SHAPE = [SCREEN_W, SCREEN_H]
    FRAME_PER_ROW = 4

    # this maintains versions and setting
    # [DATA_PROFILE, GAME, MAX_EPISODE, EPSILON_FLOOR, EPSILON_START, MOTIVTED, HYBRID_MOT]
    SCENARIOS = [
                    ["acn.v5", "pacman", 10, 0.2, 0.4, True, True],           #0    # R = Ri + Re; 500+ ESP TRAINED
                    ["acn.v6", "pacman", 10, 0.2, 0.4, True, False],          #1    # R = Ri only
                    ["acn.v7", "pacman", 10, 0.2, 0.4, False, False],         #2    # R = Re only
                    # Sort-of A3c with mixed motivation                       #3-12
                    ["asyn.v1", "pacman", 2, 0.2, 0.8, True, True],
                    ["asyn.v1", "pacman", 2, 0.2, 0.8, True, False],
                    ["asyn.v1", "pacman", 2, 0.2, 0.8, False, False],
                    ["asyn.v1", "pacman", 2, 0.2, 0.5, True, True],
                    ["asyn.v1", "pacman", 2, 0.2, 0.5, True, False],
                    ["asyn.v1", "pacman", 2, 0.2, 0.5, False, False],
                    ["asyn.v1", "pacman", 2, 0.2, 0.3, True, True],
                    ["asyn.v1", "pacman", 2, 0.2, 0.3, True, False],
                    ["asyn.v1", "pacman", 2, 0.2, 0.3, False, False],
                    ["asyn.v1", "pacman", 2, 0.1, 0.2, True, True],
                    # Sort-of A3c with Hybrid Motivation                      #13-44
                    ["asyn.v2", "pacman", 2, 0.2, 0.8, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.7, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.6, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.5, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.4, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.3, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.2, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.1, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.8, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.7, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.6, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.5, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.4, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.3, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.2, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.1, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.8, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.7, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.6, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.5, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.4, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.3, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.2, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.1, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.8, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.7, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.6, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.5, True, True],
                    ["asyn.v2", "pacman", 2, 0.2, 0.4, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.3, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.2, True, True],
                    ["asyn.v2", "pacman", 2, 0.1, 0.1, True, True],
                    # Sort-of A3c with only Intrincis Motivation              #44-51
                    ["asyn.v3", "pacman", 2, 0.2, 0.8, True, False],
                    ["asyn.v3", "pacman", 2, 0.2, 0.7, True, False],
                    ["asyn.v3", "pacman", 2, 0.2, 0.6, True, False],
                    ["asyn.v3", "pacman", 2, 0.2, 0.5, True, False],
                    ["asyn.v3", "pacman", 2, 0.2, 0.4, True, False],
                    ["asyn.v3", "pacman", 2, 0.1, 0.3, True, False],
                    ["asyn.v3", "pacman", 2, 0.1, 0.2, True, False],
                    ["asyn.v3", "pacman", 2, 0.1, 0.1, True, False],
                    # Sort-of A3c with only Extrinsic Motivation              #52-58
                    ["asyn.v4", "pacman", 2, 0.2, 0.8, False, False],
                    ["asyn.v4", "pacman", 2, 0.2, 0.7, False, False],
                    ["asyn.v4", "pacman", 2, 0.2, 0.6, False, False],
                    ["asyn.v4", "pacman", 2, 0.2, 0.5, False, False],
                    ["asyn.v4", "pacman", 2, 0.2, 0.4, False, False],
                    ["asyn.v4", "pacman", 2, 0.1, 0.3, False, False],
                    ["asyn.v4", "pacman", 2, 0.1, 0.2, False, False],
                    ["asyn.v4", "pacman", 2, 0.1, 0.1, False, False]
                ]
    CURRENT_SCENARIO = 0    # default

    FRAME_PER_ACTION = 1

    LEARNING_RATE = 1e-5
    EPSILON_DECAY = 0.9999
    EPSILON_FLOOR = SCENARIOS[CURRENT_SCENARIO][3]
    EPSILON = SCENARIOS[CURRENT_SCENARIO][4]
    GAMMA = 0.95      # discount factor

    # ACNetwork
    ALPHA = 0.5     # coefficient for loss_value
    BETA = 0.05      # coefficient for self entropy

    # ICM
    AGENT_SELF_MOTIVATED = SCENARIOS[CURRENT_SCENARIO][5]                # flag this to enable intrinsic reward
    MOTIVATED_BY_HYBRID_MODE = SCENARIOS[CURRENT_SCENARIO][6]             # when agent is self-motivated, flag this enable r = r_i + r_e
    ICM_LAMDA = 2     # coefficient to weight against intrinsic reward
    ICM_BETA = 0.6    # weight between ICM-FORWARD & ICM-INVERSE
    ICM_LEARNING_RATE = 1e-3
    ICM_ETA = 10      # coefficient to scale R_e as R_i


    LEARNING_DATA_PATH = "./data/{0}.{1}.data".format(SCENARIOS[CURRENT_SCENARIO][1], SCENARIOS[CURRENT_SCENARIO][0])
    TRAINING_LOG_PATH = "./data/{0}.{1}.log".format(SCENARIOS[CURRENT_SCENARIO][1], SCENARIOS[CURRENT_SCENARIO][0])
    LOSSES_LOG_PATH = "./data/{0}.{1}.losses.log".format(SCENARIOS[CURRENT_SCENARIO][1], SCENARIOS[CURRENT_SCENARIO][0])

    def load_scenario(idx = 0):
        Config.NUM_EPISODE = Config.SCENARIOS[idx][2]
        Config.EPSILON_FLOOR = Config.SCENARIOS[idx][3]
        Config.EPSILON = Config.SCENARIOS[idx][4]
        Config.AGENT_SELF_MOTIVATED = Config.SCENARIOS[idx][5]
        Config.MOTIVATED_BY_HYBRID_MODE = Config.SCENARIOS[idx][6]
        Config.LEARNING_DATA_PATH = "./data/{0}.{1}.data".format(Config.SCENARIOS[idx][1], Config.SCENARIOS[idx][0])
        Config.TRAINING_LOG_PATH = "./data/{0}.{1}.log".format(Config.SCENARIOS[idx][1], Config.SCENARIOS[idx][0])
        Config.LOSSES_LOG_PATH = "./data/{0}.{1}.losses.log".format(Config.SCENARIOS[idx][1], Config.SCENARIOS[idx][0])
        print("========================================================")
        print("Loading Profile[{0}] {1}.{2}:".format(idx, Config.SCENARIOS[idx][1], Config.SCENARIOS[idx][0]))
        print("Config Updated with the following...")
        print("Config.NUM_EPISODE: {0}".format(Config.NUM_EPISODE))
        print("Config.EPSILON_FLOOR: {0}".format(Config.EPSILON_FLOOR))
        print("Config.EPSILON: {0}".format(Config.EPSILON))
        print("Config.AGENT_SELF_MOTIVATED: {0}".format(Config.AGENT_SELF_MOTIVATED))
        print("Config.MOTIVATED_BY_HYBRID_MODE: {0}".format(Config.MOTIVATED_BY_HYBRID_MODE))
        print("Config.LEARNING_DATA_PATH: {0}".format(Config.LEARNING_DATA_PATH))
        print("Config.TRAINING_LOG_PATH: {0}".format(Config.TRAINING_LOG_PATH))
        print("Config.LOSSES_LOG_PATH: {0}".format(Config.LOSSES_LOG_PATH))
        print("========================================================")
