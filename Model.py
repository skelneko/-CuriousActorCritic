# Discounted Reward, OO based

from Config import Config
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import datetime
from pathlib import Path

# dummy Model class for inheritance
class Model(object):
    def __init__(self, input_tensor_shape, output_tensor_shape):
        self.input_tensor_shape = input_tensor_shape
        self.output_tensor_shape = output_tensor_shape

    ### basic helpers ###
    def new_weights(self, shape, name="weights"):
        return tf.Variable(tf.random_normal(shape=shape, mean=0, stddev=0.05, name=name))

    def new_biases(self, length, name="biases"):
        return tf.Variable(tf.random_normal(shape=[length], mean=0, stddev=0.05, name=name))

    def new_sigmoid_layer(self, input, num_input, num_output, use_softmax=True, name="sig"):
        sig_w = self.new_weights(shape=[num_input, num_output], name=name+"-weights")
        sig_b = self.new_biases(length=num_output, name=name+"-biases")
        layer = tf.matmul(input, sig_w) + sig_b
        layer = tf.nn.sigmoid(layer)

        if use_softmax:
            layer = tf.nn.softmax(layer, name=name+"-softmax")

        return layer, sig_w, sig_b


    def storeGraph(self, sess):
        summary_writer = tf.summary.FileWriter('graph_log', graph=tf.get_default_graph())

    def forward(self, input):
        return None

    # placeholder for standlone network testing
    def test_network(self, sess):
        return None


class CNN(Model):
    # expecting [B, W, H  C] format for input_tensor_shape
    def __init__(self, input_tensor_shape, output_tensor_shape):
        Model.__init__(self, input_tensor_shape, output_tensor_shape)
        self.input_shape_batch = -1
        self.input_shape_width = input_tensor_shape[1]
        self.input_shape_height = input_tensor_shape[2]
        self.input_shape_channels = input_tensor_shape[3]

        # assumed to be 1D only
        self.output_shape_length = output_tensor_shape[0]

    # filter_shape = [filter_size, filter_size, num_input_channels, num_filters]
    def new_cnn_layer(self, input, num_filters, filter_shape, ksize=2, stride=2, use_pooling=True, name="CNN"):
        weights = self.new_weights(shape=filter_shape, name=name+"-weights")
        biases = self.new_biases(length=num_filters, name=name+"-biases")
        layer = tf.nn.conv2d(input=input, filter=weights,
                                strides=[1,stride,stride,1],
                                padding='SAME',
                                name=name) + biases
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,
                                    ksize=[1,ksize,ksize,1], strides=[1,2,2,1],
                                    padding='SAME')
        return layer, weights, biases

    def new_fc_layer(self, input, num_input, num_output, use_relu=True, name="fc"):
        fc_w = self.new_weights(shape=[num_input, num_output], name=name+"-weights")
        fc_b = self.new_biases(length=num_output, name=name+"-biases")
        layer = tf.matmul(input, fc_w) + fc_b
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer, fc_w, fc_b

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features


# CNN that used for learning for Gym
class GymCNN(CNN):
    # expecting [B, W, H  C] format for input_tensor_shape
    def __init__(self, input_tensor_shape, output_tensor_shape):
        CNN.__init__(self, input_tensor_shape, output_tensor_shape)

        self.num_filter1 = 16
        self.filter1_shape = [4, 4, Config.FRAME_PER_ROW, self.num_filter1]
        self.filter1_kernel_d = 2   # reducdant as no pooling
        self.filter1_stride_d = 2

        self.num_filter2 = 16
        self.filter2_shape = [4, 4, self.num_filter1, self.num_filter2]
        self.filter2_kernel_d = 2   # reducdant as no pooling
        self.filter2_stride_d = 3

        self.num_filter3 = 16
        self.filter3_shape = [4, 4, self.num_filter2, self.num_filter3]
        self.filter3_kernel_d = 2   # reducdant as no pooling
        self.filter3_stride_d = 3

    def build_model(self):
        # obs_input = tf.placeholder(shape=[None, Config.SCREEN_W, Config.SCREEN_H, Config.FRAME_PER_ROW], dtype=tf.float32)
        self.obs_input = tf.placeholder(shape=[None, self.input_shape_width, self.input_shape_height, self.input_shape_channels], dtype=tf.float32)

        self.cnn1, self.cnn1_w, self.cnn1_b = self.new_cnn_layer(input=self.obs_input,
                                                num_filters=self.num_filter1,
                                                filter_shape=self.filter1_shape,
                                                ksize=self.filter1_kernel_d,
                                                stride=self.filter1_stride_d,
                                                use_pooling=False,
                                                name="CNN1")

        self.cnn2, self.cnn2_w , self.cnn2_b = self.new_cnn_layer(input=self.cnn1,
                                                num_filters=self.num_filter2,
                                                filter_shape=self.filter2_shape,
                                                ksize=self.filter2_kernel_d,
                                                stride=self.filter2_stride_d,
                                                use_pooling=False,
                                                name="CNN2")

        self.cnn3, self.cnn3_w , self.cnn3_b = self.new_cnn_layer(input=self.cnn2,
                                                num_filters=self.num_filter3,
                                                filter_shape=self.filter3_shape,
                                                ksize=self.filter3_kernel_d,
                                                stride=self.filter3_stride_d,
                                                use_pooling=False,
                                                name="CNN3")

        self.cnn3_flat, self.num_cnn3_out = self.flatten_layer(self.cnn3)

        self.fc_out, self.fc_out_w, self.fc_out_b = self.new_fc_layer(input=self.cnn3_flat,
                                                    num_input=self.num_cnn3_out,
                                                    num_output=self.num_cnn3_out,
                                                    use_relu=True,
                                                    name="FC_1")

    # an utility to fix input with batch_size = 1
    def reshape_for_batch(self, input):
        if len(np.shape(input)) == 3:    # if 3D only... i.e. it's a single state
            state = list(np.transpose(input, (1,2,0)))
            input_s = np.reshape(input, (self.input_shape_batch,
                                            self.input_shape_width,
                                            self.input_shape_height,
                                            self.input_shape_channels))
        else:
            input_s = input
        return input_s


# riding on GymCNN, we add LSTM and get QValue for action space
# Ref: Playing Atari with Deep Reinforcement Learning: https://arxiv.org/abs/1312.5602
class QValueCNN(GymCNN):
    def __init__(self, input_tensor_shape, output_tensor_shape):
        GymCNN.__init__(self, input_tensor_shape, output_tensor_shape)      # output_tensor_shape used for action listing

    # build Q_Model for both current and target
    def build_model(self):
        with tf.name_scope("QValueCNN"):

            GymCNN.build_model(self)

            # adding lstm layer
            state_size= self.num_cnn3_out
            batch_size = 1 # further study required
            with tf.variable_scope("LSTM") as scope:
                self.lstm_in = [self.fc_out]
                self.lstm_in = tf.transpose(self.lstm_in, [0, 1, 2])    # h_in

                self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=state_size)
                states = self.cell.zero_state(batch_size, tf.float32)
                h_out, states = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.lstm_in, initial_state = states)
                self.h_out_unpacked = tf.unstack(h_out, axis=0)
                self.lstm_out = self.h_out_unpacked[0]


            # adding QValue outputs
            self.q_out, self.q_out_w, self.q_out_b = self.new_fc_layer(input=self.lstm_out,
                                                        num_input=self.num_cnn3_out,
                                                        num_output=self.output_shape_length,
                                                        use_relu=True,
                                                        name="FC_Q_Out")

        # a return in older version, didn't bother to clean up though...
        return self.obs_input, self.cnn1, self.cnn1_w, self.cnn1_b, self.cnn2, self.cnn2_w, self.cnn2_b, self.fc_out, self.fc_out_w, self.fc_out_b, self.q_out, self.q_out_w, self.q_out_b


    def build_optimizer(self):
        with tf.name_scope("QValueCNNOptimizer"):
            self.q_pv = tf.placeholder(dtype=tf.float32, shape=[None], name="q_dash")
            self.max_q = tf.reduce_max(self.q_out, axis=[1])
            self.cost = tf.reduce_mean(tf.square(self.q_pv - self.max_q))  # will opt for Qout i.e. Q-Network
            self.optimizer = tf.train.AdamOptimizer(Config.LEARNING_RATE).minimize(self.cost)
        return self.q_pv, self.max_q, self.cost, self.optimizer

    def predict_action(self, sess, state):
        q = self.forward(sess, state)
        action = q.argmax(1)
        qv = q.argmax(0)
        return action, qv

    def forward(self, sess, input):
        input_s = self.reshape_for_batch(input)
        q = sess.rn(self.q_out, feed_dict={self.obs_input: input_s})
        losses = None # not implemented
        return q

    def update_gradients(self, sess, q_pv, q_obs_input):
        feed = {self.q_pv: q_pv, self.obs_input : q_obs_input}
        sess.run(self.optimizer, feed_dict = feed)

    # utility that help clone QN to QtN after training, not updated.
    def update_target_model_by_self(self, cnn_target):
        # to be updated
        cnn_target.cnn1_w.assign(self.cnn1_w)
        cnn_target.cnn1_b.assign(self.cnn1_b)
        cnn_target.cnn2_w.assign(self.cnn2_w)
        cnn_target.cnn2_b.assign(self.cnn2_b)
        cnn_target.cnn3_w.assign(self.cnn3_w)
        cnn_target.cnn3_b.assign(self.cnn3_b)
        cnn_target.fc_out_w.assign(self.fc_out_w)
        cnn_target.fc_out_b.assign(self.fc_out_b)
        cnn_target.cell.assign(self.cell)
        cnn_target.q_out_w.assign(self.q_out_w)
        cnn_target.q_out_b.assign(self.q_out_b)
        return None



# both Actor and Critic share the same CNN base thus interitance
# Ref: Asynchronous Methods for Deep Reinforcement Learning: https://arxiv.org/abs/1602.01783
class ActorCriticCNN(QValueCNN):
    def __init__(self, input_tensor_shape, output_tensor_shape):
        QValueCNN.__init__(self, input_tensor_shape, output_tensor_shape)   # output_tensor_shape used for action listing only

    def build_model(self):
        _ = QValueCNN.build_model(self) # sharing the base CNN up to LSTM feature

        with tf.name_scope("CriticLayer"):
            # for Critic V(s)
            self.v_h1, self.v_h1_w, self.v_h1_b = self.new_fc_layer(input=self.lstm_out,
                                                                        num_input= self.num_cnn3_out,
                                                                        num_output=self.num_cnn3_out,
                                                                        use_relu=False)

            self.v_out, self.v_out_w, self.v_out_b = self.new_fc_layer(input=self.v_h1,
                                                                        num_input= self.num_cnn3_out,
                                                                        num_output=1,                   # we critize s, thus output=1
                                                                        use_relu=False,
                                                                        name="Critic_out")

        with tf.name_scope("ActorLayer"):
            # for Actor Pi(s)
            self.policy_out, self.p_out_w, self.p_out_b = self.new_sigmoid_layer(input=self.v_h1,
                                                            num_input=self.num_cnn3_out,
                                                            num_output=self.output_shape_length,
                                                            use_softmax=True,
                                                            name="Policy_out")

        return self.obs_input, self.v_out, self.policy_out   # simplified return

    def build_optimizer(self):
        with tf.name_scope("ACOptimizer"):
            # optimizer requires obs_input to get v_out & policy_out
            # plus also the R_t to estiamte the advantage for convergence
            self.R_t = tf.placeholder(dtype=tf.float32, shape=[None], name="R_t")
            self.adventages = self.R_t - self.v_out

            # L_value = R - V(s)    // as R is best estimation of Q(s,a)
            self.loss_value = tf.reduce_sum(tf.square(self.adventages))

            # L_policy = log(pi(s)) * A(s,a)
            self.action_p = tf.reduce_max(self.policy_out, axis = [1]) + 1e-6   # 1E-6 to prevent NAN
            self.loss_policy = -tf.reduce_sum(tf.log(self.action_p) * self.adventages)

            # self entropy : H(Pi) = p log (p)
            # this is used to max out entropy of Pi(s) to maintain exploratory nature
            self.self_entropy = -tf.reduce_sum(self.policy_out * tf.log(self.policy_out + 1e-6)) # 1E-6 to prevent NAN

            # Loss = alpha * min(L_value) + min(L_policy) - beta * self_entropy (H(Pi))
            self.cost = (Config.ALPHA * self.loss_value) + self.loss_policy - (Config.BETA * self.self_entropy)
            self.optimizer = tf.train.AdamOptimizer(Config.LEARNING_RATE).minimize(self.cost)

        return self.R_t, self.optimizer

    def test(self, sess, input):
        input_s = self.reshape_for_batch(input)
        return sess.run(self.lstm_in, feed_dict={self.obs_input: input_s})

    def forward(self, sess, input):
        input_s = self.reshape_for_batch(input)
        p, v = sess.run([self.policy_out, self.v_out], feed_dict = {self.obs_input: input_s})
        return p, v

    def update_gradients(self, sess, r_t, obs_input):
        feed = {self.R_t : r_t, self.obs_input : obs_input}
        return sess.run((self.loss_value, self.loss_policy, self.self_entropy, self.cost, self.optimizer), feed_dict = feed)

    def predict_action(self, sess, state):
        p, v = self.forward(sess, state)
        action = p[0].argmax()
        action_p = p[0][action]
        return action, action_p, p[0]

    def retrive_losses(self, sess, r_t, obs_input):
        # retrieve this only after update_gradient
        feed = {self.R_t : r_t, self.obs_input : obs_input}
        Lv, Lp, Hp = sess.run((self.loss_value, self.loss_policy, self.self_entropy), feed_dict = feed)
        return Lv, Lp, Hp





# we use the same structure as GynCNN, without sharing with ACN
# Ref: Curiosity-driven Exploration by Self-supervised Prediction: https://arxiv.org/abs/1705.05363
class ICMN(GymCNN):

    def __init__(self, input_tensor_shape, output_tensor_shape):
        GymCNN.__init__(self, input_tensor_shape, output_tensor_shape)           # output_tensor_shape used for action listing only

    def build_model(self):
        with tf.name_scope("ICMN"):
            GymCNN.build_model(self)

            # ride on fc_out as state features
            self.state_feature_shape = self.fc_out.get_shape()
            self.state_feature_size = self.state_feature_shape[1:].num_elements()
            self.state_feature_batch, state_feature_w, state_feature_b = \
                                            self.new_sigmoid_layer(input=self.fc_out,
                                            num_input=self.state_feature_size ,
                                            num_output=self.state_feature_size ,
                                            use_softmax=False,
                                            name="icm-sigm")


            # accept batch
            self.s_feature = tf.placeholder(shape=self.state_feature_shape, dtype=tf.float32)
            self.s_dash_feature = tf.placeholder(shape=self.state_feature_shape, dtype=tf.float32)
            self.a_dist = tf.placeholder(shape=[None, self.output_shape_length], dtype=tf.float32)

            # simple nn to predict s'^ with s and a
            name_predict = "ICMN-PredictNN"
            with tf.name_scope(name_predict):
                # predict_vector = tf.concat(1, [self.a_out, self.s_feature])
                predict_vector = tf.concat([self.a_dist, self.s_feature], 1)
                predict_vector_size = self.state_feature_size+self.output_shape_length

                predict_s_h1, s_h1_w, s_h1_b = self.new_fc_layer(input=predict_vector,
                                                num_input=predict_vector_size,
                                                num_output=predict_vector_size,
                                                name=name_predict+"h1")

                predict_s_out, s_out_w, s_out_b = self.new_sigmoid_layer(input=predict_s_h1,
                                                        num_input=predict_vector_size,
                                                        num_output=self.state_feature_size,
                                                        use_softmax=False,
                                                        name=name_predict+"h2-sigm")

                self.s_dash_cap = predict_s_out


            # simple nn as inverse to predict a^ with observed s and s'
            name_inverse = "ICMN-InverseNN"
            with tf.name_scope(name_inverse):
                # inverse_vector = tf.concat(1, [self.s_feature, self.s_dash_feature])
                inverse_vector = tf.concat([self.s_feature, self.s_dash_feature], 1)
                inverse_a_h1, a_h1_w, a_h1_b = self.new_fc_layer(input=inverse_vector,
                                                    num_input=self.state_feature_size*2,
                                                    num_output=self.state_feature_size*2,
                                                    name=name_inverse+"h1")
                inverse_a_out, a_out_w, a_out_b  = self.new_sigmoid_layer(input=inverse_a_h1,
                                                    num_input=self.state_feature_size*2,
                                                    num_output=self.output_shape_length,
                                                    use_softmax=False,
                                                    name=name_inverse+"h2-sigm")
                self.a_dist_cap = inverse_a_out

            return self.obs_input, self.s_feature, self.s_dash_cap, self.a_dist_cap


    def build_optimizer(self):
        with tf.name_scope("ICMNOptimizer"):

            # Lfwd = MSE(s, s_dash), prediction errors that used as rewards as well
            self.loss_icm_fwd = tf.reduce_sum(tf.square(self.s_dash_cap - self.s_dash_feature))

            # Linv = cross entropy between a and a_cap
            # this bounds the learning towards agent's action only
            self.loss_icm_inv = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.a_dist_cap, labels=self.a_dist))

            # passing discounted R_e with V value, such that network can learn against what it has been rewarded
            self.R = tf.placeholder(dtype=tf.float32, shape=[None], name="maxR")
            r = tf.reduce_sum(self.R)

            # LAMDA governs how much we learn against R
            # BETA balancing forward and inverse losses
            self.cost = - Config.ICM_LAMDA * r + \
                            Config.ICM_BETA * self.loss_icm_fwd + \
                            (1 - Config.ICM_BETA) * self.loss_icm_inv

            self.optimizer = tf.train.AdamOptimizer(Config.ICM_LEARNING_RATE).minimize(self.cost)


    # return feature for batch of states based on the CNN part
    def featurize(self, sess, input):
        input_s = self.reshape_for_batch(input)
        f = sess.run(self.state_feature_batch, feed_dict = {self.obs_input: input_s}) #1D feature vector
        return f

    def inverse(self, sess, s, s_dash):
        input_s = self.reshape_for_batch(s)
        input_s_d = self.reshape_for_batch(s_dash)
        inv = sess.run([self.a_dist_cap], feed_dict = {self.s_feature: input_s, self.s_dash_feature: input_s_d})
        return inv

    def forward(self, sess, s, a):
        input_s = self.reshape_for_batch(s)
        fwd = sess.run([self.s_dash_cap], feed_dict = {self.s_feature: input_s, self.a_dist: a})
        return fwd

    # Lp is the policy loss from ACNetwork of the same batch
    def update_gradients(self, sess, r, s, s_dash, a_dist):
        s_f = self.featurize(sess, s)
        s_d_f = self.featurize(sess, s_dash)
        feed = {self.s_feature: s_f, self.s_dash_feature: s_d_f, self.a_dist: a_dist, self.R: r}
        return sess.run((self.loss_icm_fwd, self.loss_icm_inv, self.cost, self.optimizer), feed_dict = feed)


    def retrive_losses(self, sess, s, s_dash, a):
        s_f = self.featurize(sess, s)
        s_d_f = self.featurize(sess, s_dash)

        # for single batch
        if len(np.shape(a)) == 1:
            a = [a]


        feed = {self.s_feature: s_f, self.s_dash_feature: s_d_f, self.a_dist: a}
        return sess.run((self.loss_icm_fwd, self.loss_icm_inv), feed_dict = feed)

    def get_intrinsic_reward(self, sess, s, s_dash, a):
        Lfwd, Linv = self.retrive_losses(sess, s, s_dash, a)
        return Lfwd * Config.ICM_ETA




if __name__ == "__main__":

    # testing __main__
    input_shape = [3, 84, 84, 3]
    output_shape = [9]
    ACN = ActorCriticCNN(input_shape, output_shape)
    _ = ACN.build_model()
    _ = ACN.build_optimizer()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        ACN.test_network(sess)
        sess.close()
