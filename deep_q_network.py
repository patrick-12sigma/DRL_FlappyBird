#!/usr/bin/env python
import argparse
import os
import tensorflow as tf
import cv2
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque


class Estimator(object):
    """This is the deep CNN model for both the q-estimator and the target q-estimator"""
    def __init__(self, scope='estimator', summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self.build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, scope)
                os.makedirs(summary_dir, exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def build_model(self):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512, ACTIONS])
        b_fc2 = self.bias_variable([ACTIONS])

        # input layer
        self.s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        self.output_op = tf.matmul(h_fc1, W_fc2) + b_fc2

        # get loss and train_op
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])

        # a is one-hot, this inner product is to select the q value only for selected action
        readout_action = tf.reduce_sum(tf.multiply(self.output_op, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - readout_action))
        self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss, global_step=tf.train.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('max_q', tf.reduce_max(self.output_op))
        ])

    def predict(self, sess, s_t):
        """Predict q-values given a certain state

        Args:
            sess:
            s_t: [None, 80, 80, 4]

        Returns:
            q_values: [None, ACTIONS]

        """
        q_values = sess.run(self.output_op, feed_dict={self.s: s_t})
        return q_values

    def update(self, sess, s_j_batch, a_batch, y_batch):
        _, summaries, global_step = sess.run(
            [self.train_op, self.summaries, tf.train.get_global_step()],
            feed_dict={
                self.y: y_batch,
                self.a: a_batch,
                self.s: s_j_batch}
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()


def copy_model_parameters(sess, q_estimator, target_q_estimator):
    """Copy the parameters from q_estimator to target_q_estimator

    Args:
        sess:
        q_estimator:
        target_q_estimator:

    Returns:

    """

    def get_params(estimator):
        params = [t for t in tf.trainable_variables() if t.name.startswith(estimator.scope)]
        params = sorted(params, key=lambda t: t.name)
        return params
    params = get_params(q_estimator)
    target_params = get_params(target_q_estimator)

    assign_ops = []
    for t, target_t in zip(params, target_params):
        assign_op = tf.assign(t, value=target_t)
        assign_ops.append(assign_op)
    sess.run(assign_ops)


def preprocess_state(x_t):
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t


def train_dqn(sess, q_estimator, target_q_estimator, game_level='hard', show_play=True, FPS=30, save_dir=None,
              fix_target=True):
    """Train

    Args:
        sess:
        q_estimator:
        target_q_estimator:
        game_level:
        speedup_level:
        save_dir:

    Returns:

    """
    # open up a game state to communicate with emulator
    game_state = game.GameState(game_level=game_level, show_play=show_play, FPS=FPS)

    # store the previous observations in replay memory
    replay_memory = deque(maxlen=REPLAY_MEMORY)

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, done = game_state.frame_step(do_nothing)
    x_t = preprocess_state(x_t)
    s_t = np.stack([x_t] * 4, axis=2)

    # save and load networks
    saver = tf.train.Saver(max_to_keep=None) # save all checkpoints
    checkpoint = tf.train.get_checkpoint_state(save_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    episode_reward = 0
    episode_score = 0
    episode_length = 0
    while t < MAXITER:
        # choose an action epsilon greedily
        q_values_t = q_estimator.predict(sess, [s_t])[0]
        a_t = np.zeros([ACTIONS])

        # epsilon-greedy policy
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        # skip frames; act every FRAME_PER_ACTION
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                action_index = np.argmax(q_values_t)
        else:
            action_index = 0 # do nothing
        a_t[action_index] = 1

        # write epsilon to tensorboard
        simple_summary = tf.Summary()
        simple_summary.value.add(simple_value=epsilon, tag='epsilon')
        q_estimator.summary_writer.add_summary(simple_summary, t)

        # run the selected action and observe next state and reward
        x_t1, r_t, done = game_state.frame_step(a_t)
        episode_reward += r_t
        episode_length += 1
        if r_t == 1:
            print("*" * 80 + "SCORE!!****************")
            episode_score += 1
        x_t1 = preprocess_state(x_t1)
        s_t1 = np.append(x_t1[..., np.newaxis], s_t[..., :3], axis=2)

        # store the transition in replay_memory
        replay_memory.append((s_t, a_t, r_t, s_t1, done))

        # episode ends, write summary to tensorboard
        if done:
            simple_summary = tf.Summary()
            simple_summary.value.add(simple_value=episode_reward, tag='episode_reward')
            simple_summary.value.add(simple_value=episode_score, tag='episode_score')
            simple_summary.value.add(simple_value=episode_length, tag='episode_length')
            q_estimator.summary_writer.add_summary(simple_summary, t)
            q_estimator.summary_writer.flush()
            print('>' * 100 + 'episode_score {}, episode_reward {}'.format(episode_score, episode_reward))
            episode_reward = 0
            episode_score = 0
            episode_length = 0

        # train q_estimator
        if t > OBSERVE:
            # update target_q_estimator periodically
            if t % UPDATE_TARGET_ESTIMATOR_EVERY == 0 and fix_target:
                copy_model_parameters(sess, q_estimator, target_q_estimator)

            # sample a minibatch to train on
            minibatch = random.sample(list(replay_memory), BATCH)

            # get the batch variables
            s_j_batch, a_batch, r_batch, s_j1_batch, done_batch = zip(*minibatch)
            # s_j_batch = [d[0] for d in minibatch]
            # a_batch = [d[1] for d in minibatch]
            # r_batch = [d[2] for d in minibatch]
            # s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            if fix_target:
                q_values_j1_batch = target_q_estimator.predict(sess, s_j1_batch)
            else:
                q_values_j1_batch = q_estimator.predict(sess, s_j1_batch)
            for i in range(0, len(minibatch)):
                done = minibatch[i][4]
                # if terminal is done, only equals reward
                if done:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(q_values_j1_batch[i]))

            # update q estimator
            q_estimator.update(sess, s_j_batch, a_batch, y_batch)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % SAVE_MODEL_EVERY == 0:
            saver.save(sess, save_dir + '/' + GAME + '-dqn', global_step = t)

        # print info
        print("TIMESTEP {} / EPSILON {} / ACTION {} / REWARD {} Q_MAX {}".format(
            t, epsilon, action_index, r_t, np.max(q_values_t)))


if __name__ == "__main__":
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='deploy', type=str, help='train or deploy')
    parser.add_argument('--gpu', default='1', type=str, metavar='N', help='use gpu')
    args = parser.parse_args()

    # set global constants
    UPDATE_TARGET_ESTIMATOR_EVERY = 10000
    SAVE_MODEL_EVERY = 10000
    REPLAY_MEMORY = 50000  # number of previous transitions to remember
    GAME = 'bird'  # the name of the game being played for log files
    ACTIONS = 2  # number of valid actions
    BATCH = 32  # size of minibatch
    GAMMA = 0.99  # decay rate of past observations
    MAXITER = 3000000 # max number of frames to train/play
    EXPLORE = 2000000 # number of episodes to perform epsilon decay
    if args.task == 'train':
        OBSERVE = 10000.  # timesteps to observe before training
        FINAL_EPSILON = 0.0001  # final value of epsilon
        INITIAL_EPSILON = 0.1  # starting value of epsilon
        FRAME_PER_ACTION = 1
        FPS = -1
        show_play = False

    elif args.task == 'deploy':
        OBSERVE = 3000000.  # timesteps to observe before training
        FINAL_EPSILON = 0.0001  # final value of epsilon
        INITIAL_EPSILON = 0.0001  # starting value of epsilon
        # INITIAL_EPSILON = 0.03  # starting value of epsilon
        FRAME_PER_ACTION = 1
        FPS = -1
        show_play = True
    else:
        raise ValueError('task can only be train or deploy')

    # set gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    save_dir = 'saved_networks_fixed_target'
    summaries_dir = './summaries'
    game_level = 'hard'

    fix_target = True
    # create a glboal step variable
    # NB. this should be done before creating the q_estimator!
    tf.train.create_global_step()

    # create q_estimator and target_q_estimator
    q_estimator = Estimator(scope='estimator', summaries_dir=summaries_dir)
    target_q_estimator = Estimator(scope='target_estimator', summaries_dir=summaries_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_dqn(sess, q_estimator, target_q_estimator,
                  save_dir=save_dir, game_level=game_level, show_play=show_play, FPS=FPS, fix_target=fix_target)
