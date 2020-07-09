from __future__ import print_function

import argparse
import os
import random
import sys
from pathlib import Path
from time import localtime, strftime

import numpy as np
import psutil
import tensorflow as tf

import gridworld
import math
import replay_buffer
import itertools

import glob
import shutil

import glob

import gspread
from oauth2client.service_account import ServiceAccountCredentials

import time

scripts_dir = os.path.dirname(os.path.abspath(__file__))
local_workspace_dir = os.path.join(str(Path(scripts_dir)), 'data')

print('{} (Scripts directory)'.format(scripts_dir))
print('{} (Local Workspace directory)'.format(local_workspace_dir))

import cv2

cv2.ocl.setUseOpenCL(False)

os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'

session_config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)

# ======================================================================================================================

def copy_scripts(target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    files = glob.iglob(os.path.join(scripts_dir, '*.py'))
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, target_directory)

def sample_noise(shape):
    noise = tf.random_normal(shape)
    return noise


def noisy_dense(x, size, name, bias=True, activation_fn=tf.identity):

    # the function used in eq.7,8
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initializer of \mu and \sigma
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
    # Sample noise from gaussian
    p = sample_noise([x.get_shape().as_list()[1], 1])
    q = sample_noise([1, size])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

    # w = w_mu + w_sigma*w_epsilon
    w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init)
    w_sigma = tf.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
    w = w_mu + tf.multiply(w_sigma, w_epsilon)
    ret = tf.matmul(x, w)
    if bias:
        # b = b_mu + b_sigma*b_epsilon
        b_mu = tf.get_variable(name + "/b_mu", [size], initializer=mu_init)
        b_sigma = tf.get_variable(name + "/b_sigma", [size], initializer=sigma_init)
        b = b_mu + tf.multiply(b_sigma, b_epsilon)
        return activation_fn(ret + b), w_mu, w_sigma
    else:
        return activation_fn(ret)

# ======================================================================================================================

class DQN(object):
    def __init__(self, config, stats, id):

        self.config = config
        self.stats = stats

        self.id = id  # Owner of this model

        # Scoped names, to make them possible to be loaded at once.
        self.name = 'AGENT_' + str(self.id) + '/' + 'QNET_ONLINE'
        self.name_t = 'AGENT_' + str(self.id) + '/' + 'QNET_TARGET'

        print(self.name, self.name_t)

        self.self_n_post_init_steps = 0
        self.self_n_training_steps = 0

        self.s, self.q_values, self.mu, self.sigma  = self.build_network(self.name)
        self.s_t, self.q_values_t, _, _ = self.build_network(self.name_t)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        trainable_vars_by_name = {var.name[len(self.name):]: var for var in trainable_vars}

        trainable_vars_t = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_t)
        trainable_vars_by_name_t = {var.name[len(self.name_t):]: var for var in trainable_vars_t}

        copy_ops = [target_var.assign(trainable_vars_by_name[var_name])
                    for var_name, target_var in trainable_vars_by_name_t.items()]
        self.update_target_weights = tf.group(*copy_ops)

        self.a, self.y, self.loss, self.grads_update, self.error = self.build_training_op()

        self.s_exp_f, self.val_exp_f = self.build_exploration_network('AGENT_' + str(self.id) + '/' + 'EXPNET_FIXED', 3.0)
        self.s_exp_m, self.val_exp_m = self.build_exploration_network('AGENT_' + str(self.id) + '/' + 'EXPNET_ONLINE', 1.0)

        self.exp_y, self.exp_loss, self.exp_grads_update, self.exp_error = self.build_exp_training_op()

        # self.replay_memory = Memory(self.config['replay_memory_capacity'])
        self.replay_memory = replay_buffer.PrioritizedReplayBuffer(self.config['replay_memory_capacity'],
                                                                   alpha=0.6)

        self.replay_memory_ts = replay_buffer.ReplayBuffer(8)


        self.replay_memory_beta = 0.5
        self.replay_memory_beta_inc = (1.0 - self.replay_memory_beta)/float(self.config['n_optimizer_steps'])

        #self.per_alpha = 0.6
        #self.per_alpha_teacher = 0.9

    def init(self):
        pass

    def save(self):
        pass

    def restore(self):
        pass

    def policy_net(self, scope, inputs, n_actions):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            adv_layer_1, _, _ = noisy_dense(inputs,
                                          size=self.config['hidden_layer_size'],
                                          bias=True,
                                          activation_fn=tf.nn.relu,
                                          name='adv_layer_1')

            adv_layer_2, _, _ = noisy_dense(adv_layer_1,
                                          size=n_actions,
                                          bias=True,
                                          name='adv_layer_2')

            val_layer_1, _, _ = noisy_dense(inputs,
                                          size=self.config['hidden_layer_size'],
                                          bias=True,
                                          activation_fn=tf.nn.relu,
                                          name='val_layer_1')

            val_layer_2, mu, sigma = noisy_dense(val_layer_1,
                                          size=1,
                                          bias=True,
                                          name='val_layer_2')

            advt = (adv_layer_2 - tf.reduce_mean(adv_layer_2, axis=-1, keepdims=True))
            value = tf.tile(val_layer_2, [1, n_actions])
            output = advt + value

            return output, mu, sigma

    def exploration_net(self, scope, inputs, std):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_1 = tf.layers.dense(inputs,
                                      units=256,
                                      use_bias=True,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.initializers.random_normal(stddev=std),
                                      name='dense_layer_1')

            layer_2 = tf.layers.dense(layer_1,
                                      units=1,
                                      use_bias=True,
                                      activation=None,
                                      kernel_initializer=tf.initializers.random_normal(stddev=std),
                                      name='dense_layer_2')

            return layer_2

    def build_network(self, name):
        input = tf.placeholder(tf.float32,
                                [None, 2 * (self.config['n_agents'] - 1 + self.config['n_landmarks']) + 1],
                               #[None, 2 * (self.config['n_agents'] - 1 + self.config['n_landmarks'])],
                               name=name + '_input')

        q_values, mu, sigma = self.policy_net(name, input, self.config['n_actions'])
        return input, q_values, mu, sigma

    def build_exploration_network(self, name, mean):
        if self.config['observation_type'] == 'vector':
            input = tf.placeholder(tf.float32,
                                    [None, 2 * (self.config['n_agents'] - 1 + self.config['n_landmarks']) + 1],
                                   #[None, 2 * (self.config['n_agents'] - 1 + self.config['n_landmarks']) + 1],
                                   name=name + '_input')

            value = self.exploration_net(name, input, mean)
            return input, value

    def build_training_op(self):
        # Make this scope too
        a = tf.placeholder(tf.int32, [None], name='a_' + str(self.id))
        y = tf.placeholder(tf.float32, [None], name='y_' + str(self.id))
        #c = tf.placeholder(tf.float32, [None], name='c_' + str(self.id))

        a_one_hot = tf.one_hot(a, self.config['n_actions'], 1.0, 0.0)
        q_values = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        error = tf.abs(y - q_values)

        loss = tf.losses.huber_loss(labels=y, predictions=q_values, delta=self.config['huber_loss_delta'])

        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'], epsilon=self.config['adam_epsilon'])
        grads_update = optimizer.minimize(loss)

        return a, y, loss, grads_update, error

    def build_exp_training_op(self):
        y = tf.placeholder(tf.float32, [None, 1], name='y_' + str(self.id))

        error = tf.abs(y - self.val_exp_m)
        loss = tf.losses.mean_squared_error(labels=y, predictions=self.val_exp_m)

        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'], epsilon=self.config['adam_epsilon'])
        #optimizer = tf.train.GradientDescentOptimizer(0.0002)
        grads_update = optimizer.minimize(loss)

        return y, loss, grads_update, error

    def get_error(self):
        pass

    def get_q_values(self, observation):
        return session.run(self.q_values, feed_dict={self.s: [observation.astype(dtype=np.float32)]})[0]

    def greedy_action(self, observation):
        return np.argmax(self.get_q_values(observation))

    def random_action(self):
        return random.randrange(self.config['n_actions'])

    def observe(self, experience, is_advised):

        self.replay_memory.add(experience[0],
                                experience[1],
                                experience[2],
                                experience[3],
                                experience[4],
                                is_advised)

        #if is_advised:
        #    self.replay_memory_ts.add(experience[0],
        #                        experience[1],
        #                        experience[2],
        #                        experience[3],
        #                        experience[4],
        #                        is_advised)

        if self.replay_memory.__len__() >= self.config['replay_memory_init_size']:

            if self.id == 0:  # TODO Move this to controller object
                self.stats.n_post_init_steps += 1
            self.self_n_post_init_steps += 1

            if self.self_n_post_init_steps % self.config['train_period'] == 0:
                self.train_network()

                if self.self_n_training_steps % self.config['target_update_period'] == 0:
                    session.run(self.update_target_weights)


    def get_state_uncertainty(self, observation):
        feed_dict = {}
        feed_dict.update({self.s_exp_f: [observation.astype(dtype=np.float32)]})
        feed_dict.update({self.s_exp_m: [observation.astype(dtype=np.float32)]})
        val_exp_f, val_exp_t = session.run([self.val_exp_f, self.val_exp_m], feed_dict=feed_dict)
        return np.sqrt(np.power(val_exp_f[0][0] - val_exp_t[0][0], 2))


    def train_network(self):
        if self.id == 0:
            self.stats.n_training_steps += 1
        self.self_n_training_steps += 1

        if self.replay_memory_ts.__len__() > 1:
            bs = self.config['batch_size'] - self.replay_memory_ts.__len__()
            bs_ts = self.replay_memory_ts.__len__()
        else:
            bs = self.config['batch_size']
            bs_ts = 0

        minibatch = self.replay_memory.sample(bs, beta=self.replay_memory_beta)
        self.replay_memory_beta += self.replay_memory_beta_inc

        if bs_ts > 0:
            minibatch_ts = self.replay_memory_ts.sample(bs_ts)

            state_batch = np.concatenate((minibatch[0], minibatch_ts[0]), axis=0)
            action_batch = np.concatenate((minibatch[1], minibatch_ts[1]), axis=0)
            reward_batch = np.concatenate((minibatch[2], minibatch_ts[2]), axis=0)
            next_state_batch =np.concatenate((minibatch[3], minibatch_ts[3]), axis=0)
            terminal_batch = np.concatenate((minibatch[4], minibatch_ts[4]), axis=0)
            is_advised_batch = np.concatenate((minibatch[5], minibatch_ts[5]), axis=0)
            batch_idx = minibatch[7]

        else:
            state_batch = minibatch[0]
            action_batch = minibatch[1]
            reward_batch = minibatch[2]
            next_state_batch = minibatch[3]
            terminal_batch = minibatch[4]
            is_advised_batch = minibatch[5]
            #weights = minibatch[6]
            batch_idx = minibatch[7]

        feed_dict = {}
        feed_dict.update({self.s: next_state_batch.astype(dtype=np.float32)})
        feed_dict.update({self.s_t: next_state_batch.astype(dtype=np.float32)})
        feed_dict.update({self.s_exp_f: state_batch.astype(dtype=np.float32)})
        feed_dict.update({self.s_exp_m: state_batch.astype(dtype=np.float32)})

        q_values_next_batch, q_values_next_t_batch, exp_y, _ = \
            session.run([self.q_values, self.q_values_t, self.val_exp_f, self.val_exp_m], feed_dict=feed_dict)

        action_next_batch = np.argmax(q_values_next_batch, axis=1)

        #q_values = None
        #if self.config['target_correcting']:
        #    feed_dict = {}
        #    feed_dict.update({self.s: state_batch.astype(dtype=np.float32)})
        #    q_values = session.run([self.q_values], feed_dict=feed_dict)
        #    q_values = np.squeeze(q_values)

        y_batch = []
        for j in range(len(reward_batch)):
            target = reward_batch[j] + (1.0 - terminal_batch[j]) * self.config['gamma'] * \
                     q_values_next_t_batch[j][action_next_batch[j]]

            #if self.config['target_correcting'] and is_advised_batch[j]:  # Target correcting
            #    if target < np.max(q_values[j]):
            #        target = np.max(q_values[j])

            y_batch.append(target)

        feed_dict = {}
        feed_dict.update({self.s: state_batch.astype(dtype=np.float32)})
        feed_dict.update({self.a: action_batch})
        feed_dict.update({self.y: y_batch})
        feed_dict.update({self.s_exp_m: state_batch.astype(dtype=np.float32)})
        feed_dict.update({self.exp_y: exp_y})

        #print(y_batch)
        #print(feed_dict)

        loss, _, error, exp_loss, _, exp_error = session.run([self.loss, self.grads_update, self.error,
                                                              self.exp_loss, self.exp_grads_update, self.exp_error],
                                                             feed_dict=feed_dict)

        #print(np.abs(error))


        self.replay_memory.update_priorities(batch_idx, np.abs(error[:bs]) + float(1e-6))
            #self.replay_memory_ts.update_priorities(batch_idx[batch_size:], np.abs(error[batch_size:]) + float(1e-6))


        self.stats.agent_loss[self.id] += loss

        self.stats.average_loss += (loss / self.config['n_agents'])

# ======================================================================================================================

class Statistics(object):
    def __init__(self, n_agents):
        # Number of environment interactions
        self.n_env_steps = 0
        self.n_env_steps_var = tf.Variable(0)

        # Number of environment interactions - after initialization step
        self.n_post_init_steps = 0
        self.n_post_init_steps_var = tf.Variable(0)

        # Number of training steps
        self.n_training_steps = 0
        self.n_training_steps_var = tf.Variable(0)

        # Number of episodes
        self.n_episodes = 0
        self.n_episodes_var = tf.Variable(0)



        self.agent_loss = [0.0 for _ in range(n_agents)]
        self.agent_loss_var = [tf.Variable(0.) for _ in range(n_agents)]
        self.agent_loss_ph = [tf.placeholder(tf.float32) for _ in range(n_agents)]

        self.agent_advices_given = [0.0 for _ in range(n_agents)]
        self.agent_advices_given_var = [tf.Variable(0.) for _ in range(n_agents)]
        self.agent_advices_given_ph = [tf.placeholder(tf.float32) for _ in range(n_agents)]

        self.agent_advices_given_10k = [0.0 for _ in range(n_agents)]
        self.agent_advices_taken_10k = [0.0 for _ in range(n_agents)]

        self.agent_advices_taken = [0.0 for _ in range(n_agents)]
        self.agent_advices_taken_var = [tf.Variable(0.) for _ in range(n_agents)]
        self.agent_advices_taken_ph = [tf.placeholder(tf.float32) for _ in range(n_agents)]

        self.agent_advices_given_10k = [0.0 for _ in range(n_agents)]
        self.agent_advices_taken_10k = [0.0 for _ in range(n_agents)]
        self.agent_advices_given_20k = [0.0 for _ in range(n_agents)]
        self.agent_advices_taken_20k = [0.0 for _ in range(n_agents)]

        self.average_loss = 0.0
        self.average_loss_var = tf.Variable(0.)
        self.average_loss_ph = tf.placeholder(tf.float32)


        # Training episodes
        self.episode_reward = 0.0
        self.episode_reward_var = tf.Variable(0.)
        self.episode_reward_ph = tf.placeholder(tf.float32)

        self.episode_duration = 0
        self.episode_duration_var = tf.Variable(0.)
        self.episode_duration_ph = tf.placeholder(tf.float32)


        # Number of evaluations
        self.n_evaluations = 0
        self.n_evaluations_var = tf.Variable(0)

        # Evaluation
        self.evaluation_score = 0.0
        self.evaluation_score_var = tf.Variable(0.)
        self.evaluation_score_ph = tf.placeholder(tf.float32)

        self.evaluation_score_last = 0.0

        self.evaluation_score_auc = 0.0
        self.evaluation_score_auc_var = tf.Variable(0.)
        self.evaluation_score_auc_ph = tf.placeholder(tf.float32)

        self.agent_mean_uncertainty = [0.0 for _ in range(n_agents)]
        self.agent_mean_uncertainty_var = [tf.Variable(0.) for _ in range(n_agents)]
        self.agent_mean_uncertainty_ph = [tf.placeholder(tf.float32) for _ in range(n_agents)]

        self.auc_to_sh = []
        self.final_to_sh = []

        self.summary_op_step, self.summary_op_episode, self.summary_op_evaluation = self.setup_summary(n_agents)

    def setup_summary(self, n_agents):
        average_loss_sc = tf.summary.scalar('Average Loss', self.average_loss_var)
        agent_loss_sc = [tf.summary.scalar('Agent Loss ' + str(i), self.agent_loss_var[i]) for i in range(n_agents)]

        agent_advices_given_sc = [tf.summary.scalar('Advices Given ' + str(i), self.agent_advices_given_var[i]) for i in range(n_agents)]
        agent_advices_taken_sc = [tf.summary.scalar('Advices Taken ' + str(i), self.agent_advices_taken_var[i]) for i in
                                  range(n_agents)]

        episode_reward_sc = tf.summary.scalar('Episode Reward', self.episode_reward_var)
        episode_duration_sc = tf.summary.scalar('Episode Duration', self.episode_duration_var)

        evaluation_score_sc = tf.summary.scalar('Evaluation Score', self.evaluation_score_var)
        evaluation_score_auc_sc = tf.summary.scalar('Evaluation Score AUC', self.evaluation_score_auc_var)
        agent_mean_uncertainty_sc = [tf.summary.scalar('Mean Uncertainty ' + str(i),
                                                     self.agent_mean_uncertainty_var[i]) for i in range(n_agents)]

        return tf.summary.merge([average_loss_sc] + agent_loss_sc), \
               tf.summary.merge([episode_reward_sc, episode_duration_sc] + agent_advices_given_sc + agent_advices_taken_sc), \
               tf.summary.merge([evaluation_score_sc, evaluation_score_auc_sc] + agent_mean_uncertainty_sc)


# ======================================================================================================================


class Agent(object):
    def __init__(self, config, statistics, id, type):
        assert type in {'dqn', 'expert'}

        self.config = config
        self.statistics = statistics
        self.id = id
        self.type = type
        self.name = 'AGENT_' + str(id)
        self.model = DQN(config, statistics, id)

        self.agents = []  # all agents - order == id
        self.other_agents = []

        self.last_action_is_advised = False

        self.state_importances = []
        self.evaluation_uncertainties = []

        # ==============================================================================================================

        self.budget_ask = config['budget_ask']
        self.budget_give = config['budget_give']

        self.initial_model = None

        # ==============================================================================================================

        #self.n_steps = 0
        #self.advices_given = [0, 0, 0]
        #self.advices_taken = [0, 0, 0]

    def save_initial_state(self):
        filepath = os.path.join(save_summary_path, 'agent_'+str(self.id))
        file = open(filepath, "w")
        file.write("{}: {}\n".format('initial_model', self.initial_model))
        file.close()

    def best_action(self, observation):
        agents = [None] * self.config['n_agents']
        landmarks = [None] * self.config['n_landmarks']

        for i, _ in enumerate(agents):
            if i == 0:
                agents[i] = np.array([0, 0])
            else:
                start = (i - 1) * 2
                end = start + 2
                agents[i] = np.array(observation[start:end])

        for i, _ in enumerate(landmarks):
            start = (self.config['n_agents'] - 1) * 2 + (i) * 2
            end = start + 2
            landmarks[i] = np.array(observation[start:end])

        distances = np.zeros((self.config['n_agents'], self.config['n_landmarks']), dtype=np.float32)

        for i, agent in enumerate(agents):
            for j, landmark in enumerate(landmarks):
                distances[i, j] = np.sum(np.absolute(agent - landmark))

        agent_p = list(itertools.permutations(range(self.config['n_agents'])))
        landmark_p = list(itertools.permutations(range(self.config['n_landmarks'])))

        pairings_list = []
        for i in range(len(agent_p)):
            for j in range(len(landmark_p)):
                pairings = []
                for k in range(min(self.config['n_agents'], self.config['n_landmarks'])):
                    pairings.append((agent_p[i][k], landmark_p[j][k]))
                pairings_list.append(sorted(pairings))

        pairings_set = set(tuple(i) for i in pairings_list)

        min_cost = np.inf
        min_cost_pairing = None
        for pairing in pairings_set:
            cost = 0.0
            for pair in pairing:
                cost += distances[pair[0], pair[1]]

            if cost < min_cost:
                min_cost = cost
                min_cost_pairing = pairing

        target_landmark = min_cost_pairing[0][1]

        action = 0

        if landmarks[target_landmark][0] > 0:
            action = 1
        elif landmarks[target_landmark][0] < 0:
            action = 2

        if landmarks[target_landmark][1] > 0:
            action = 3
        elif landmarks[target_landmark][1] < 0:
            action = 4

        return action

    def restore(self, model_name, checkpoint):  # TODO: Make source agent selectable.
        print('restore', len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)))
        loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
        loader.restore(session, os.path.join(os.path.join(models_dir, model_name), 'model-'+str(checkpoint)+'.ckpt'))
        self.initial_model = model_name+'_'+str(checkpoint)

    def observe(self, experience):
        self.model.observe(experience, self.last_action_is_advised)

    def get_state_importance(self, observation):
        q_values = self.model.get_q_values(observation)
        return np.max(q_values) - np.min(q_values)

    def get_state_uncertainty(self, observation):
        return self.model.get_state_uncertainty(observation)

    def pick_action(self, observation, evaluation=False):

        uncertainty = self.get_state_uncertainty(observation)

        #state_importance = self.get_state_importance(observation)
        #if self.id == 0:
        #     self.state_importances.append(state_importance)

        if evaluation:
            self.evaluation_uncertainties.append(uncertainty)
            return self.model.greedy_action(observation)
        else:
            action = None
            intended_action = self.model.greedy_action(observation)
            # print("intended_action:", intended_action)

            if self.config['use_teaching'] and \
                    self.budget_ask > 0 and \
                    uncertainty >= self.config['threshold_ask']:

                # print("ask for actions")
                if self.config['advice_asking_mode'] == 0:  # Oracle advice
                    action = self.best_action(observation)

                elif self.config['advice_asking_mode'] == 1:  # Always ask the expert agent (preset, agent 0)
                    if self.id != 0 and self.agents[0].budget_give > 0:
                        action = self.agents[0].model.greedy_action(observation)
                        self.statistics.agent_advices_given[0] += 1
                        self.agents[0].budget_give -= 1

                elif self.config['advice_asking_mode'] == 2:  # Always ask the agents with smallest uncertainty
                    uncertainties = [agent.get_state_uncertainty(observation) for agent in self.agents]
                    best_agent = int(np.argmin(uncertainties))

                    #if self.id == 0:
                    #    print(uncertainties)

                    if best_agent != self.id and self.agents[best_agent].budget_give > 0:
                        action = self.agents[best_agent].model.greedy_action(observation)
                        self.statistics.agent_advices_given[best_agent] += 1
                        self.agents[best_agent].budget_give -= 1

                elif self.config['advice_asking_mode'] == 3:  # Ask every agent - Realistic scenario

                    advices = [agent.give_advice(observation, intended_action) for agent in self.other_agents]
                    advices = list(filter(None.__ne__, advices))


                    if advices:
                        # print("ask for advices:", advices)
                        advices = np.array(advices)
                        bin_count = np.bincount(advices, minlength=5)
                        action = np.random.choice(np.flatnonzero(bin_count == bin_count.max()))

            if action is None:
                self.last_action_is_advised = False
                action = intended_action
            else:
                self.last_action_is_advised = True
                self.statistics.agent_advices_taken[self.id] += 1
                self.budget_ask -= 1

            return action


    def give_advice(self, observation, intended_action):

        advice = None
        uncertainty = self.get_state_uncertainty(observation)
        state_importance = self.get_state_importance(observation)

        if self.budget_give > 0 and uncertainty <= self.config['threshold_give']:

            if self.config['advice_giving_mode'] == 0:  # Early advising
                advice = self.model.greedy_action(observation)

            elif self.config['advice_giving_mode'] == 1:  # Importance advising
                if state_importance >= self.config['importance_threshold_give']:
                    advice = self.model.greedy_action(observation)

            elif self.config['advice_giving_mode'] == 2:  # Mistake correcting
                advice = self.model.greedy_action(observation)
                if advice != intended_action and state_importance >= self.config['importance_threshold_give']:
                    pass
                else:  # No advice
                    advice = None

        if advice is not None:
            self.statistics.agent_advices_given[self.id] += 1
            self.budget_give -= 1

        return advice

# ======================================================================================================================

# A centralized wrapper to control the decentralized executions of agents.

class Controller(object):
    def __init__(self, config, save_summary_path, save_visualizations_path):
        self.config = config
        self.save_summary_path = save_summary_path
        self.save_visualizations_path = save_visualizations_path

        self.stats = Statistics(self.config['n_agents'])
        self.agents = [Agent(self.config, self.stats, id, 'dqn') for id in range(self.config['n_agents'])]

        # Create link between agents
        for agent in self.agents:
            agent.agents = self.agents

        for agent in self.agents:
            for i, agent_i in enumerate(self.agents):
                if i != agent.id:
                    agent.other_agents.append(agent_i)

        for agent in self.agents:
            print(agent.agents)
            print(agent.other_agents)

        self.evaluation_dir = None

        self.video_frames = 0
        self.video_average_image = None

        self.asop_1 = self.stats.episode_reward_var.assign(self.stats.episode_reward_ph)
        self.asop_2 = self.stats.episode_duration_var.assign(self.stats.episode_duration_ph)

        self.asop_3 = self.stats.average_loss_var.assign(self.stats.average_loss_ph)

        self.asops = []
        for i in range(self.config['n_agents']):
            self.asops.append(self.stats.agent_loss_var[i].assign(self.stats.agent_loss_ph[i]))

        self.asop_4 = self.stats.evaluation_score_var.assign(self.stats.evaluation_score_ph)
        self.asop_5 = self.stats.evaluation_score_auc_var.assign(self.stats.evaluation_score_auc_ph)

        self.asopcert = []
        for i in range(self.config['n_agents']):
            self.asopcert.append(self.stats.agent_mean_uncertainty_var[i].assign(self.stats.agent_mean_uncertainty_ph[i]))

        self.asopsag = []
        for i in range(self.config['n_agents']):
            self.asopsag.append(self.stats.agent_advices_given_var[i].assign(self.stats.agent_advices_given_ph[i]))

        self.asopsat = []
        for i in range(self.config['n_agents']):
            self.asopsat.append(self.stats.agent_advices_taken_var[i].assign(self.stats.agent_advices_taken_ph[i]))


    def init(self, evaluation=False):
        if not evaluation:
            self.stats.episode_duration = 0
            self.stats.episode_reward = 0
            self.stats.n_episodes += 1

    def get_used_budget(self):
        budgets = [[self.config['budget_ask'] - agent.budget_ask, self.config['budget_give'] - agent.budget_give]
                   for agent in self.agents]
        return np.mean(budgets, axis=0).astype(str)

    def observe(self, experience):
        for i, agent in enumerate(self.agents):
            agent.observe((experience[i][0],  # previous_observation
                           experience[i][1],  # action
                           experience[i][2],  # reward
                           experience[i][3],  # observation
                           experience[i][4],  # done
                           ))

        self.stats.episode_reward += experience[0][2]  # Notion of global reward is lost.
        self.stats.episode_duration += 1

    def pick_action(self, observation, evaluation=False):
        return [agent.pick_action(observation[i], evaluation) for i, agent in enumerate(self.agents)]

    def write_video(self, images, filename):
        video = cv2.VideoWriter(filename + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (500, 500))
        for image in images:
            video.write(image)
        video.release()

    def update_summary_step(self):
        if self.stats.n_training_steps > 0 and \
                self.stats.n_post_init_steps % self.config['train_period'] == 0 and \
                self.stats.n_training_steps % self.config['summary_update_period'] == 0:

            requested_ops = []
            feed_dict = {}

            self.stats.average_loss /= float(self.config['summary_update_period'])
            requested_ops.append(self.asop_3)
            feed_dict.update({self.stats.average_loss_ph: self.stats.average_loss})

            for i in range(self.config['n_agents']):
                self.stats.agent_loss[i] /= float(self.config['summary_update_period'])
                requested_ops.append(self.asops[i])
                feed_dict.update({self.stats.agent_loss_ph[i]: self.stats.agent_loss[i]})

            # print(self.stats.n_training_steps, self.stats.n_post_init_steps, self.stats.average_loss)

            session.run(requested_ops, feed_dict=feed_dict)

            summary = session.run(self.stats.summary_op_step)
            summary_writer.add_summary(summary, self.stats.n_training_steps)

            self.stats.average_loss = 0.0
            self.stats.agent_loss = [0.0 for _ in range(self.config['n_agents'])]

    def update_summary_episode(self):

        requested_ops = []
        feed_dict = {}

        # Normalize episode score between 0 and 1
        #print(self.stats.episode_reward, max_score)
        #self.stats.episode_reward /= max_score

        requested_ops.append(self.asop_1)
        feed_dict.update({self.stats.episode_reward_ph: self.stats.episode_reward})

        requested_ops.append(self.asop_2)
        feed_dict.update({self.stats.episode_duration_ph: self.stats.episode_duration})

        for i in range(self.config['n_agents']):
            requested_ops.append(self.asopsag[i])
            feed_dict.update({self.stats.agent_advices_given_ph[i]: self.stats.agent_advices_given[i]})

        for i in range(self.config['n_agents']):
            requested_ops.append(self.asopsat[i])
            feed_dict.update({self.stats.agent_advices_taken_ph[i]: self.stats.agent_advices_taken[i]})

        session.run(requested_ops, feed_dict=feed_dict)

        summary = session.run(self.stats.summary_op_episode)
        summary_writer.add_summary(summary, self.stats.n_episodes)

        #print(self.agents[0].state_importances)
        #self.agents[0].state_importances = []

        if self.stats.n_episodes == 10e3:
            for i in range(self.config['n_agents']):
                self.stats.agent_advices_taken_10k[i] = self.stats.agent_advices_taken[i]
                self.stats.agent_advices_given_10k[i] = self.stats.agent_advices_given[i]

        if self.stats.n_episodes == 20e3:
            for i in range(self.config['n_agents']):
                self.stats.agent_advices_taken_20k[i] = self.stats.agent_advices_taken[i]
                self.stats.agent_advices_given_20k[i] = self.stats.agent_advices_given[i]

    def update_summary_evaluation(self):

        requested_ops = [self.asop_4, self.asop_5]
        feed_dict = {self.stats.evaluation_score_ph: self.stats.evaluation_score,
                     self.stats.evaluation_score_auc_ph: self.stats.evaluation_score_auc}

        for i in range(self.config['n_agents']):
            requested_ops.append(self.asopcert[i])
            feed_dict.update({self.stats.agent_mean_uncertainty_ph[i]: self.stats.agent_mean_uncertainty[i]})

        session.run(requested_ops, feed_dict=feed_dict)

        summary = session.run(self.stats.summary_op_evaluation)
        # summary_writer.add_summary(summary, self.stats.n_evaluations)
        summary_writer.add_summary(summary, self.stats.n_episodes)

    def save_model(self):
        model_path = os.path.join(save_model_path, 'model-{}.ckpt').format(self.stats.n_episodes)
        print('[{}] Saving model... {}'.format(self.stats.n_episodes, model_path))
        saver.save(session, model_path)

        # print('New summary.')
        # summary_writer = tf.summary.FileWriter(save_summary_path, session.graph)


# ======================================================================================================================
def load_results_to_file(dir_name, file_name, values):
    with open(os.path.join(dir_name, file_name), 'w') as f:
        for val in values:
            f.write(str(val)+"\n")
        f.close()

def upload_results_to_spreadsheet(worksheet_name, values):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name('***TO BE FILLED***', scope)

    gc = gspread.authorize(credentials)
    spreadsheet = gc.open_by_key('***TO BE FILLED***')
    worksheet = spreadsheet.worksheet(worksheet_name)

    first_available_row = len(worksheet.col_values(1)) + 1

    cell_list = worksheet.range('A' + str(first_available_row) + ':AE' + str(first_available_row))

    for i, cell in enumerate(cell_list):
        cell.value = str(values[i])

    worksheet.update_cells(cell_list)

def print_memory_usage():
    print('-- RAM: {}'.format(process.memory_info().rss / (1024 * 1024)))


def save_config(config, filepath):
    fo = open(filepath, "w")
    for k, v in config.items():
        fo.write(str(k) + '>> ' + str(v) + '\n')
    fo.close()


def main(config):
    global session, summary_writer, scripts_dir, t_start, process, saver, save_model_path, models_dir, \
        max_score, e_max_score, save_summary_path, save_visualizations_path

    config['n_optimizer_steps'] = \
        int((config['t_max']*config['max_episodes'] - config['replay_memory_init_size'])/config['train_period'])

    print(config)

    print(config['use_teaching'])

    process = psutil.Process(os.getpid())

    env = gridworld.World(config['game_height'],
                          config['game_width'],
                          config['frame_stack_size'],
                          config['n_agents'],
                          config['n_landmarks'],
                          config['t_max'],
                          config['seed'],
                          config['evaluation_seed'],
                          config['n_evaluation_levels'])

    max_eval_scores = np.sum(env.evaluation_levels_scores)

    os.environ['PYTHONHASHSEED'] = str(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    tf.set_random_seed(config['seed'])

    run_id = config['run_id'] if config['run_id'] is not None else strftime("%Y%m%d-%H%M%S", localtime()) \
                                                                   + '-' + str(config['process_index'])
    print('Run ID: {}'.format(run_id))

    runs_local_dir = os.path.join(local_workspace_dir, 'Runs')
    if not os.path.exists(runs_local_dir):
        os.makedirs(runs_local_dir)

    summaries_dir = os.path.join(runs_local_dir, 'Summaries')
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

    models_dir = os.path.join(runs_local_dir, 'Models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    copy_scripts_dir = os.path.join(runs_local_dir, 'Scripts')
    if not os.path.exists(copy_scripts_dir):
        os.makedirs(copy_scripts_dir)

    visualizations_dir = os.path.join(runs_local_dir, 'Visualizations')
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    results_dir = os.path.join(runs_local_dir, 'Results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_summary_path = os.path.join(summaries_dir, run_id)
    save_model_path = os.path.join(models_dir, run_id)
    save_scripts_path = os.path.join(copy_scripts_dir, run_id)
    save_visualizations_path = os.path.join(visualizations_dir, run_id)
    save_results_path = os.path.join(results_dir, run_id)

    if not os.path.exists(save_visualizations_path):
        os.makedirs(save_visualizations_path)

    copy_scripts(save_scripts_path)

    # ==================================================================================================================

    controller = Controller(config, save_summary_path, save_visualizations_path)

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Number of parameters: {}'.format(total_parameters))

    session = tf.InteractiveSession(graph=tf.get_default_graph(), config=session_config)
    summary_writer = tf.summary.FileWriter(save_summary_path, session.graph)
    saver = tf.train.Saver()

    if config['run_id'] is not None:
        print('Restoring...')

        saver.restore(session, os.path.join(save_model_path, 'model.ckpt'))

        controller.stats.n_post_init_steps = controller.stats.n_post_init_steps_var.eval()
        controller.stats.n_training_steps = controller.stats.n_training_steps_var.eval()
        controller.stats.n_episodes = controller.stats.n_episodes_var.eval()

        print('\r DONE.')

    else:
        save_config(config, os.path.join(save_summary_path, 'config'))
        session.run(tf.global_variables_initializer())

    if config['agents_knowledge_state'] == 1:
        controller.agents[0].restore('20190527-161558-0', 20000)
        controller.agents[1].restore('20190527-161558-0', 10000)

    elif config['agents_knowledge_state'] == 2:
        controller.agents[0].restore('20190316-020132-1-1', 10000)
        controller.agents[1].restore('20190316-133250-2-2', 10000)
        controller.agents[2].restore('20190316-154911-3-3', 10000)

    elif config['agents_knowledge_state'] == 3:
        controller.agents[0].restore('20190314-182524-3', 20000)
        controller.agents[1].restore('20190314-182524-3', 20000)

    tf.get_default_graph().finalize()

    obs = None
    done = True

    while True:

        # the end of each episode
        if done:
            if controller.stats.n_episodes > 0:
                controller.update_summary_episode()

                if controller.stats.n_episodes % config['model_save_period'] == 0:
                    controller.save_model()

            # evaluate the policy at every 100 episodes
            if controller.stats.n_episodes % config['evaluation_period'] == 0:

                render = controller.stats.n_episodes > 0 and controller.stats.n_episodes % 10e3 == 0

                print("Running evaluation... " + str(controller.stats.n_episodes))

                controller.stats.n_evaluations += 1
                controller.evaluation_dir = \
                    os.path.join(controller.save_visualizations_path, str(controller.stats.n_episodes))

                if render:
                    os.mkdir(controller.evaluation_dir)

                e_obs = None
                e_done = True
                controller.stats.evaluation_score = 0.0
                e_episode_reward = 0

                # the number of evaluation episodes is n_evaluation_levels
                for evaluation_step in range(config['n_evaluation_levels']):
                    if e_done:
                        # do not add the episode reward and number of episodes
                        controller.init(evaluation=True)

                        # used the evaluation_step-th randomly initilized world map
                        e_obs = env.reset(evaluation_level_id=evaluation_step, render=render)
                        e_episode_reward = 0
                        e_done = False

                    while not e_done:

                        e_actions = controller.pick_action(e_obs, evaluation=True)
                        e_obs, e_reward, e_done = env.step(e_actions)
                        e_episode_reward += e_reward

                        if e_done:
                            # in the evaluation episode, the score is the cumulatived rewards
                            controller.stats.evaluation_score += e_episode_reward

                            if render:  # for each 10000 episodes
                                obs_images, obs_trace_image = env.get_visualization()
                                controller.write_video(obs_images,
                                                       os.path.join(controller.evaluation_dir, str(evaluation_step)))
                                cv2.imwrite(os.path.join(controller.evaluation_dir, str(evaluation_step)) + '.png',
                                            obs_trace_image)

                # after the evaluation step
                controller.stats.evaluation_score /= max_eval_scores

                if controller.stats.n_episodes > 0:
                    controller.stats.evaluation_score_auc += np.trapz([controller.stats.evaluation_score_last,
                                                                       controller.stats.evaluation_score])

                    if controller.stats.n_episodes % 1e3 == 0:
                        print("auc:", controller.stats.evaluation_score_auc, "--eval score:",controller.stats.evaluation_score
                              ,'--budget ask and give:', " ".join(controller.get_used_budget()))

                        # print(controller.stats.evaluation_score_auc, controller.stats.evaluation_score)

                    if controller.stats.n_episodes % 10e3 == 0:
                        controller.stats.auc_to_sh.append(controller.stats.evaluation_score_auc)
                        controller.stats.final_to_sh.append(controller.stats.evaluation_score)

                # the evaluation score of last evaluation period
                controller.stats.evaluation_score_last = controller.stats.evaluation_score

                for i, agent in enumerate(controller.agents):
                    controller.stats.agent_mean_uncertainty[i] = np.mean(np.asarray(agent.evaluation_uncertainties))
                    agent.evaluation_uncertainties = []

                controller.update_summary_evaluation()

                if controller.stats.n_episodes == config['max_episodes']:
                    break

            # the end of training episode
            if controller.stats.n_episodes < config['max_episodes']:
                # every time the world init, n_episodes + 1
                controller.init()

                obs = env.reset(curriculum_level=None)
                done = False
            else:
                break
        actions = controller.pick_action(obs)
        obs_prev = obs

        obs, reward, done = env.step(actions)

        # the experience of each agent
        controller.observe([(obs_prev[i], actions[i], reward, obs[i], float(done)) for i in range(config['n_agents'])])
        controller.update_summary_step()

    controller.save_model()

    print(controller.stats.auc_to_sh)
    print(controller.stats.final_to_sh)

    agents_knowledge_states = [[0, 0, 0], [20, 10, 0], [10, 10, 10]]

    hyperparameters_to_be_uploaded = [run_id, config,
     agents_knowledge_states[config['agents_knowledge_state']],
     str(int(controller.stats.agent_advices_taken_10k[0])),
     str(int(controller.stats.agent_advices_taken_10k[1])),
     str(int(controller.stats.agent_advices_taken_10k[2])),
     str(int(controller.stats.agent_advices_given_10k[0])),
     str(int(controller.stats.agent_advices_given_10k[1])),
     str(int(controller.stats.agent_advices_given_10k[2])),
     '{:.4f}'.format(controller.stats.auc_to_sh[0]),
     '{:.4f}'.format(controller.stats.final_to_sh[0]),
     str(int(controller.stats.agent_advices_taken_20k[0])),
     str(int(controller.stats.agent_advices_taken_20k[1])),
     str(int(controller.stats.agent_advices_taken_20k[2])),
     str(int(controller.stats.agent_advices_given_20k[0])),
     str(int(controller.stats.agent_advices_given_20k[1])),
     str(int(controller.stats.agent_advices_given_20k[2])),
     '{:.4f}'.format(controller.stats.auc_to_sh[1]),
     '{:.4f}'.format(controller.stats.final_to_sh[1])]

    load_results_to_file(save_results_path, config['machine_name'], hyperparameters_to_be_uploaded)

    session.close()

# ======================================================================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--process-index', type=int, default=10)
    parser.add_argument('--evaluation-seed', type=int, default=200)

    parser.add_argument('--game-height', type=int, default=10)
    parser.add_argument('--game-width', type=int, default=10)
    parser.add_argument('--t-max', type=int, default=25)  # the maximum time step of each episode
    parser.add_argument('--n-agents', type=int, default=3)
    parser.add_argument('--n-landmarks', type=int, default=3)
    parser.add_argument('--n-actions', type=int, default=5)
    parser.add_argument('--frame-stack-size', type=int, default=1)
    parser.add_argument('--observation-type', type=str, default='vector')

    parser.add_argument('--n-evaluation-levels', type=int, default=50)
    parser.add_argument('--evaluation-period', type=int, default=100)

    parser.add_argument('--model-save-period', type=int, default=10e3)

    # ==================================================================================================================

    parser.add_argument('--max-episodes', type=int, default=10e3)
    parser.add_argument('--replay-memory-init-size', type=int, default=10e3)
    parser.add_argument('--replay-memory-capacity', type=int, default=25e3)

    parser.add_argument('--target-update-period', type=int, default=10e3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train-period', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--summary-update-period', type=int, default=100)
    parser.add_argument('--adam-epsilon', type=float, default=1.5e-4)
    parser.add_argument('--huber-loss-delta', type=float, default=1.0)
    parser.add_argument('--hidden-layer-size', type=int, default=256)
    parser.add_argument('--hysteretic-multiplier', type=float, default=1.0)

    # ==================================================================================================================

    parser.add_argument('--machine-name', type=str, default='LAB-1')

    parser.add_argument('--seed', type=int, default=303)

    parser.add_argument('--threshold-ask', type=float, default=10.0)
    parser.add_argument('--threshold-give', type=float, default=3.0)
    parser.add_argument('--importance-threshold-give', type=float, default=1.0)

    parser.add_argument('--budget-ask', type=float, default=5000)
    parser.add_argument('--budget-give', type=float, default=5000)

    parser.add_argument('--advice-asking-mode', type=int, default=3)
    parser.add_argument('--advice-giving-mode', type=int, default=0)

    parser.add_argument('--use-teaching', action='store_true', default=False)

    # 0: Scratch, 1: 20-10-0, 2: 10-10-10 (Regional)
    parser.add_argument('--agents-knowledge-state', type=int, default=0)

    # ==================================================================================================================

    args = vars(parser.parse_args())
    main(args)

    # ==================================================================================================================
