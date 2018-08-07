#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import sys
import argparse

from network import ActorCriticFFNetwork#, ActorCriticLSTMNetwork
from training_thread import A3CTrainingThread
from scene_loader import THORDiscreteEnvironment as Environment

from utils.ops import sample_action

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE
from constants import USE_LSTM
from constants import EVAL_INIT_LOC

from constants import TASK_TYPE
from constants import TEST_TASK_LIST

import pdb

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--time', help='Name of checkpoint file.',
                      default=None)
  args = parser.parse_args()

  device = "/cpu:0" # use CPU for display tool
  network_scope = TASK_TYPE
  list_of_tasks = TEST_TASK_LIST
  scene_scopes = list_of_tasks.keys()

  if USE_LSTM:
    global_network = ActorCriticLSTMNetwork(action_size=ACTION_SIZE,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)
  else:
    global_network = ActorCriticFFNetwork(action_size=ACTION_SIZE,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)




  




  if checkpoint and checkpoint.model_checkpoint_path:
    if args.time:
      checkpoint_path = "checkpoints/checkpoint-" + args.time
    else:
      checkpoint_path = checkpoint.model_checkpoint_path
    saver.restore(sess, checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint_path))
  else:
    print("Could not find old checkpoint")


  print(checkpoint_path)


  pdb.set_trace()

  scene_stats = dict()
  for scene_scope in scene_scopes:

    scene_stats[scene_scope] = []
    for task_scope in list_of_tasks[scene_scope]:

      env = Environment({
        'scene_name': scene_scope,
        'terminal_state_id': int(task_scope),
        'initial_state': EVAL_INIT_LOC,
      })
      ep_rewards = []
      ep_lengths = []
      ep_collisions = []

      scopes = [network_scope, scene_scope, task_scope]

      for i_episode in range(NUM_EVAL_EPISODES):

        env.reset()
        terminal = False
        ep_reward = 0
        ep_collision = 0
        ep_t = 0
        ep_action = []

        while not terminal:

          pi_values = global_network.run_policy(sess, env.s_t, env.target, scopes)
          action = sample_action(pi_values)
          ep_action.append(action)
          env.step(action)
          env.update()

          terminal = env.terminal
          if ep_t == 100: break
          if env.collided: ep_collision += 1
          ep_reward += env.reward
          ep_t += 1

        if USE_LSTM:
          global_network.reset_state()
        ep_lengths.append(ep_t)
        ep_rewards.append(ep_reward)
        ep_collisions.append(ep_collision)
        if VERBOSE:
          print("episode #{} ends after {} steps".format(i_episode, ep_t))
          print(ep_action)

      print('evaluation: %s %s' % (scene_scope, task_scope))
      print('mean episode reward: %.2f' % np.mean(ep_rewards))
      print('mean episode length: %.2f' % np.mean(ep_lengths))
      print('mean episode collision: %.2f' % np.mean(ep_collisions))

      scene_stats[scene_scope].extend(ep_lengths)

print('\nResults (average trajectory length):')
for scene_scope in scene_stats:
  print('%s: %.2f steps'%(scene_scope, np.mean(scene_stats[scene_scope])))
