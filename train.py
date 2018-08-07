#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import numpy as np
import os
import time

import signal
import random
import os

from network import ActorCriticFFNetwork
from discrim import Discriminator_WGAN
from training_thread import A3CTrainingThread

from utils.ops import log_uniform
from utils.rmsprop_applier import RMSPropApplier
from utils.rmsprop_applier_d import RMSPropApplier_d

from constants import ACTION_SIZE
from constants import PARALLEL_SIZE
from constants import INITIAL_ALPHA_LOW
from constants import INITIAL_ALPHA_HIGH
from constants import INITIAL_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import CHECKPOINT_FREQ
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import TASK_TYPE
from constants import TASK_LIST

import pdb

if __name__ == '__main__':

  device = "/gpu:0" if USE_GPU else "/cpu:0"
  device2="/cpu:0"
  network_scope = TASK_TYPE
  list_of_tasks = TASK_LIST
  scene_scopes = list_of_tasks.keys()
  global_t = 0
  stop_requested = False


  if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

  initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                      INITIAL_ALPHA_HIGH,
                                      INITIAL_ALPHA_LOG_RATE)

  global_network = ActorCriticFFNetwork(action_size = ACTION_SIZE, #get all the scene_scopes 
                                        device = device,
                                        network_scope = network_scope,
                                        scene_scopes = scene_scopes)


  
  global_discriminator = Discriminator_WGAN(action_size = ACTION_SIZE, #get all the scene_scopes 
                                        device = device,
                                        network_scope = network_scope,
                                        scene_scopes = scene_scopes)

  

  branches = []

 

  for scene in scene_scopes:
    for task in list_of_tasks[scene]:
      branches.append((scene, task))

  NUM_TASKS = len(branches)

  assert PARALLEL_SIZE >= NUM_TASKS, \
    "Not enough threads for multitasking: at least {} threads needed.".format(NUM_TASKS)

  learning_rate_input = tf.placeholder("float")
  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = RMSP_ALPHA,
                                momentum = 0.0,
                                epsilon = RMSP_EPSILON,
                                clip_norm = GRAD_NORM_CLIP,
                                device = device)
  grad_applier_discriminator = RMSPropApplier_d(learning_rate = learning_rate_input,
                                decay = RMSP_ALPHA,
                                momentum = 0.0,
                                epsilon = RMSP_EPSILON,
                                clip_norm = GRAD_NORM_CLIP,
                                device = device)

  # instantiate each training thread
  # each thread is training for one target in one scene
  training_threads = []
  
  for i in range(PARALLEL_SIZE):  #each local network has dedicated scene and a target 
    scene, task = branches[i%NUM_TASKS]

    training_thread = A3CTrainingThread(i, global_network,global_discriminator, initial_learning_rate, #each thread trained of seperate secene and task
                                        learning_rate_input,
                                        grad_applier,grad_applier_discriminator, MAX_TIME_STEP,
                                        device = device,
                                        network_scope = "thread-%d"%(i+1),
                                        scene_scope = scene,
                                        task_scope = task)
    training_threads.append(training_thread)

  




   
  # prepare session
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True))

  init = tf.global_variables_initializer()
  sess.run(init)


  # create tensorboard summaries
  summary_op = dict()
  summary_placeholders = dict()

  for i in range(PARALLEL_SIZE):
    scene, task = branches[i%NUM_TASKS]
    key = scene + "-" + task

    # summary for tensorboard
    episode_reward_input = tf.placeholder("float")
    episode_length_input = tf.placeholder("float")
    episode_max_q_input  = tf.placeholder("float")

    scalar_summaries = [
      tf.summary.scalar(key+"/Episode Reward", episode_reward_input),
      tf.summary.scalar(key+"/Episode Length", episode_length_input),
      tf.summary.scalar(key+"/Episode Max Q", episode_max_q_input)
    ]

    summary_op[key] = tf.summary.merge(scalar_summaries)
    summary_placeholders[key] = {
      "episode_reward_input": episode_reward_input,
      "episode_length_input": episode_length_input,
      "episode_max_q_input": episode_max_q_input,
      "learning_rate_input": learning_rate_input
    }

  log_file = os.path.join(LOG_FILE, time.strftime("%Y-%m-%d-%H%M%S"))
  summary_writer = tf.summary.FileWriter(log_file, sess.graph)
  print("Prinitng the Number Of Tasks",NUM_TASKS)


  # init or load checkpoint with saver
  # if you don't need to be able to resume training, use the next line instead.
  # it will result in a much smaller checkpoint file.
  saver = tf.train.Saver(max_to_keep=10, var_list=global_network.get_vars())
  #saver = tf.train.Saver(max_to_keep=10) #Make sure this has the variable list expet old params
  '''
  
  var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  for v in var_list:
    print(v)
  '''


  

  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)


  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[1])
    print(">>> global step set: {}".format(global_t))
  else:
    print("Could not find old checkpoint")




  def train_function(parallel_index):
    global global_t
   

    training_thread = training_threads[parallel_index] #get the pre assigned trainign thread
    last_global_t = 0
    
    scene, task = branches[parallel_index % NUM_TASKS]
    key = scene + "-" + task


    while global_t <= MAX_TIME_STEP and not stop_requested: #max time step is 10000000.0

      
      diff_global_t = training_thread.process(sess, global_t, summary_writer, #each threat with dedicated scene and task
                                              summary_op[key], summary_placeholders[key])
      global_t += diff_global_t
      # periodically save checkpoints to disk
      if parallel_index == 0 and (global_t - last_global_t >= CHECKPOINT_FREQ or global_t == MAX_TIME_STEP):
        print('Save checkpoint at timestamp %d' % global_t)
        saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
        last_global_t = CHECKPOINT_FREQ * (global_t // CHECKPOINT_FREQ)


  def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True

  train_threads = []
  for i in range(PARALLEL_SIZE):
    train_threads.append(threading.Thread(target=train_function, args=(i,))) #starting the ruuning of the threads
 


  signal.signal(signal.SIGINT, signal_handler)
  

  # start each training thread
  for t in train_threads:
    t.start()
    



  print('Press Ctrl+C to stop.')
  signal.pause()

  # wait for all threads to finish
  for t in train_threads:
    t.join()
 

  

  print('Now saving data. Please wait.')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
  summary_writer.close()

