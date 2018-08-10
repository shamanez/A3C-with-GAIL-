# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops

class Discriminator(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size
    

  def prepare_loss_D(self, entropy_beta, scopes):


    scope_key = self._get_key(scopes[:-1])

    with tf.device(self._device):      
      score_e = self.sc_e[scope_key]
      score_a = self.sc_a[scope_key]

      self.S_A=score_a 

      self.total_loss_d= -tf.reduce_mean(score_e)+tf.reduce_mean(score_a)

    

  def get_vars(self):
    raise NotImplementedError()



  def clip_weights(self):
    dst_vars = self.get_vars() #Slave - current thread
    clip_ops = []  #This is sync_ops


    with tf.device(self._device):
      with tf.name_scope("clipWeights", []) as name:
        for src_var in dst_vars: 
          clip =src_var.assign(tf.clip_by_value(src_var, -0.01, 0.01))
          clip_ops.append(clip)

        return tf.group(*clip_ops, name=name)



  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc_d'):

    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc_d'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv_d'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv_d'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def _get_key(self, scopes):
    return '/'.join(scopes)


  def sync_to(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars() #Master net - This is the common networ
    dst_vars = self.get_vars() #Slave - current thread
    
    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]


    sync_ops = []  #This is sync_ops

    with tf.device(self._device):
      with tf.name_scope(name, "Discriminator_l_g", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars): 
          sync_op = tf.assign(src_var,dst_var )  #apply the updated slave(local) network variables to the Master(global) network variables
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)


  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars() #Master net - This is the common networ
    dst_vars = self.get_vars() #Slave - current thread
    
    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]


    sync_ops_G = []  #This is sync_ops


    with tf.device(self._device):
      with tf.name_scope(name, "Discriminator_g_l", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars): 
          sync_op = tf.assign(dst_var,src_var )  #apply the updated slave(local) network variables to the Master(global) network variables
          sync_ops_G.append(sync_op)
        return tf.group(*sync_ops_G, name=name)


class Discriminator_WGAN(Discriminator):
 
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    network_scope=network_scope+"_d"
    Discriminator.__init__(self, action_size, device)

    self.sc_e = dict()
    self.sc_a = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_ac_emb=dict()

    #self.W_fc3 = dict()
    #self.b_fc3 = dict()


    self.W_sc = dict()
    self.b_sc = dict()
    

    with tf.device(self._device):

      # state expert (input)
      self.s_e = tf.placeholder("float", [None, 2048, 4])
      self.Actions_e = tf.placeholder("int32", [None])
      

      # agent expert (input)
      self.s_a = tf.placeholder("float", [None, 2048, 4])
      self.Actions_a = tf.placeholder("int32", [None])


      # target expert (input)
      self.t_e = tf.placeholder("float", [None, 2048, 4])

      # target agent (input)
      self.t_a = tf.placeholder("float", [None, 2048, 4])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope
       

        # flatten input
        self.s_e_flat = tf.reshape(self.s_e, [-1, 8192])
        self.t_e_flat = tf.reshape(self.t_e, [-1, 8192])

        self.s_a_flat = tf.reshape(self.s_a, [-1, 8192])
        self.t_a_flat = tf.reshape(self.t_a, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 128]) #256
        self.b_fc1[key] = self._fc_bias_variable([128], 8192) #256

        self.W_ac_emb[key]=self._fc_weight_variable([4, 128],name='Embed')

       

        ###### Action embedding Shoud concatenate this embeddings with the 
        E_ac_embed = embedding_ops.embedding_lookup(self.W_ac_emb[key], self.Actions_e)
        A_ac_embed = embedding_ops.embedding_lookup(self.W_ac_emb[key], self.Actions_a)


    
        #For the Expert
        h_s_e_flat = tf.nn.relu(tf.matmul(self.s_e_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_e_flat = tf.nn.relu(tf.matmul(self.t_e_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1_e = tf.concat(values=[h_s_e_flat, h_t_e_flat], axis=1)

        #For the Agent
        h_s_a_flat = tf.nn.relu(tf.matmul(self.s_a_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_a_flat = tf.nn.relu(tf.matmul(self.t_a_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1_a = tf.concat(values=[h_s_a_flat, h_t_a_flat], axis=1)



        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([256, 128]) #256
        self.b_fc2[key] = self._fc_bias_variable([128], 256) #256



        #For the expert
        h_fc2_e = tf.nn.relu(tf.matmul(h_fc1_e, self.W_fc2[key]) + self.b_fc2[key])

        #For the agent
        h_fc2_a = tf.nn.relu(tf.matmul(h_fc1_a, self.W_fc2[key]) + self.b_fc2[key])


        h_fc2_e=tf.concat(values=[h_fc2_e,E_ac_embed], axis=1)
        h_fc2_a=tf.concat(values=[h_fc2_a,A_ac_embed], axis=1)

        self.W_sc[key] = self._fc_weight_variable([256, 1])
        self.b_sc[key] = self._fc_bias_variable([1], 256)

        sc_e = tf.matmul(h_fc2_e, self.W_sc[key]) + self.b_sc[key]
        sc_a = tf.matmul(h_fc2_a, self.W_sc[key]) + self.b_sc[key]



        self.sc_e[key] = tf.reshape(sc_e, [-1])
        self.sc_a[key] = tf.reshape(sc_a, [-1])


        '''
        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])



          with tf.variable_scope(scene_scope):
            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)


            h_fc3_e = tf.nn.relu(tf.matmul(h_fc2_e, self.W_fc3[key]) + self.b_fc3[key])
            h_fc3_a = tf.nn.relu(tf.matmul(h_fc2_a, self.W_fc3[key]) + self.b_fc3[key])



            # weight for score output layer
            self.W_sc[key] = self._fc_weight_variable([512, 1])
            self.b_sc[key] = self._fc_bias_variable([1], 512)

            # value (output)
            sc_e = tf.matmul(h_fc3_e, self.W_sc[key]) + self.b_sc[key]
            sc_a = tf.matmul(h_fc3_a, self.W_sc[key]) + self.b_sc[key]



            self.sc_e[key] = tf.reshape(sc_e, [-1])
            self.sc_a[key] = tf.reshape(sc_a, [-1])
          '''



  def run_critic(self, sess, state, target,action, scopes):
    k = self._get_key(scopes[:-1])
    R_out = sess.run( self.sc_a[k], feed_dict = {self.s_a : state, self.t_a: target,self.Actions_a: action} )
    return R_out

  def run_critic_expert(self, sess, state, target,action, scopes):
    k = self._get_key(scopes[:-1])
    R_out = sess.run( self.sc_e[k] , feed_dict = {self.s_e : state, self.t_e: target,self.Actions_e: action} )
    return R_out


  def get_vars(self):

    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_ac_emb,
      self.W_sc, self.b_sc
      
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs


