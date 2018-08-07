# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import embedding_ops
import pdb

# Actor-Critic Network Base Class
# The policy network and value network architecture
# should be implemented in a child class of this one
class ActorCriticNetwork(object):
  def __init__(self,
               action_size,
               device="/cpu:0"):
    self._device = device
    self._action_size = action_size

  def prepare_loss(self, entropy_beta, scopes):

    # drop task id (last element) as all tasks in
    # the same scene share the same output branch
    scope_key = self._get_key(scopes[:-1])
    scope_key_old=self._get_key_old(scopes[:-1])


    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])


      self.r = tf.placeholder("float", [None])


      ########################################## -Returns 
      self.Returns=tf.placeholder("float", [None])
      self.Advantages=tf.placeholder("float", [None])
      ###################################################




      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0))
      log_pi_old = tf.log(tf.clip_by_value(self.pi_old[scope_key_old], 1e-20, 1.0))

      #########################################################################
      value_pi=self.v[scope_key]
      old_value_pi=self.v_old[scope_key_old]
      vpredclipped = old_value_pi + tf.clip_by_value(value_pi - old_value_pi, - 0.2, 0.2)
      vf_losses1 = tf.square(self.v[scope_key] - self.r) 
      vf_losses2 = tf.square(vpredclipped - self.r)
      vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) #This is the vf loss
      ######################################################################

      cur_pi=tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1)
      old_pi=tf.reduce_sum(tf.multiply(log_pi_old, self.a), axis=1)



      ratio = tf.exp(cur_pi - old_pi)
      pg_losses = self.td * ratio #sourragete loss unclipeed
      pg_losses2 = self.td * tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2) #clipped
      pg_loss = tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))
      #approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC)) #this is some kind of an approximation

      #clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))


      # policy entropy
      entropy = -tf.reduce_sum(self.pi[scope_key] * log_pi, axis=1)

      # policy loss (output)
      #policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * entropy_beta)
      

      policy_loss = -tf.reduce_sum(pg_loss+ entropy * entropy_beta)
      # R (input for value)
      

      # value loss (output)
      # learning rate for critic is half of actor's

      #value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key]) #this is shared paramters so we can use them in both
      #value_loss = 0.5 * tf.nn.l2_loss(self.Returns - self.v[scope_key])

      # gradienet of policy and value are summed up
      #self.total_loss = policy_loss + value_loss
      self.total_loss = policy_loss + vf_loss

  def run_policy_and_value(self, sess, s_t, task):
    raise NotImplementedError()

  def run_policy_and_value_old(self, sess, s_t, task):
    raise NotImplementedError()

  def run_policy(self, sess, s_t, task):
    raise NotImplementedError()

  def run_value(self, sess, s_t, task):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def  sync_curre_old(self):
    raise NotImplementedError()


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




    sync_ops = []  #This is sync_ops

    with tf.device(self._device):
      with tf.name_scope(name, "ActorCriticNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars): 
          sync_op = tf.assign(dst_var, src_var)  #apply the updated Master network variables to the slave network variables
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)



  def sync_curre_old(self):

    current_parms = self.get_vars() #Slave - current thread
    Old_net_params= self.get_vars_old() #get the PPO loss have to keep the an old network



    curr_var_names = [self._local_var_name(x) for x in current_parms]    
    old_var_names = [self._local_var_name(x) for x in Old_net_params]


    # keep only variables from both src and dst
    curr_var = [x for x in current_parms
      if self._local_var_name(x) in old_var_names]
    old_var = [x for x in Old_net_params
      if self._local_var_name(x) in curr_var_names]



    sync_ops_old_n = []  #This is sync_ops

    with tf.device(self._device):
      with tf.name_scope("PPO_old", []) as name:
        for(src_var, dst_var) in zip(curr_var, old_var): 
          sync_op = tf.assign(dst_var, src_var)  #apply the updated Master network variables to the slave network variables
          sync_ops_old_n.append(sync_op)

        return tf.group(*sync_ops_old_n, name=name)

  # variable (global/scene/task1/W_fc:0) --> scene/task1/W_fc:0
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  # weight initialization based on muupan's code
  # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def _get_key(self, scopes):
    return '/'.join(scopes)

  def _get_key_old(self, scopes):
    return ("old_"+'/'.join(scopes))

# Actor-Critic Feed-Forward Network
class ActorCriticFFNetwork(ActorCriticNetwork):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    ActorCriticNetwork.__init__(self, action_size, device)

    self.pi = dict()
    self.v = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_value = dict()
    self.b_value = dict()


    self.pi_old = dict()
    self.v_old = dict()

    self.W_fc1_old = dict()
    self.b_fc1_old = dict()

    self.W_fc2_old = dict()
    self.b_fc2_old = dict()

    self.W_fc3_old = dict()
    self.b_fc3_old = dict()

    self.W_policy_old = dict()
    self.b_policy_old = dict()

    self.W_value_old = dict()
    self.b_value_old = dict()



    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4])

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 4])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)



        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])


          with tf.variable_scope(scene_scope):
            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])


            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)



            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(h_fc3, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])

      network_scope_old=  "old_"+network_scope    
      with tf.variable_scope(network_scope_old):
        # network key
        key = network_scope_old

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])


        # shared siamese layer  
        self.W_fc1_old[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1_old[key] = self._fc_bias_variable([512], 8192)



        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1_old[key]) + self.b_fc1_old[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1_old[key]) + self.b_fc1_old[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)

        # shared fusion layer
        self.W_fc2_old[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2_old[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2_old[key]) + self.b_fc2_old[key])



        for scene_scope in scene_scopes:
          # scene-specific key
      
          key = self._get_key([network_scope_old, scene_scope])
          

          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3_old[key] = self._fc_weight_variable([512, 512])
            self.b_fc3_old[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3_old[key]) + self.b_fc3_old[key])


            # weight for policy output layer
            self.W_policy_old[key] = self._fc_weight_variable([512, action_size])
            self.b_policy_old[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy_old[key]) + self.b_policy_old[key]
            self.pi_old[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value_old[key] = self._fc_weight_variable([512, 1])
            self.b_value_old[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(h_fc3, self.W_value_old[key]) + self.b_value_old[key]
            self.v_old[key] = tf.reshape(v_, [-1])



        

  def run_policy_and_value(self, sess, state, target, scopes):  #to run the Initaila policy
    k = self._get_key(scopes[:2])
    pi_out, v_out = sess.run( [self.pi[k], self.v[k]], feed_dict = {self.s : [state], self.t: [target]} )
    return (pi_out[0], v_out[0])

  def run_policy_and_value_old(self, sess, state, target, scopes): #to keep the track of old policy only in the generator training process
    k = self._get_key_old(scopes[:2])
    pi_out, v_out = sess.run( [self.pi_old[k], self.v_old[k]], feed_dict = {self.s : [state], self.t: [target]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[k], feed_dict = {self.s : [state], self.t: [target]} )
    return pi_out[0]

  def run_value(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    v_out = sess.run( self.v[k], feed_dict = {self.s : [state], self.t: [target]} )
    return v_out[0]

  def get_vars(self):

    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs

  def get_vars_old(self):
    var_list = [
      self.W_fc1_old, self.b_fc1_old,
      self.W_fc2_old, self.b_fc2_old,
      self.W_fc3_old, self.b_fc3_old,
      self.W_policy_old, self.b_policy_old,
      self.W_value_old, self.b_value_old
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
  

# Actor-Critic LSTM Network
'''
class ActorCriticLSTMNetwork(ActorCriticNetwork):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  def __init__(self,
               action_size,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    ActorCriticNetwork.__init__(self, action_size, device)

    self.pi = dict()
    self.v = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_value = dict()
    self.b_value = dict()

    self.lstm = dict()
    self.lstm_state = dict()

    self.W_lstm = dict()
    self.b_lstm = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4], name="s_placeholder")

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 4], name="t_placeholder")

      # place holder for LSTM unrolling time step size.
      # self.step_size = tf.placeholder(tf.float32, [1], name="step_size_placeholder")

      self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 512], name="initial_lstm_state0_placeholder")
      self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 512], name="initial_lstm_state1_placeholder")
      self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state0,
                                                              self.initial_lstm_state1)

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)

        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope) as scope:

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])

            h_fc3_reshaped = tf.reshape(h_fc3, [1, -1, 512])
            print(h_fc3_reshaped.shape,"Shamamaneee")

            # lstm
            self.lstm[key] = tf.contrib.rnn.BasicLSTMCell(512)

            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state[key] = tf.nn.dynamic_rnn(self.lstm[key],
                                                                   h_fc3_reshaped,
                                                                   initial_state=self.initial_lstm_state,
                                                                  #  sequence_length=self.step_size,
                                                                   time_major=False,
                                                                   scope=scope)

            # lstm_outputs: (1,5,512) for back prop, (1,1,512) for forward prop.
            lstm_outputs = tf.reshape(lstm_outputs, [-1, 512])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(lstm_outputs, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(h_fc3, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])

            scope.reuse_variables()
            self.W_lstm[key] = tf.get_variable("basic_lstm_cell/weights")
            self.b_lstm[key] = tf.get_variable("basic_lstm_cell/biases")
            self.reset_state()


  def reset_state(self):
    self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 512]),
                                                        np.zeros([1, 512]))

  def run_policy_and_value(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    print(k,"zzzzz")
    pi_out, v_out, self.lstm_state_out = sess.run([self.pi[k], self.v[k], self.lstm_state[k]],
                                                  feed_dict={self.s: [state],
                                                             self.t: [target],
                                                             self.initial_lstm_state0: self.lstm_state_out[0],
                                                             self.initial_lstm_state1: self.lstm_state_out[1],})
                                                            #  self.step_size: [1]
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    pi_out, self.lstm_state_out = sess.run([self.pi[k], self.lstm_state[k]],
                                           feed_dict={self.s: [state],
                                                      self.t: [target],
                                                      self.initial_lstm_state0: self.lstm_state_out[0],
                                                      self.initial_lstm_state1: self.lstm_state_out[1],})
                                                      # self.step_size: [1]
    return pi_out[0]

  def run_value(self, sess, state, target, scopes):
    # This run_value() is used for calculating V for bootstrapping at the
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    prev_lstm_state_out = self.lstm_state_out
    k = self._get_key(scopes[:2])
    v_out, _ = sess.run([self.v[k], self.lstm_state[k]],
                        feed_dict={self.s: [state],
                                   self.t: [target],
                                   self.initial_lstm_state0: self.lstm_state_out[0],
                                   self.initial_lstm_state1: self.lstm_state_out[1],})
                                  #  self.step_size: [1]
    self.lstm_state_out = prev_lstm_state_out
    return v_out[0]

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_lstm, self.b_lstm,
      self.W_policy, self.b_policy,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
'''
