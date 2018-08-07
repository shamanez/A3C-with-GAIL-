# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from utils.accum_trainer import AccumTrainer
from scene_loader import THORDiscreteEnvironment as Environment
from network import ActorCriticFFNetwork
from discrim import Discriminator_WGAN

from dagger_policy_generators import ShortestPathOracle


from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import VERBOSE
from constants import USE_LSTM
import pdb

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               global_discriminator,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               grad_applier_discriminator,
               max_global_time_step,
               device,
               network_scope="network",
               scene_scope="scene",
               task_scope="task"):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.network_scope = network_scope
    self.network_scope_D = network_scope+"_d"
    self.scene_scope = scene_scope
    self.task_scope = task_scope
    self.scopes = [network_scope, scene_scope, task_scope]

    self.scopes_d=[self.network_scope_D, scene_scope, task_scope]

    self.local_network = ActorCriticFFNetwork(
                           action_size=ACTION_SIZE,
                           device=device,
                           network_scope=network_scope,
                           scene_scopes=[scene_scope])
    
    self.local_network.prepare_loss(ENTROPY_BETA, self.scopes)


    self.trainer = AccumTrainer(device)

    self.trainer.prepare_minimize(self.local_network.total_loss,  #getting the gradients of for the local network variablkes
                                  self.local_network.get_vars())

    #This part is for the newly added PPO loss (we need to keep old and new update parameters)
    new_variable_list=self.local_network.get_vars()
    old_varaible_list=self.local_network.get_vars_old()

    #For the ppo loss begining of the each iteration we need to sync old with current
    self.old_new_sync=self.local_network.sync_curre_old()

    self.accum_gradients = self.trainer.accumulate_gradients() #This is to assign gradients 
    self.reset_gradients = self.trainer.reset_gradients() #after applying the grads to variables we need to resent those variables
     
    accum_grad_names = [self._local_var_name(x) for x in self.trainer.get_accum_grad_list()] #get the name list of all the grad vars
  
    global_net_vars = [x for x in global_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names] #check whether the global_network vars are mentioned in gradiet computations for them
    local_net_vars = [x for x in self.local_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names]

    self.apply_gradients = grad_applier.apply_gradients(global_net_vars, self.trainer.get_accum_grad_list())
    self.apply_gradients_local = grad_applier.apply_gradients_local_net(local_net_vars, self.trainer.get_accum_grad_list())
      
    #If this is unstable it is desireable to first apply the gradients on the local network and then clip and after that we apply
    self.sync = self.local_network.sync_from(global_network) #this is to sync from the glocal network Apply updated global params to the local network

   


#This part is for the Discriminator
#########################################################################################
                                                                                     # 
    self.local_discriminator = Discriminator_WGAN(                                      #
                           action_size=ACTION_SIZE,                                     # 
                           device="/cpu:0",                                             #  
                           network_scope=network_scope,                                 #
                           scene_scopes=[scene_scope])                                  #
                                                                                        #
    self.local_discriminator.prepare_loss_D(ENTROPY_BETA, self.scopes_d)                #           
                                                                                        #
    self.trainer_D = AccumTrainer(device="/cpu:0",name="AccumTrainer_d")                #
                                                                                        #
    self.trainer_D.prepare_minimize(self.local_discriminator.total_loss_d,              #
                                  self.local_discriminator.get_vars())                  #
                                                                                        #
                                                                                        #
    self.accum_gradients_d = self.trainer_D.accumulate_gradients()                      #
    self.reset_gradients_d = self.trainer_D.reset_gradients()                           #
                                                                                        #
    accum_grad_names_discrimi = [self._local_var_name(x) for x in self.trainer_D.get_accum_grad_list()]
                                                                                        #
                                                                                        #
    global_discri_vars = [x for x in global_discriminator.get_vars() if self._get_accum_grad_name(x) in accum_grad_names_discrimi]
                                                                                        #
    self.apply_gradients_discriminator=grad_applier_discriminator.apply_gradients(global_discri_vars, self.trainer_D.get_accum_grad_list())
                                                                                        #    
    self.clip_global_d_weights = global_discriminator.clip_weights() #here we are clipping the global net weights directly. 
                                                                                        #
    self.sync_discriminator =self.local_discriminator.sync_from(global_discriminator)  #
                                                                                        #                                                                                #
    self.D_var=global_discriminator.get_vars()                                                                                    #
                                                                                        #
#########################################################################################                                                                                       





    self.env = None
    self.local_t = 0
    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0
    self.episode_length = 0
    self.episode_max_q = -np.inf

  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_accum_grad_name(self, var):
    return self._local_var_name(var).replace(':','_') + '_accum_grad:0'

  def _anneal_learning_rate(self, global_time_step):
    time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
    learning_rate = self.initial_learning_rate * time_step_to_go / self.max_global_time_step
    return learning_rate

  def choose_action(self, pi_values):
    values = []
    sum = 0.0
    for rate in pi_values:
      sum = sum + rate
      value = sum
      values.append(value)

    r = random.random() * sum
    for i in range(len(values)):
      if values[i] >= r:
        return i

    # fail safe
    return len(values) - 1

  def _record_score(self, sess, writer, summary_op, placeholders, values, global_t):
    feed_dict = {}
    for k in placeholders:
      feed_dict[placeholders[k]] = values[k]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    if VERBOSE: print('writing to summary writer at time %d\n' % (global_t))
    writer.add_summary(summary_str, global_t)
    # writer.flush()

  def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):

    if self.env is None:
      # lazy evaluation
      time.sleep(self.thread_index*1.0)
      self.env = Environment({
        'scene_name': self.scene_scope,
        'terminal_state_id': int(self.task_scope)
      })

      self.env.reset() #resetting the environment for each thread



    self.env_Oracle = Environment({ #Every iteration in the thread the expert start with the current state of the agent
      'scene_name': self.scene_scope,
      'terminal_state_id': int(self.task_scope),
      'initial_state':self.env.current_state_id
    })

    self.env_Oracle.reset()


    states = [] #to keeep state ,actions ,targets and other stae
    actions = []
    rewards = []
    values = []
    targets = []
    dones=[]


    states_oracle = []
    actions_oracle = [] 
    targets_oracle=[] 
    

    terminal_end = False #in the start terminal state_end is false

    # reset accumulated gradients
    sess.run(self.reset_gradients) #resetting the gradient positions when starting the process for each Iteration

    # copy weights from shared to local
    sess.run(self.sync)

    start_local_t = self.local_t
    self.oracle = ShortestPathOracle(self.env_Oracle, ACTION_SIZE)

    #########################################################################################
    #Sampling the Expert Trajectories 
    for i in range(100):
      #We might need to use an for loop to finish the expert trajectory first     
      oracle_pi = self.oracle.run_policy(self.env_Oracle.current_state_id) #get the policy of the oracle which means the shotest path kind of action in the given state
      oracle_action = self.choose_action(oracle_pi) 

      states_oracle.append(self.env_Oracle.s_t) 
      actions_oracle.append(oracle_action)
      targets_oracle.append(self.env_Oracle.target)

      self.env_Oracle.step(oracle_action)

      terminal_o = self.env_Oracle.terminal

      self.env_Oracle.update()

      if terminal_o:
        break
    ##############################################################################################

    # t_max times loop
    for i in range(LOCAL_T_MAX): #one thread will run for maximum amoound to 5 iterations then do a gradiet uodate

      pi_, value_ = self.local_network.run_policy_and_value(sess, self.env.s_t, self.env.target, self.scopes)
    
      
      action = self.choose_action(pi_)

      states.append(self.env.s_t) 
      actions.append(action)
      values.append(value_)
      targets.append(self.env.target)


      if VERBOSE and (self.thread_index == 0) and (self.local_t % 1000) == 0:
        sys.stdout.write("Pi = {0} V = {1}\n".format(pi_, value_))

      # process game
      self.env.step(action)
      

      # receive game result
      reward = self.env.reward  #getting the reward from the env
      terminal = self.env.terminal #geting whether the agent went to a terminal state

     
      # ad-hoc reward for navigation
      reward = 10.0 if terminal else -0.01 #this is the normal reward here 10 if terminal all the others it is -0.01 (ollision donesst take in to the accout)
      if self.episode_length > 5e3: terminal = True #Here we do not let agent to run more that 5000 steps so we make it terminal
      #but the above terminal thing has no effect on giving 10 as the rwaerd because we set the rweard above

      self.episode_reward += reward
      self.episode_length += 1
      #this is what is the maximum value got in the episode
      self.episode_max_q = max(self.episode_max_q, np.max(value_)) #self.episode_max_q-This is -inf in the beggining 

      # clip reward
      rewards.append(np.clip(reward, -1, 1)) #make sure the rewartds is between -1 and +1 even thore rtthere is a 10
      
      self.local_t += 1

      # s_t1 -> s_t
      self.env.update()
      
      if terminal: #if we go to the terminal state we will surely break the function
        sys.stdout.write("Pi = {0} V = {1}\n".format(pi_, value_))
        terminal_end = True
        sys.stdout.write("time %d | thread #%d | scene %s | target #%s\n%s %s episode reward = %.3f\n%s %s episode length = %d\n%s %s episode max Q  = %.3f\n" % (global_t, self.thread_index, self.scene_scope, self.task_scope, self.scene_scope, self.task_scope, self.episode_reward, self.scene_scope, self.task_scope, self.episode_length, self.scene_scope, self.task_scope, self.episode_max_q))

        summary_values = {
          "episode_reward_input": self.episode_reward,
          "episode_length_input": float(self.episode_length),
          "episode_max_q_input": self.episode_max_q,
          "learning_rate_input": self._anneal_learning_rate(global_t)
        }

        self._record_score(sess, summary_writer, summary_op, summary_placeholders,
                           summary_values, global_t)
        self.episode_reward = 0 #after terminal state we gonna make all these variables to zero
        self.episode_length = 0 #Now the AI need to start from new position
        self.episode_max_q = -np.inf #after a terminaltion we do this
        self.env.reset()

        break


      
    R = 0.0 #In the terminal Return is nothing  #If it's terminal end we do not have a return from the final state

    if not terminal_end: #But if it's not the turminal Return is the next value function
      R = self.local_network.run_value(sess, self.env.s_t, self.env.target, self.scopes)

    #Agent's Samples  
    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    #Expert's Samples 
    states_oracle.reverse() 
    actions_oracle.reverse()
    actions_oracle.reverse()

    #Agent's batch
    batch_si = []
    batch_a = []
    batch_actions=[]
    batch_td = []
    batch_R = []
    batch_t = []

    #Expert's Batch
    batch_si_ex=[]
    batch_a_ex=[]
    batch_t_ex=[]

    batch_si_d = []
    batch_t_d = []
    batch_actions_d=[]



    #This is for the   
    for(s_e,a_e,t_e) in zip(states_oracle,actions_oracle,targets_oracle):
      batch_si_ex.append(s_e)
      batch_a_ex.append(a_e)
      batch_t_ex.append(t_e)

    for(ai, si, ti) in zip(actions, states, targets):

      batch_actions_d.append(ai)
      batch_si_d.append(si)
      batch_t_d.append(ti)


    cur_learning_rate = self._anneal_learning_rate(global_t)

 



    
    for i in range(5):

      sess.run(self.reset_gradients_d)
   
      sess.run(self.accum_gradients_d, #since we update the algorithm for given action ,given state, given advatns and given value and given reward we do not care about the sequence
              feed_dict = {
              self.local_discriminator.s_e: batch_si_ex,
              self.local_discriminator.Actions_e: batch_a_ex,
              self.local_discriminator.s_a: batch_si_d,
              self.local_discriminator.Actions_a: batch_actions_d,
              self.local_discriminator.t_e: batch_t_ex,
              self.local_discriminator.t_a: batch_t_d })

      
      sess.run(self.apply_gradients_discriminator, #directly gradients get apply on the global discri
              feed_dict = { self.learning_rate_input: cur_learning_rate } )

      sess.run(self.clip_global_d_weights) #every update make sure u clip weihtfs
    
      sess.run(self.sync_discriminator) #from the Global Discriminator we sync to the local

      


    critic_r = self.local_discriminator.run_critic(sess, batch_si_d, batch_t_d,batch_actions_d, self.scopes_d)   
   
    critic_r=critic_r*0.1
 
    rewards=rewards+critic_r  #We concatenate the rewrds function 

    # Compute the advantage function , return and stack them as batches in Agent
    for(ai, ri, si, Vi, ti) in zip(actions, rewards, states, values, targets):
      R = ri + GAMMA * R  #calculatung the adcantage function
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1 #making the actions one hot
      batch_actions.append(ai)
      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)
      batch_t.append(ti)


    #syncying the new paramters to the old network in the thread PPO
    sess.run(self.old_new_sync)
    for i in range(4):
      #sess.run(self.reset_gradients) #reset the gradients    
      sess.run( self.accum_gradients, #since we update the algorithm for given action ,given state, given advatns and given value and given reward we do not care about the sequence
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.t: batch_t,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,})
    
      sess.run(self.apply_gradients_local, #apply the gradients to the local networ
                feed_dict = { self.learning_rate_input: cur_learning_rate } )




      

    
    #theoritcally we can have one accume gradient operation here  
    sess.run(self.apply_gradients,
              feed_dict = { self.learning_rate_input: cur_learning_rate } )

   

    if VERBOSE and (self.thread_index == 0) and (self.local_t % 100) == 0:
      sys.stdout.write("Local timestep %d\n" % self.local_t)



    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

