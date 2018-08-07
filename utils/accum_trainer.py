# -*- coding: utf-8 -*-

import tensorflow as tf
import pdb

class AccumTrainer(object):
  def __init__(self,
               device="/cpu:0",
               name="AccumTrainer"):
    self._name = name
    self._device = device


  def _create_accum_grad(self, var):
    """
    Create Variable where to accumulate gradients.
    """
    zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
    name = var.name.replace(":", "_") + "_accum_grad"
    accum_grad = tf.Variable(zero, name=name, trainable=False)
    return accum_grad

  def prepare_minimize(self, loss, var_list):
    with tf.device(self._device):
      var_refs = [v._ref() for v in var_list]


      grads = tf.gradients( #getting the gradients of the above varaibles with respect to the var_list
        loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)


      self._var_list = var_list
      self._grad_list = grads
      self._accum_grad_list = [] #this list concoit of varaibles to put calculated gradients




      with tf.control_dependencies(None):
        for var in var_list:
          accum_grad = self._create_accum_grad(var) #Create Variable where to accumulate gradients. we put gradietns inside this
          self._accum_grad_list.append(accum_grad)


  def get_accum_grad_list(self):
    return self._accum_grad_list

  def accumulate_gradients(self, name=None): #This part basically is to 
    with tf.device(self._device):
      accumulate_ops = []

      with tf.name_scope(name, self._name, []) as name: #AccumTrainer this is the name 

        for var, grad, accum_grad in zip(self._var_list, self._grad_list, self._accum_grad_list): #assign the gradeints to the dedicated gradients variables
          with tf.name_scope("accum_" + var.op.name):
            #tf.assign_add =Update 'ref' by adding 'value' to it. This adds
            accumulate_ops.append( tf.assign_add(accum_grad, grad) )#This operation outputs "ref" after the update is done. This makes it easier to chain operations that need to use the reset value.
        return tf.group(*accumulate_ops, name=name) #Create an op that groups multiple operations. 

  def reset_gradients(self,  name=None):  #Resetting the gradinets of the 
    with tf.device(self._device):
      reset_ops = []
      with tf.name_scope(name, self._name, []) as name:
        for var, accum_grad in zip(self._var_list, self._accum_grad_list):
  
          with tf.name_scope("reset_" + var.op.name):
            zero = tf.zeros(accum_grad.get_shape())
            reset = accum_grad.assign(zero)
            reset_ops.append(reset)
        return tf.group(*reset_ops, name=name)
