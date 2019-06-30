from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

import random
import collections

from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

#from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear as linear

def lstm1(enc_inp, sent_len, input_size, batch_size, sequence_length=None):
	num_units=input_size
	fw_cell=rnn.LSTMCell(num_units)
	bw_cell=rnn.LSTMCell(num_units)
	enc_inp=tf.unstack(enc_inp, axis=0) 
	results, output_fw, output_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell, enc_inp, dtype=tf.float32)
	return results, output_fw, output_bw

def lstm2(decoder_inputs, input_size, batch_size, cell, initial_state, loop_function=None, scope=None):
	with variable_scope.variable_scope(scope or "rnn_decoder"):
		state = initial_state
		outputs = []
		prev = None
		for i, inp in enumerate(decoder_inputs):
		  if loop_function is not None and prev is not None:
		    with variable_scope.variable_scope("loop_function", reuse=True):
		      inp = loop_function(prev, i)
		  if i > 0:
		    variable_scope.get_variable_scope().reuse_variables()
		  output, state = cell(inp, state)
		  outputs.append(output)
		  if loop_function is not None:
		    prev = output
	return outputs, state
