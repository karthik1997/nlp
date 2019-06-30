from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import tensorflow as tf
import numpy as np

import random
import collections

from maxlength import max_length
from input_matrix import input_matrix, feed_matrix
from main import lstm1,lstm2

learningrate = 0.1
n_input_size=10
length,counter=max_length()
sentences=counter
batch_size=20
n_steps=length
epoch = 20

str1='English'
str2='eng'

matrixI = input_matrix(str1, sentences, n_steps, n_input_size)
matrixO = input_matrix(str2, sentences, n_steps, n_input_size)
#print(matrixI)

x = tf.placeholder("float", [n_steps, batch_size, n_input_size])
y = tf.placeholder("float", [n_steps, batch_size, n_input_size])

'''
weightsE = { 'encoder': tf.Variable(tf.random_normal([2*n_input_size, n_input_size])) }
biasesE = {	'encoder': tf.Variable(tf.random_normal([n_input_size])) }
weightsD = { 'decoder': tf.Variable(tf.random_normal([n_input_size, n_input_size])) }
biasesD = {	'decoder': tf.Variable(tf.random_normal([n_input_size])) }
#print("hi")'''
outputs_enc, out_state_fw, out_state_bw = lstm1(x, n_steps, n_input_size, batch_size)
#print(outputs_enc)

st=tf.unstack(outputs_enc,axis=1)

'''pred=[]
for o in st:
	pred.append(tf.matmul(o, weightsE['encoder']))'''

op = tf.stack(st, axis=1)
#final_op_enc = (op + biasesE['encoder'])

final_op_enc = tf.unstack(op, axis=0)
cell=rnn.LSTMCell(n_input_size)
initial_state = cell.zero_state(batch_size, tf.float32)

outputs_dec, states = lstm2(final_op_enc, n_input_size, batch_size, cell, initial_state)

st_d = tf.unstack(outputs_dec,axis=1)
'''pred_d=[]
for o in st_d:
	pred_d.append(tf.matmul(o, weightsD['decoder']))'''
	
op_d = tf.stack(st_d, axis=1)
#final_op_dec = (op_d + biasesD['decoder'])




cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = op_d, labels=y))
print('Cost finished')
optimizer = tf.train.RMSPropOptimizer(learning_rate=learningrate).minimize(cost)
print('Optimization finished')


correct_pred = tf.equal(tf.argmax(op_d,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
print('Accuracy finished')
##training
init = tf.global_variables_initializer()
with tf.Session() as sess:
	print("In Session")
	sess.run(init)
	#s = int(np.ceil(800/batch_size))
	e = 0
	while(e<epoch):
		c=0																	
		while(c<=2000):
			print("Batch ",c+1," Started")
			batch_x = feed_matrix(matrixI, c, batch_size, n_steps, n_input_size)
			batch_y = feed_matrix(matrixO, c, batch_size, n_steps, n_input_size)
			sess.run(optimizer, feed_dict={ x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			acc=sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
			
			print(" Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
			c+=1
		print("epoch",e+1,"completed")
		e+=1
		#print("hello")
		
	
	##Testing
	s_t=2001
	tm=0
	sum=0
	while tm<499: 
		batch_x = feed_matrix(matrixI, (s_t+tm), batch_size, n_steps, n_input_size)
		batch_y = feed_matrix(matrixO, (s_t+tm), batch_size, n_steps, n_input_size)
		
		acc=sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
		print(acc)
		#sum=sum+acc
		#print(sum/500)
		tm+=1
	#print("Final Accuracy:" + str(sum))

