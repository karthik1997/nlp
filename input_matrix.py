import numpy as np
import gensim, logging, os

from Word2Vec import w2vector

def input_matrix(string, tot_sent, sent_len, input_size):
	mat=np.zeros((tot_sent, sent_len, input_size))
	def split_line(dict_name):
		with open(string) as f:
			i=-1
			for line in f:
				i+=1
				l=line.split(' ')
				j=-1
				for w in l:
					w=w.rstrip('\n')
					w=w.strip()
					j+=1
					if w in dict_name:
						vector=dict_name[w]
						mat[i][j]=vector    
		return mat		
	model=w2vector(string)
	matrix=split_line(model.wv)
	return matrix

def feed_matrix( m, z, batch_size, nsteps, input_size):
	matrix = np.zeros((batch_size, nsteps, input_size))
	#print batch_size
	t = (z-1)*batch_size
	for j in range(batch_size):
		matrix[j] = m[t+j]
		#print matrix[j]
	
	list_inputs=[]
	
	for j in range(nsteps):
		temp = np.zeros((batch_size, input_size)) 
		for k in range(batch_size):
			temp[k] = matrix[k][j]
		list_inputs.append(temp)
		#print list_inputs
	return list_inputs
			
			
			
			
			
			
			
	
