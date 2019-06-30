import gensim, logging, os

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def w2vector(s):
	if(s=='English'):
		with open(s) as f:
			st=[]
			for line in f:
				#list1=line.split('\n')
				list1=line.split(' ')
				st.append(list1)
		fname = 'eng.model'
	#	sentences = gensim.models.word2vec.LineSentence(infile) # a memory-friendly iterator
		model = Word2Vec(st,10, min_count=1)
		model.save(fname)
		return model
	else:
		with open(s) as f:
			st=[]
			for line in f:
				list1=line.split(' ')
				#list1=line.split('\n')
				st.append(list1)
		fname = 'hindi.model'
		
	#	sentences = gensim.models.word2vec.LineSentence('TelSmall') # a memory-friendly iterator
		
		model = Word2Vec(st,10, min_count=1)
		#print model
		model.save(fname)
		
		return model
	
	#print model
#w2vector('TelSmall')
