import nltk

l1,l2=0,0
c1,c2=0,0
with open ("Hindi") as f2:
	for line in f2:
		c2=c2+1
		str2=line.split(' ')
	print str2

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
#there may be several references
BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], str2)
print BLEUscore
