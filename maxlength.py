def max_length():
	l1,l2=0,0
	c1,c2=0,0
	with open ("English") as f1:
		for line in f1:
			c1=c1+1
			str1=line.split(' ')
			#print str1
			if len(str1)>l1:
				l1=len(str1)
			#print l1
	with open ("eng") as f2:
		for line in f2:
			c2=c2+1
			str2=line.split(' ')
			#print str2
			if len(str2)>l2:
				l2=len(str2)
	if(l1>l2):
		return (l1, c1)
	else:
		return (l2, c2)
max_length()
'''a=max_len()
print a'''
