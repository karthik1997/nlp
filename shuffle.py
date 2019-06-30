import random
'''
a = ['a', 'b', 'c']
b = [1, 2, 3]

c = list(zip(a, b))

random.shuffle(c)

a, b = zip(*c)

print a
print b
str1=[]
def shuffle():
	l1,l2=0,0
	c1,c2=0,0
	with open ("EngSmall") as f1:
		for line in f1:
			c1=c1+1
			str1=line.split('.')
			#print str1
	with open ("TelSmall") as f2:
		for line in f2:
			c2=c2+1
			str2=line.split('.')
	c=list(zip(str1,str2))
	random.shuffle(c)
	str1,str2=zip(*c)

	print str1[0]
	print str2[0]
thefile = open('test.txt', 'w')
for item in str1:
	thefile.write("%s\n" % item)
shuffle()'''
with open('EngSmall', 'r') as f:
    x = f.readlines()
#print x
with open('TelSmall', 'r') as f:
    y = f.readlines()
#print y
c=list(zip(x,y))
random.shuffle(c)
x,y=zip(*c)
#print x
thefile = open('test.txt', 'w')
for item in x:
	thefile.write("%s" % item)
thefile = open('test1.txt', 'w')
for item in y:
	thefile.write("%s" % item)
