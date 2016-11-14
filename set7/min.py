import random

def experiment():
	e_1, e_2 = random.uniform(0,1), random.uniform(0,1)
	return min(e_1, e_2)

num = 1000000
sum_ = 0
for n in xrange(num):
	sum_ += experiment()
print sum_/num