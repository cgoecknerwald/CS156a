import math
import random
import numpy as np

# Gradient function
def gradient(pt, y, w):
	return -y*pt/(1+math.exp(y*np.dot(w,pt)))

# Return N random points (1, [-1,1], [-1,1])
def points(N):
	return [[1, random.uniform(-1,1),random.uniform(-1,1)] for n in xrange(N)]

# Classify N pts based on supplied function f
def classify(pts, f):
	return [np.sign(pt[2] - f(pt[1])) for pt in pts]

def error_out(w, f):
	pts = points(1000)
	y = classify(pts, f)
	pts = zip(pts, y)
	error = 0
	for pt,y in pts:
		error += math.log(1 + math.exp(-y * np.dot(w, pt)))
	return error/1000


def experiment(dimension, N,eta):
	pts = points(N)
	# create a random line
	x1,y1 = -1,-1
	x2,y2 = 1,1
	# x1,y1 = random.uniform(-1,1), random.uniform(-1,1)
	# x2,y2 = random.uniform(-1,1), random.uniform(-1,1)
	slope = (y2-y1)/(x2-x1)
	# classify based on the random line
	y = classify(pts, lambda x: slope*(x - x1) + y1)
	pts = zip(pts, y)
	w = np.array([0]*(dimension+1))
	w2 = np.array([1]*(dimension+1))
	epoch = 0
	while(np.linalg.norm(w - w2) > 0.01):
		w2 = w
		random.shuffle(pts)
		# Go through each point, updating the weight
		for pt,y in pts:
			w = w - eta*gradient(np.array(pt), y, w)
		epoch += 1

	# Find E_out
	err = error_out(w, lambda x: slope*(x - x1) + y1)

	return (epoch, np.ndarray.tolist(w), err)
		
def run_experiment(num, dimension, N, eta):
	sum_ = np.array([0] * 3)
	epoch_ = 0
	err_ = 0
	for i in xrange(num):
		out = experiment(dimension, N, eta)
		sum_ = sum_ + out[1]
		epoch_ = epoch_ + out[0]
		err_ = err_ + out[2]
	return (epoch_/num, sum_/num, err_/num)

# 2 dimensions, 100 data points, eta = 0.01
print run_experiment(10, 2, 100, 0.01)

# EXAMPLE OUTPUT
# (388, array([ 0.01597646, -7.2606088 ,  7.1004079 ]), 0.11362448539496177)
