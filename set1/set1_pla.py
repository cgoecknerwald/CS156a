# Claire Goeckner-Wald
# CS156a set 1
# Perceptron Learning Algorithm

from __future__ import division
import random
import numpy

# returns [the number of iterations for h to approximate f, the approximate correctness of h to f]
# n is the number of points that the algorithm is trained on
def perceptron(N):
	# data space X = [-1, 1] by [-1, 1]
	# dimensionality = 2

	# create a starting line
	point1 = [random.uniform(-1,1), random.uniform(-1,1)]
	point2 = [random.uniform(-1,1), random.uniform(-1,1)]
	slope = (point1[1] - point2[1]) / (point1[0] - point2[0])

	# Our target function, f, is a random line.
	def f(x): 
		return slope*(x - point1[0]) + point1[1]

	# Classifies a point (x1,x2) based on location relative to line
	# If the point is above the line, return 1. If on the line, 0. Otherwise, -1.
	# This is based off of our target function 'f' 
	def classify(x1, x2):
		return numpy.sign(x2 - f(x1))

	# array of data points
	data_points = []

	weight_vector = [0, 0, 0]

	# generate the data points 
	# one point = [threshold=1, x coord, y coord]
	# N is supplied by user
	for n in range(N):
		data_points.append([1, random.uniform(-1,1),random.uniform(-1,1)])

	# Perceptron Learning Algorithm
	# The algorithm's function:
	# h(x) = sign(w dot x)
	# index is the index of the data_point in question
	# this function is dependent on a weight_vector
	def h(index, arr):
		return numpy.sign(weight_vector[0]*arr[index][0]+weight_vector[1]*arr[index][1]+weight_vector[2]*arr[index][2])
	
	# keep track whenever a misclassified point causes the weight vector to change
	num_iterations = 0

	# index is used to cycle through the data_point array
	# index+=1 is used to advance when a correctly classified point is discovered
	# index is reset to 0 when a misclassified point is discovered
	index = 0

	# the loop ends upon successful traversal of data_points without misclassified points
	# N is the length of data_points used initially
	while index < N:
		# DEBUGGING CODE :)
		# arr = []
		# for k in range(N):
		# 	arr.append("%d, %d" % (classify(data_points[k][1], data_points[k][2]), h(k)))
		# print index
		# print arr

		# discover if a point is misclassifed.
		# If the classification of the data_point at index 
		# is different from the algorithm function h's classification,
		# update the weight vector.
		correct_sign = classify(data_points[index][1], data_points[index][2])
		if h(index, data_points) != correct_sign:
			# weight_vector = weight_vector + sign*point_vector
			# j is used to iterate through the 3-d weight and point vectors
			for j in range(3):
				weight_vector[j] = weight_vector[j] + correct_sign*data_points[index][j]
			# since we updated a point, restart cycling through the array
			index = 0
			# keep track of the number of iterations
			num_iterations += 1
		# the point at index is not misclassified
		else:
			# update index to move forward in the array
			index += 1

	# generate C number of data points to test correctness
	# one point = [threshold=1, x coord, y coord]
	test_points = []
	num_incorrect = 0
	C = 100
	for n in range(C):
		test_points.append([1, random.uniform(-1,1),random.uniform(-1,1)])
	for c in range(C):
		correct_sign1 = classify(test_points[c][1], test_points[c][2])
		if h(c, test_points) != correct_sign1:
			num_incorrect += 1

	incorrectness = num_incorrect/C

	return [num_iterations, incorrectness] 

# perceptron code ends

# test PLA for N = 10
num_tests = 1000
sum_iterations = 0
sum_incorrectness = 0
for i in range(num_tests):
	sum_iterations += perceptron(10)[0]
	sum_incorrectness += perceptron(10)[1]

print ("N = 10")
print "Average iterations:", sum_iterations/num_tests
print "Average incorrectness:", sum_incorrectness/num_tests

# test PLA for N = 100
num_tests = 1000
sum_iterations = 0
sum_incorrectness = 0
for i in range(num_tests):
	sum_iterations += perceptron(100)[0]
	sum_incorrectness += perceptron(100)[1]

print ("\nN = 100")
print "Average iterations:", sum_iterations/num_tests
print "Average incorrectness:", sum_incorrectness/num_tests

# EXAMPLE OUTPUT FROM ONE RUN
#
# N = 10
# Average iterations: 0.882
# Average incorrectness: 0.11017
# 
# N = 100
# Average iterations: 881.502
# Average incorrectness: 0.01409


