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

    # generate the data points 
    # one point = [bias=1, x coord, y coord]
    # N is supplied by user
    data_points = []
    for n in range(N):
        data_points.append([1, random.uniform(-1,1),random.uniform(-1,1)])

    # Perceptron Learning Algorithm
    # The algorithm's function:
    # h(x) = sign(w dot x)
    # index is the index of the data_point in question
    # this function is dependent on a weight_vector
    weight_vector = [0, 0, 0]
    def h(x):
        return numpy.sign(numpy.dot(weight_vector, x))


    def iterate():
        # keep track whenever a misclassified point causes the weight vector to change
        num_iterations = 0
        index = 0
        # the loop ends upon successful traversal of data_points without misclassified points
        while index < N:
            # DEBUGGING CODE :)
            # arr = []
            # for k in range(N):
            #   arr.append("%d, %d" % (classify(data_points[k][1], data_points[k][2]), h(k)))
            # print index
            # print arr

            # discover if a point is misclassifed.
            # If the classification of the data_point at index 
            # is different from the algorithm function h's classification,
            # update the weight vector.
            correct_sign = classify(data_points[index][1], data_points[index][2])
            if h(data_points[index]) != correct_sign:
                # weight_vector = weight_vector + sign*point_vector
                # j is used to iterate through the 3-d weight and point vectors
                for j in xrange(3):
                    weight_vector[j] = weight_vector[j] + correct_sign*data_points[index][j]
                # since we updated a point, restart cycling through the array
                index = 0
                # keep track of the number of iterations
                num_iterations += 1
            # the point at index is not misclassified
            else:
                # update index to move forward in the array
                index += 1

        return num_iterations

    # generate num_test_points number of data points to test correctness
    # one point = [threshold=1, x coord, y coord]
    def get_incorrectness():
        num_incorrect = 0
        num_test_points = 100
        for n in xrange(num_test_points):
            test_point = [1, random.uniform(-1,1),random.uniform(-1,1)]
            correct_sign1 = classify(test_point[1], test_point[2])
            if h(test_point) != correct_sign1:
                num_incorrect += 1

        return num_incorrect/num_test_points

    return [iterate(), get_incorrectness()] 

# perceptron code ends

def experiment(num_tests):
    # test PLA for N = 10
    N = 10
    sum_iterations = 0
    sum_incorrectness = 0
    for i in range(num_tests):
        temp_arr = perceptron(N)
        sum_iterations += temp_arr[0]
        sum_incorrectness += temp_arr[1]

    print "Number of tests:", num_tests
    print ("\nN = 10")
    print "Average iterations:", sum_iterations/num_tests
    print "Average incorrectness:", sum_incorrectness/num_tests

    # test PLA for N = 100
    N = 100
    sum_iterations = 0
    sum_incorrectness = 0
    for i in range(num_tests):
        temp_arr = perceptron(N)
        sum_iterations += temp_arr[0]
        sum_incorrectness += temp_arr[1]

    # Print out the data
    print ("\nN = 100")
    print "Average iterations:", sum_iterations/num_tests
    print "Average incorrectness:", sum_incorrectness/num_tests

# RUN THE EXPERIMENT 
num_tests = 100
experiment(num_tests)

# EXAMPLE OUTPUT FROM ONE RUN
# Number of tests: 10000
# 
# N = 10
# Average iterations: 11.29
# Average incorrectness: 0.11017
# 
# N = 100
# Average iterations: 165.03
# Average incorrectness: 0.01409


