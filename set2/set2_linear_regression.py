# Claire Goeckner-Wald
# CS156a set 2
# Linear Regression

from __future__ import division
import random
import numpy

# returns [the number of iterations for h to approximate f, the approximate correctness of h to f]
# n is the number of points that the algorithm is trained on
def linear_regression(N):
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

    # generate the data points 
    # one point = [bias=1, x coord, y coord]
    # N is supplied by user
    for n in range(N):
        data_points.append([1, random.uniform(-1,1),random.uniform(-1,1)])

    # classify the data points
    y = []
    for n in range(N):
        y.append(classify(data_points[n][1] , data_points[n][2]))

    # to minimize E_in, find w = psuedo-inverse of x * y
    weight_vector = numpy.dot(numpy.linalg.pinv(data_points), y)

    # Linear Regression
    # The algorithm's function:
    # g(x) = sign(w_transpose dot x)
    # index is the index of the data_points's point in question
    # this function is dependent on a weight_vector
    # arr is the data_points's point at index's info stored in an array
    def g(x):
        return numpy.sign(numpy.dot(weight_vector, x))

    # IN SAMPLE INCORRECTNESS FOR ALL N DATA_POINTS
    def get_in_sample_error():
        num_incorrect_in = 0
        for n in xrange(N):
            # get the correct sign
            correct_sign = classify(data_points[n][1], data_points[n][2])
            # compare the correct sign to what the function g predicts
            if g(data_points[n]) != correct_sign:
                # increment the number of incorrectness if necessary
                num_incorrect_in += 1
        # calculate the proportion of in-sample points found to be incorrect
        return num_incorrect_in/N


    # OUT OF SAMPLE INCORRECTNESS FOR 1000 FRESH TEST POINTS
    # generate num_test_points data points to test correctness
    # one point = [threshold=1, x coord, y coord]
    def get_out_sample_error():
        num_test_points = 1000 
        num_incorrect_out = 0
        # generate the test points
        for k in xrange(num_test_points):
            test_point = [1, random.uniform(-1,1),random.uniform(-1,1)]
            # classify the test points and compare it to g(x)'s classification
            # get the correct sign
            correct_sign1 = classify(test_point[1], test_point[2])
            # compare the correct sign to what the function g predicts
            if g(test_point) != correct_sign1:
                # increment the number of incorrectness if necessary
                num_incorrect_out += 1
        # calculate the proportion of out-sample points found to be incorrect
        return num_incorrect_out/num_test_points

    # return the weight vector, too, for problem #6
    return (get_in_sample_error(), get_out_sample_error(), weight_vector)

# \\ linear regression function

def experiment(num_tests):
    # test linear regression for N = 100 classified points
    n = 100
    # keep track of the avg incorrectness
    sum_in_sample_error = 0
    sum_out_sample_error = 0
    # keep track of g(x) by storing the weight vector
    weight_vector_arr = []
    for i in range(num_tests):
        temp_arr = linear_regression(n)
        sum_in_sample_error += temp_arr[0]
        sum_out_sample_error += temp_arr[1]
        weight_vector_arr.append(temp_arr[2])

    # Print the results
    print "Number of tests:", num_tests
    print "N = 100"
    print "Average in-sample error:", sum_in_sample_error/num_tests
    print "Average out-sample error:", sum_out_sample_error/num_tests

# RUN THE EXPERIMENT
num_tests = 100
experiment(num_tests)

# EXAMPLE OUTPUT
# number of tests: 10000
# N = 100
# Average in-sample error: 0.039185
# Average out-sample error: 0.0484795
# [Finished in 80.5s]

