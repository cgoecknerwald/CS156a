# Claire Goeckner-Wald
# CS156a set 2
# Linear Regression

from __future__ import division
import random
import numpy

# returns [num_iterations for h to approximate f, incorrectness]
# n is the number of points that the algorithm is trained on
# This function differs from set1_pla.py's implementation
def perceptron(N, weight_vector, data_points, y, f):
    # data space X = [-1, 1] by [-1, 1]
    # dimensionality = 2

    # Perceptron Learning Algorithm
    # The algorithm's function:
    # h(x) = sign(w dot x)
    # index is the index of the data_point in question
    # this function is dependent on a weight_vector
    def h(x):
        return numpy.sign(numpy.dot(weight_vector, x))

    def update_weight_vector(weight_vector, data_point, correct_sign):
        for j in xrange(len(weight_vector)):
            weight_vector[j] = weight_vector[j] + correct_sign*data_point[j]
        return weight_vector

    def iterate(weight_vector):
        # keep track whenever a misclassified point causes the weight vector to change
        num_iterations = 0
        # index used to determine when to quit iterating
        index = 0
        # the loop ends upon successful traversal of data_points without misclassified points
        while index < N:
            # Find the correct sign of the data_point[index] using the correct classifications 
            correct_sign = y[index]
            # the point is misclassified by h
            if h(data_points[index]) != correct_sign:
                # weight_vector = weight_vector + sign*point_vector
                # j is used to iterate through the 3-d weight and point vectors
                weight_vector = update_weight_vector(weight_vector, data_points[index], correct_sign)
                # since we updated a point, restart cycling through the array
                index = 0
                # keep track of the number of iterations
                num_iterations += 1
            # the point at index is not misclassified
            else:
                # update index to move forward in the array
                index += 1

        return num_iterations

    return iterate(weight_vector)

# \\ perceptron function

# returns [in_sample_error, out_sample_error, all weight_vectors, the data_points trained on, 
# the classifications, and the function used to classify]
# N is the number of points that the algorithm is trained on
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
    return (get_in_sample_error(), get_out_sample_error(), weight_vector, data_points, y, f)

# \\ linear regression function

def experiment_linear(num_tests):
    # test linear regression for N = 100 classified points
    N = 100
    # keep track of the avg incorrectness
    sum_in_sample_error = 0
    sum_out_sample_error = 0
    for i in range(num_tests):
        in_sample_error, out_sample_error, _, _, _, _ = linear_regression(N)
        sum_in_sample_error += in_sample_error
        sum_out_sample_error += out_sample_error
    # Print the results
    print "Number of tests:", num_tests
    print "\nN = 100"
    print "Average in-sample error:", sum_in_sample_error/num_tests
    print "Average out-sample error:", sum_out_sample_error/num_tests

# prints the average number of iterations 
def experiment_pla(num_tests):
    # test linear regression for N = 10 classified points
    N = 10
    # keep track of the avg number of iterations
    sum_num_iterations = 0
    for i in range(num_tests):
        _, _, weight_vector, data_points, y, f = linear_regression(N)
        weight_vector = numpy.ndarray.tolist(weight_vector)
        # pla_arr is num_iterations for h to approximate f
        sum_num_iterations += perceptron(len(data_points), weight_vector, data_points, y, f)
    print "\nAverage number of PLA iterations:", sum_num_iterations/num_tests


# RUN THE EXPERIMENT
num_tests = 10000
experiment_linear(num_tests)
experiment_pla(num_tests)

# EXAMPLE OUTPUT
# Number of tests: 10000
#
# N = 100
# Average in-sample error: 0.039191
# Average out-sample error: 0.0480881
#
# Average number of PLA iterations: 11.5908
# [Finished in 13.9s]
