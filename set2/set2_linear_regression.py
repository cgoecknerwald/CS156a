# Claire Goeckner-Wald
# CS156a set 2
# Linear Regression

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
        #   arr.append("%d, %d" % (classify(data_points[k][1], data_points[k][2]), h(k)))
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

    # vector = [threshold=0, weight, weight]
    weight_vector = [0, 0, 0]

    # to minimize E_in, find w = psuedo-inverse of x * y
    weight_vector = numpy.dot(numpy.linalg.pinv(data_points), y)

    # Linear Regression
    # The algorithm's function:
    # g(x) = sign(w_transpose dot x)
    # index is the index of the data_points's point in question
    # this function is dependent on a weight_vector
    # arr is the data_points's point at index's info stored in an array
    def g(index, arr):
        return numpy.sign(numpy.dot(weight_vector, arr[index]))

    # IN SAMPLE INCORRECTNESS FOR ALL N DATA_POINTS
    num_incorrect = 0
    for n in xrange(N):
        # get the correct sign
        correct_sign = classify(data_points[n][1], data_points[n][2])
        # compare the correct sign to what the function g predicts
        if g(n, data_points) != correct_sign:
            # increment the number of incorrectness if necessary
            num_incorrect += 1
    # calculate the proportion of in-sample points found to be incorrect
    in_sample_error = num_incorrect/N

    # OUT OF SAMPLE INCORRECTNESS FOR 1000 FRESH TEST POINTS
    # generate num_test_points data points to test correctness
    # one point = [threshold=1, x coord, y coord]
    test_points = []
    num_test_points = 1000 
    # generate the test points
    for k in xrange(num_test_points):
        test_points.append([1, random.uniform(-1,1),random.uniform(-1,1)])
    # classify the test points and compare it to g(x)'s classification
    num_incorrect1 = 0
    for n1 in xrange(num_test_points):
        # get the correct signh
        correct_sign1 = classify(test_points[n1][1], test_points[n1][2])
        # compare the correct sign to what the function g predicts
        if g(n1, test_points) != correct_sign1:
            # increment the number of incorrectness if necessary
            num_incorrect1 += 1

    # calculate the proportion of out-sample points found to be incorrect
    out_sample_error = num_incorrect1/num_test_points

    # return the weight vector, too, for problem #6
    return [in_sample_error, out_sample_error, weight_vector]

# \\ linear regression function

# test linear regression for N = 100 classified points
n = 100
# number of tests
num_tests = 100
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
print "number of tests:", num_tests
print "N = 100"
print "Average in-sample error:", sum_in_sample_error/num_tests
print "Average out-sample error:", sum_out_sample_error/num_tests

# EXAMPLE OUTPUT
# number of tests: 10000
# N = 100
# Average in-sample error: 0.039185
# Average out-sample error: 0.0484795
# [Finished in 80.5s]

