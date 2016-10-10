# Claire Goeckner-Wald
# CS156a set 2
# Linear Regression + Nonlinear Transformation

from __future__ import division
import random
import numpy

# returns in_sample_error
# n is the number of points that the algorithm is trained on
def linear_regression_no_transform(N):
    # data space X = [-1, 1] by [-1, 1]
    # dimensionality = 2

    # Assigned target function
    def f(x1, x2):
        return numpy.sign(x1**2 + x2**2 - 0.6)

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
        correct_sign = f(data_points[n][1] , data_points[n][2])
        # 10% chance of incorrect label = noise
        if random.randrange(10) == 1: 
            y.append(-correct_sign)
        else: 
            y.append(correct_sign)

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
            correct_sign = f(data_points[n][1], data_points[n][2])
            # compare the correct sign to what the function g predicts
            if g(data_points[n]) != correct_sign:
                # increment the number of incorrectness if necessary
                num_incorrect_in += 1
        # calculate the proportion of in-sample points found to be incorrect
        return num_incorrect_in/N

    # return the weight vector, too, for problem #6
    return (get_in_sample_error(), data_points)

# \\ linear regression no transform function

# returns [out_sample_error, weight_vector]
# n is the number of points that the algorithm is trained on
def linear_regression_transform(N, data_points):
    # data space X = [-1, 1] by [-1, 1]
    # dimensionality = 2

    # Assigned target function
    def f(x1, x2):
        return numpy.sign(x1**2 + x2**2 - 0.6)

    # transform the data points 
    # one point = [bias=1, x coord, y coord, xy, x^2, y^2]
    # N is supplied by user
    for n in range(N):
        x1 = data_points[n][1]
        x2 = data_points[n][2]
        data_points[n].extend([x1*x2, x1**2, x2**2])

    # classify the data points
    y = []
    for n in range(N):
        correct_sign = f(data_points[n][1] , data_points[n][2])
        # 10% chance of incorrect label = noise
        if random.randrange(10) == 1: 
            y.append(-correct_sign)
        else: 
            y.append(correct_sign)

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

    # OUT OF SAMPLE INCORRECTNESS FOR 1000 FRESH TEST POINTS
    # generate num_test_points data points to test correctness
    # one point = [threshold=1, x coord, y coord]
    def get_out_sample_error():
        num_test_points = 1000 
        num_incorrect_out = 0
        # generate the test points
        for k in xrange(num_test_points):
            # create a test point
            x1 = random.uniform(-1,1)
            x2 = random.uniform(-1,1)
            test_point = [1, x1, x2, x1*x2, x1**2, x2**2]
            # classify the test points and compare it to g(x)'s classification
            # get the correct sign
            correct_sign1 = f(test_point[1], test_point[2])
            # compare the correct sign to what the function g predicts
            if g(test_point) != correct_sign1:
                # increment the number of incorrectness if necessary
                num_incorrect_out += 1
        # calculate the proportion of out-sample points found to be incorrect
        return num_incorrect_out/num_test_points

    # return the weight vector, too, for problem #6
    return (get_out_sample_error(), weight_vector)

# \\ linear regression no transform function

def experiment(num_tests):
    # test linear regression for N = 1000 classified points
    N = 1000
    # keep track of the avg incorrectness
    sum_in_sample_error = 0
    sum_out_sample_error = 0
    for i in range(num_tests):
        temp_arr = linear_regression_no_transform(N)
        sum_in_sample_error += temp_arr[0]
        # we reuse the data_points array
        data_points = temp_arr[1]
        sum_out_sample_error += linear_regression_transform(N, data_points)[0]

    # Print the results
    print "Number of tests of no-transform linear regression:", num_tests
    print "\nN =", N
    print "Average in-sample error:", sum_in_sample_error/num_tests
    print "Number of tests of transformed linear regression:", num_tests
    print "\nN =", N
    print "Average out-sample error:", sum_out_sample_error/num_tests
  
def experiment_prob8(num_tests):
    # Actual hypothesis. x is an array of length 6
    def g(x):
        return numpy.sign(numpy.dot(weight_vector, x))
    def g_a(x1, x2):
        return numpy.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1**2 + 1.5*x2**2)
    def g_b(x1, x2):
        return numpy.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 1.5*x1**2 + 15*x2**2)
    def g_c(x1, x2):
        return numpy.sign(-1 - 0.05*x1 + 0.08*x2 + 0.13*x1*x2 + 15*x1**2 + 1.5*x2**2)
    def g_d(x1, x2):
        return numpy.sign(-1 - 1.5*x1 + 0.08*x2 + 0.13*x1*x2 + 0.05*x1**2 + 0.05*x2**2)
    def g_e(x1, x2):
        return numpy.sign(-1 - 0.05*x1 + 0.08*x2 + 1.5*x1*x2 + 0.15*x1**2 + 0.15*x2**2)

    # test linear regression for N = 1000 classified points
    N = 1000
    num_test_points = 1000
    g_accuracy_arr = [0,0,0,0,0]
    for i in range(num_tests):
        # we reuse the data_points array
        data_points = linear_regression_no_transform(N)[1]
        # keep track of g(x) by storing the weight vector
        weight_vector = linear_regression_transform(N, data_points)[1]
        g_accuracy_arr = [0,0,0,0,0]
        for i in xrange(num_test_points):
            x1 = random.uniform(-1,1)
            x2 = random.uniform(-1,1)
            test_point = [1, x1, x2, x1*x2, x1**2, x2**2]
            sign_g = g(test_point)
            g_accuracy_arr[0] += int(sign_g == g_a(x1, x2))
            g_accuracy_arr[1] += int(sign_g == g_b(x1, x2))
            g_accuracy_arr[2] += int(sign_g == g_c(x1, x2))
            g_accuracy_arr[3] += int(sign_g == g_d(x1, x2))
            g_accuracy_arr[4] += int(sign_g == g_e(x1, x2))
        for k in xrange(5):
            g_accuracy_arr[k] = g_accuracy_arr[k]/num_test_points
        print g_accuracy_arr
        # \\ num_test_points

    # \\ num_tests

# RUN THE EXPERIMENT
num_tests = 1000
experiment(num_tests)
num_tests = 5
experiment_prob8(num_tests)

# EXAMPLE OUTPUT
# Number of tests of no-transform linear regression: 1000
#
# N = 1000
# Average in-sample error: 0.51429
#
# Number of tests of transformed linear regression: 1000
#
# N = 1000
# Average out-sample error: 0.033848