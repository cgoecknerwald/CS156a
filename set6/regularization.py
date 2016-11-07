import math
import numpy

# Turn a file into a list of lists of tuples
def get_data_set(filename):
    file = open(filename)
    data = []
    classification = []
    for line in file:
        nums = filter(bool, line.split())
        data.append((float(nums[0]), float(nums[1])))
        classification.append(float(nums[2]))
    return (data, classification)

def transform(x1, x2):
    return (1, x1, x2, x1**2, x2**2, x1*x2, abs(x1 - x2), abs(x1 + x2))

#[(x1, x2), ... (x41, x42)]
def phi(data):
    transformed_data = []
    for x1, x2 in data:
        transformed_data.append(transform(x1,x2))
    return transformed_data

def w_reg(Z, y, lambda_):
    z_sqrt = numpy.dot(numpy.transpose(Z),Z)
    calc = z_sqrt + lambda_*numpy.identity(z_sqrt.shape[0])
    calc2 = numpy.dot(numpy.linalg.inv(calc), numpy.transpose(Z))
    return numpy.dot(calc2, y)

def w_lin(Z, y):
    return numpy.dot(numpy.linalg.pinv(Z), y)

# g(x) = sign(w_transpose dot x)
def g(weight, x):
    return numpy.sign(numpy.dot(weight, x))

def error_in(weight, data, Y):
    sum_error = 0
    for x, y in zip(data, Y):
        hypothesis = g(weight, x)
        if (hypothesis != y):
            sum_error += 1
    return sum_error/float(len(Y))

def error_out(weight, data, Y):
    sum_error = 0
    for x, y in zip(data, Y):
        hypothesis = g(weight, x)
        if (hypothesis != y):
            sum_error += 1
    return sum_error/float(len(Y))


def experiment_linear():
    # [(x1, x2), ...] [y1, ...]
    data_in, y_in = get_data_set("in.txt")
    data_out, y_out = get_data_set("out.txt")
    # transform the data
    phi_in, phi_out = phi(data_in), phi(data_out)
    linear_weight = w_lin(phi_in, y_in)
    # Get the in-sample and out-sample errors
    in_error = error_in(linear_weight, phi_in, y_in)
    out_error = error_out(linear_weight, phi_out, y_out)
    return in_error, out_error

def experiment_regularization(k):
    lambda_ = 10**k
    # [(x1, x2), ...] [y1, ...]
    data_in, y_in = get_data_set("in.txt")
    data_out, y_out = get_data_set("out.txt")
    # transform the data
    phi_in, phi_out = phi(data_in), phi(data_out)
    regularized_weight = w_reg(phi_in, y_in, lambda_)
    # Get the in-sample and out-sample errors
    in_error = error_in(regularized_weight, phi_in, y_in)
    out_error = error_out(regularized_weight, phi_out, y_out)
    return in_error, out_error

# Question 2
print "\nlinear experiment: \n"
print experiment_linear()
# Questions 3 - 5
print "\nregularized experiment: \n"
for k in range(-3, 4):
    print "k:", k, "\t", experiment_regularization(k)
# Question 6
min_out = None
best_k = None
for k in range(-1000, 100):
    _, out = experiment_regularization(k)
    if min_out is None or out < min_out:
        best_k = k
        min_out = out
print best_k, min_out



# EXAMPLE OUTPUT

# linear experiment: 
# (0.02857142857142857, 0.084)
# regularized experiment: 
# k: -3   (0.02857142857142857, 0.08)
# k: -2   (0.02857142857142857, 0.084)
# k: -1   (0.02857142857142857, 0.056)
# k: 0    (0.0, 0.092)
# k: 1    (0.05714285714285714, 0.124)
# k: 2    (0.2, 0.228)
# k: 3    (0.37142857142857144, 0.436)
# -1 0.056
