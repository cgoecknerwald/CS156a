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
    return numpy.dot(numpy.inverse(calc), y)

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


def experiment():
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



print experiment()

