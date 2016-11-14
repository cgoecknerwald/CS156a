# Claire Goeckner-Wald, set 7
import math
import numpy

# Turn a file into a list of lists of tuples
def get_data_set(filename):
    file = open(filename)
    data = []
    for line in file:
        data.append(list(map(float, filter(bool, line.split()))))
    return data

 #[(x1, x2), y ... (x41, x42), y] -> [(1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)), ...]
def phi(data, k):
    transformed_data = []
    for x1, x2, y in data:
        transformed_data.append(transform(x1, x2, y, k))
    return transformed_data

# (x1, x2), y -> (1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)), y
def transform(x1, x2, y, k):
    return ((1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2))[:k+1], y)

# g(x) = sign(w_transpose dot x)
def g(weight, x):
    return numpy.sign(numpy.dot(weight, x))

# Split data into 25 training points and 10 validation points
def split(data, N=25):
    return (data[:N], data[N:])

def w_lin(data):
    Z, y = zip(*data)
    return numpy.dot(numpy.linalg.pinv(Z), y)

def error(weight, data):
    sum_error = 0
    # unzip data
    Z, Y = zip(*data)
    for i in xrange(len(Y)):
        hypothesis = g(weight, Z[i])
        if (hypothesis != Y[i]):
            sum_error += 1
    return sum_error/float(len(Y))

def experiment(k, N=25, reversed=False):
    data_in, data_out = get_data_set("in.txt"), get_data_set("out.txt")
    # transform the data
    phi_in, phi_out = phi(data_in, k), phi(data_out, k)
    if reversed: # 25 validation pts and 10 training pts
        validation,training = split(phi_in, N)
    else: # 25 training pts and 10 validation pts
        training, validation = split(phi_in, N)
    linear_weight = w_lin(training)
    # Get all the errors
    in_error = error(linear_weight, training)
    val_error = error(linear_weight, validation)
    out_error = error(linear_weight, phi_out)
    return in_error, val_error, out_error

# Questions 1, 2
print "Problems 1&2: "
for k in xrange(3, 8):
    print "k:", k, ":", experiment(k, N=25, reversed=False)
# Questions 3, 4
print "Problems 3&4: "
for k in xrange(3, 8):
    print "k:", k, ":", experiment(k, N=25, reversed=True)

# EXAMPLE OUTPUT
# Problems 1&2: 
# k: 3 : (0.44, 0.3, 0.42)
# k: 4 : (0.32, 0.5, 0.416)
# k: 5 : (0.08, 0.2, 0.188)
# k: 6 : (0.04, 0.0, 0.084)
# k: 7 : (0.04, 0.1, 0.072)
# Problems 3&4: 
# k: 3 : (0.4, 0.28, 0.396)
# k: 4 : (0.3, 0.36, 0.388)
# k: 5 : (0.2, 0.2, 0.284)
# k: 6 : (0.0, 0.08, 0.192)
# k: 7 : (0.0, 0.12, 0.196)

