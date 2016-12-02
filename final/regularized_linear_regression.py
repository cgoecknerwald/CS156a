# Claire Goeckner-Wald, Final
import math
import numpy

def get_data(filename):
    '''
    Parses file to [[digit, intensity, symmetry], [digit, intensity, symmetry], ... ]
    '''
    return eval(open(filename).read())

def separate_data(data):
    '''
    Separates [[digit, intensity, symmetry], ... ] to [[digit, ...],[[intensity, symmetry], ...]]
    '''
    data_x, data_y = zip(*((x,y) for x, *y in data))
    return (list(data_x), list(data_y))


def edit_data(Y, num):
    '''
    Creates num-versus-all data sets with supplied, ordered X and Y sets. 
    '''
    return list(map(lambda x: 1 if x == num else -1 , Y))

def del_data(X, Y, num1, num2):
    '''
    To be used in case of one-versus-one training. (Deletes values that are not num1 or num2.)
    '''
    data = list(filter(lambda dat: dat[0] in (num1, num2), zip(Y, X)))
    data_x, data_y = zip(*((x,y) for x, *y in data))
    return list(zip(*data))

def transform(x1, x2):
    return (1, x1, x2, x1*x2, x1**2, x2**2)

def add_bias(x1, x2):
    return (1, x1, x2)

#[(x1, x2), ... (x41, x42)]
def phi(data, to_transform):
    transformed_data = []
    for x1, x2 in data:
        if to_transform:
            transformed_data.append(transform(x1,x2))
        else:
            transformed_data.append(add_bias(x1,x2))
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

def experiment_regularization(num1=None, num2=None, lambda_=1, to_transform=True):
    # [(x1, x2), ...] [y1, ...]
    y_test, x_test = separate_data(get_data("features_test.txt"))
    y_train, x_train = separate_data(get_data("features_train.txt"))

    # Trims data to case of one-versus-one training. (Deletes values that are not num1 or num2.)
    # If num2 is not specified, then it is one-versus-all or all-versus-all. 
    if num2 is not None:
        y_test, x_test = del_data(x_test, y_test, num1, num2)
        y_train, x_train = del_data(x_train, y_train, num1, num2)

    # To be used in case of one-versus-all training. Sets values that are num1 to 1, and otherwise to -1. 
    if num1 is not None:
        y_test = edit_data(y_test, num1)
        y_train = edit_data(y_train, num1)

    # transform the data
    phi_train, phi_test = phi(x_train, to_transform), phi(x_test, to_transform)
    weight = w_reg(phi_train, y_train, lambda_)
    # Get the in-sample and out-sample errors
    in_error = error_in(weight, phi_train, y_train)
    out_error = error_out(weight, phi_test, y_test)
    return in_error, out_error

# Ghost
# print("Ghost:\n")
# for num1 in range(0,5):
#     print("num:", num1, "\t(E_in, E_out):", experiment_regularization(num1=num1, lambda_ = 0.5, to_transform=True))

# Question 7
print ("\nQuestion 7: \n")
for num1 in range(5,10):
    print("num:", num1, "\t(E_in, E_out):", experiment_regularization(num1=num1, lambda_=1, to_transform=False))
# Question 8
print ("\nQuestion 8: \n")
for num1 in range(0, 5):
    print("num:", num1, "\t(E_in, E_out):", experiment_regularization(num1=num1, lambda_=1, to_transform=True))
# Question 9
print ("\nQuestion 9: \n")
for num1 in range(0, 10):
    not_trans = experiment_regularization(num1=num1, lambda_=1, to_transform=False)
    trans = experiment_regularization(num1=num1, lambda_=1, to_transform=True)
    print("num:", num1, "\t not trans / trans (E_in, E_out):", not_trans, ",", trans)
# Question 10
print ("\nQuestion 10: \n")
for lambd in [0.01, 1]:
    print("lambda: %.3f" % lambd, "\t(E_in, E_out):", experiment_regularization(num1=1, num2=5, lambda_=lambd, to_transform=True))

# Question 7: 

# num: 5  (E_in, E_out): (0.07625840076807022, 0.07972097658196313)
# num: 6  (E_in, E_out): (0.09107118365107666, 0.08470353761833582)
# num: 7  (E_in, E_out): (0.08846523110684405, 0.07324364723467862)
# num: 8  (E_in, E_out): (0.07433822520916199, 0.08271051320378675)
# num: 9  (E_in, E_out): (0.08832807570977919, 0.08819133034379671)

# Question 8: 

# num: 0  (E_in, E_out): (0.10231792621039638, 0.10662680617837568)
# num: 1  (E_in, E_out): (0.012343985735838706, 0.02192326856003986)
# num: 2  (E_in, E_out): (0.10026059525442327, 0.09865470852017937)
# num: 3  (E_in, E_out): (0.09024825126868742, 0.08271051320378675)
# num: 4  (E_in, E_out): (0.08942531888629818, 0.09965122072745392)

# Question 9: 

# num: 0   not trans / trans (E_in, E_out): (0.10931285146070498, 0.11509715994020926) , (0.10231792621039638, 0.10662680617837568)
# num: 1   not trans / trans (E_in, E_out): (0.01522424907420107, 0.02242152466367713) , (0.012343985735838706, 0.02192326856003986)
# num: 2   not trans / trans (E_in, E_out): (0.10026059525442327, 0.09865470852017937) , (0.10026059525442327, 0.09865470852017937)
# num: 3   not trans / trans (E_in, E_out): (0.09024825126868742, 0.08271051320378675) , (0.09024825126868742, 0.08271051320378675)
# num: 4   not trans / trans (E_in, E_out): (0.08942531888629818, 0.09965122072745392) , (0.08942531888629818, 0.09965122072745392)
# num: 5   not trans / trans (E_in, E_out): (0.07625840076807022, 0.07972097658196313) , (0.07625840076807022, 0.07922272047832586)
# num: 6   not trans / trans (E_in, E_out): (0.09107118365107666, 0.08470353761833582) , (0.09107118365107666, 0.08470353761833582)
# num: 7   not trans / trans (E_in, E_out): (0.08846523110684405, 0.07324364723467862) , (0.08846523110684405, 0.07324364723467862)
# num: 8   not trans / trans (E_in, E_out): (0.07433822520916199, 0.08271051320378675) , (0.07433822520916199, 0.08271051320378675)
# num: 9   not trans / trans (E_in, E_out): (0.08832807570977919, 0.08819133034379671) , (0.08832807570977919, 0.08819133034379671)

# Question 10: 

# lambda: 0.010   (E_in, E_out): (0.004484304932735426, 0.02830188679245283)
# lambda: 1.000   (E_in, E_out): (0.005124919923126201, 0.025943396226415096)
