from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import random
import sys
import numpy as np

def get_data(filename):
    '''
    Parses file to [[digit, intensity, symmetry], [digit, intensity, symmetry], ... ]
    '''
    return eval(open(filename).read())

def error(machine, X, Y):
    '''
    Returns the machines error (1 - accuracy) based on data X with correct labels Y
    '''
    return 1 - machine.score(X, Y)

def error_cross_validate(machine, X, Y, n_fold):
    '''
    Returns average cross-validation error, splitting in n_foldths.
    '''
    data = list(zip(X, Y))
    chunk_size = len(data) // n_fold
    sum_error = 0
    for chunk in range(0, len(data), chunk_size):
        x_train, y_train = zip(*(data[:chunk] + data[chunk+chunk_size:]))
        x_test, y_test = zip(*data[chunk:chunk+chunk_size])
        machine.fit(X=x_train, y=y_train)
        sum_error += (1 - machine.score(X=x_test, y=y_test))
    return sum_error/n_fold

def separate_data(data):
    '''
    Separates [[digit, intensity, symmetry], ... ] to [[digit, ...],[[intensity, symmetry], ...]]
    '''
    data_x, data_y = zip(*((x,y) for x, *y in data))
    return (list(data_x), list(data_y))

def del_data(X, Y, num1, num2):
    '''
    To be used in case of one-versus-one training. (Deletes values that are not num1 or num2.)
    '''
    data = list(filter(lambda dat: dat[0] in (num1, num2), zip(Y, X)))
    data_x, data_y = zip(*((x,y) for x, *y in data))
    return list(zip(*data))


def edit_data(Y, num):
    '''
    Creates num-versus-all data sets with supplied, ordered X and Y sets. 
    '''
    return list(map(lambda x: 1 if x == num else -1 , Y))

def progress_print():
    '''
    ... dots!
    '''
    print(".", end="")
    sys.stdout.flush()



def experiment(x_test, y_test, x_train, y_train, C=0.01, kernel='poly', Q=2, gamma=1.0, coef0=1.0, num1=None, num2=None, error_in=False, error_out=False, num_sv=False, error_cv = False):

    # Trims data to case of one-versus-one training. (Deletes values that are not num1 or num2.)
    # If num2 is not specified, then it is one-versus-all or all-versus-all. 
    if num2 is not None:
        y_test, x_test = del_data(x_test, y_test, num1, num2)
        y_train, x_train = del_data(x_train, y_train, num1, num2)

    # To be used in case of one-versus-all training. Sets values that are num1 to 1, and otherwise to -1. 
    if num1 is not None:
        y_test = edit_data(y_test, num1)
        y_train = edit_data(y_train, num1)

    # Create and fit the support vector machine.
    machine = SVC(C=C, kernel=kernel, degree=Q, gamma=gamma, coef0=coef0)
    machine.fit(X=x_train, y=y_train)
    
    # Deciding what to return 
    output = []
    if error_in: # In-sample error using the training set
        output.append(error(machine, x_train, y_train))
    if error_out: # Out-sample error using the testing set
        output.append(error(machine, x_test, y_test))
    if num_sv: # Number of support vectors used
        output.append(sum(list(machine.n_support_)))
    if error_cv:
        # reset machine just in case
        machine_cv =  SVC(C=C, kernel=kernel, degree=Q, gamma=gamma, coef0=coef0)
        output.append(error_cross_validate(machine_cv, x_train, y_train, n_fold=10))

    return output

def answer(q2=False, q3=False, q4=False, q5=False, q6=False, q7=False, q8=False, q9=False, q10=False):
    '''
    Allows user to dictate runtime by selectively flagging which questions should be answered.
    '''

    # Pull the data from the given files
    y_test, x_test = separate_data(get_data("features_test.txt"))
    y_train, x_train = separate_data(get_data("features_train.txt"))

    def question2():
        return 0


    # Answer and print the questions that are flagged.
    if q2:
        print("Question 2: " + str(question2()))
    

# Answer questions by selectively flagging which ones should be answered.
answer(q7=True, q8=True)