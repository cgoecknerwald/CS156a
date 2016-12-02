from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import random
import sys
import numpy as np
import math

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

def error_lloyds(w, X, Y, k, g, c):
    sum_error = 0
    for x1, y in zip(X, Y):
        hypothesis = h(x=x1, K=k, weights=w, gamma=g, centers=c)
        if (hypothesis != y):
            sum_error += 1
    return sum_error/float(len(Y))

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

def get_weight(phi, y):
    return np.dot(np.linalg.pinv(phi), y)

def progress_print():
    '''
    ... dots!
    '''
    print(".", end="")
    sys.stdout.flush()

def f(x):
    x1, x2 = x
    return np.sign(x2 - x1 + 0.25*math.sin(math.pi*x1))

def lloyds_f(gamma, x, mu):
    return math.exp(-gamma*((x[0] - mu[0])**2 + (x[1] - mu[1])**2))

def h(x, K, weights, gamma, centers):
    sum_ = 0
    for k in range(1, K): # don't hit w[0], which is b
        sum_ += weights[k]*lloyds_f(gamma, x, centers[k])
    return np.sign(sum_ + weights[0])

def generate_data(num_points=0):
    x_data = []
    y_data = []
    for _ in range(num_points):
        pt = (random.uniform(-1, 1), random.uniform(-1, 1))
        x_data.append(pt)
        y_data.append(f(pt))
    return (y_data, x_data)

def k_random_centers(k):
    centers = []
    for i in range(k):
        centers.append( (random.uniform(-1, 1), random.uniform(-1, 1)) )
    return centers

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

def lloyds(x_test, y_test, x_train, y_train, k=9, gamma=None, error_in=False, error_out=False):
    def get_centers(x_train):
        while True:
            centers = k_random_centers(k) # (x1, x2)
            cluster_pts = [[] for _ in range(k)] 
            quit = False
            while (not quit):
                cluster_pts = [[] for _ in range(k)]
                # Put all pts in the cluster_pts array
                for pt in x_train: # (x1, x2)
                    closest_k_index = -100000
                    closest_dist = 10000
                    for i, x in enumerate(centers):
                        dist = math.sqrt((pt[0]-x[0])**2 + (pt[1]-x[1])**2)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_k_index = i
                    cluster_pts[closest_k_index].append(pt)
                # break if the length of any cluster_pts array is 0
                for arr in cluster_pts:
                    if len(arr) == 0:
                        quit = True
                if quit:
                    break
                # clusters become the average of their points
                is_dif = False
                for i in range(k):
                    # get the avg of each array in cluster_pts
                    x_sum = 0
                    y_sum = 0
                    num_pts = len(cluster_pts[i])
                    for pt in cluster_pts[i]:
                        x_sum += pt[0]
                        y_sum += pt[1]
                    x_avg = x_sum/num_pts
                    y_avg = y_sum/num_pts
                    # set the k'th center to be that avg
                    prev_x = centers[i][0]  # (x1, x2)
                    prev_y = centers[i][1]  # (x1, x2)
                    is_dif = is_dif or (x_avg != prev_x or y_avg != prev_y)
                    centers[i] = (x_avg, y_avg)
                # break if all updated clusters are the same
                if (not is_dif):
                    return centers
    centers = get_centers(x_train)
    # END OF WHILE-LOOP
    # Build the phi matrix
    phi = []
    for pt in x_train:
        row = [1]
        for mu in centers:
            row.append(lloyds_f(gamma, pt, mu))
        phi.append(row)
    w = get_weight(phi, y_train)

    # Deciding what to return 
    output = []
    if error_in: # In-sample error using the training set
        output.append(error_lloyds(w=w, X=x_train, Y=y_train, k=k, g=gamma, c=centers))
    if error_out: # Out-sample error using the testing set
        output.append(error_lloyds(w=w, X=x_test, Y=y_test, k=k, g=gamma, c=centers))

    return output


def answer(q13=False, q14=False, q15=False, q16=False, q17=False, q18=False):
    '''
    Allows user to dictate runtime by selectively flagging which questions should be answered.
    '''

    # Pull the data from the given files
    y_test, x_test = generate_data(num_points=100)
    y_train, x_train = generate_data(num_points=100)

    def question13():
        num_iterations = 10
        num_e_not_zero = 0
        for _ in range(num_iterations): # c is massive for hard-margin svm
            e_in = experiment(x_test, y_test, x_train, y_train, C=1e100, kernel='rbf', Q=2, gamma=1.5, error_in=True)[0]
            if (e_in != 0):
                num_e_not_zero += 1.0
        return num_e_not_zero/num_iterations

    def question14():
        nonlocal x_test, y_test, x_train, y_train
        num_iterations = 100
        num_kernel_beat_reg = 0
        for _ in range(num_iterations): 
            e_out_kernel = experiment(x_test, y_test, x_train, y_train, C=1e100, kernel='rbf', Q=2, gamma=1.5, error_out=True)[0]
            e_out_reg = lloyds(x_test, y_test, x_train, y_train, k=9, gamma=1.5, error_out=True)[0]
            if (e_out_kernel < e_out_reg):
                num_kernel_beat_reg += 1.0
            y_test, x_test = generate_data(num_points=100)
            y_train, x_train = generate_data(num_points=100)
        return num_kernel_beat_reg/num_iterations

    def question15():
        nonlocal x_test, y_test, x_train, y_train
        num_iterations = 100
        num_kernel_beat_reg = 0
        for _ in range(num_iterations): 
            e_out_kernel = experiment(x_test, y_test, x_train, y_train, C=1e100, kernel='rbf', Q=2, gamma=1.5, error_out=True)[0]
            e_out_reg = lloyds(x_test, y_test, x_train, y_train, k=12, gamma=1.5, error_out=True)[0]
            if (e_out_kernel < e_out_reg):
                num_kernel_beat_reg += 1.0
            y_test, x_test = generate_data(num_points=100)
            y_train, x_train = generate_data(num_points=100)
        return num_kernel_beat_reg/num_iterations

    def question16():
        nonlocal x_test, y_test, x_train, y_train
        num_iterations = 100
        e9_in = 0
        e9_out = 0
        e12_in = 0
        e12_out = 0
        for _ in range(num_iterations): 
            e9 = lloyds(x_test, y_test, x_train, y_train, k=9, gamma=1.5, error_in=True, error_out=True)
            e9_in += e9[0]
            e9_out += e9[1]
            e12 = lloyds(x_test, y_test, x_train, y_train, k=12, gamma=1.5, error_in=True, error_out=True)
            e12_in += e12[0]
            e12_out += e12[1]
            y_test, x_test = generate_data(num_points=100)
            y_train, x_train = generate_data(num_points=100)
        return ((e9_in/num_iterations, e9_out/num_iterations),(e12_in/num_iterations, e12_out/num_iterations))

    def question17():
        nonlocal x_test, y_test, x_train, y_train
        num_iterations = 100
        e1_in = 0
        e1_out = 0
        e2_in = 0
        e2_out = 0
        for _ in range(num_iterations): 
            e1 = lloyds(x_test, y_test, x_train, y_train, k=9, gamma=1.5, error_in=True, error_out=True)
            e1_in += e1[0]
            e1_out += e1[1]
            e2 = lloyds(x_test, y_test, x_train, y_train, k=9, gamma=2, error_in=True, error_out=True)
            e2_in += e2[0]
            e2_out += e2[1]
            y_test, x_test = generate_data(num_points=100)
            y_train, x_train = generate_data(num_points=100)
        return ((e1_in/num_iterations, e1_out/num_iterations),(e2_in/num_iterations, e2_out/num_iterations))

    def question18():
        num_iterations = 10
        num_e_zero = 0
        for _ in range(num_iterations): # c is massive for hard-margin svm
            e_in = lloyds(x_test, y_test, x_train, y_train, k=9, gamma=1.5, error_in=True)[0]
            if (e_in == 0):
                num_e_zero += 1.0
        return num_e_zero/num_iterations

    if q13:
        print("Question 13:\t", question13() )

    if q14:
        print("Question 14:\t", question14() )

    if q15:
        print("Question 15:\t", question15() )

    if q16:
        print("Question 16:\t", question16() )

    if q17:
        print("Question 17:\t", question17() )

    if q18:
        print("Question 18:\t", question18() )

# Answer questions by selectively flagging which ones should be answered.
answer(q13=True, q14=True, q15=True, q16=True, q17=True, q18=True)

# EXAMPLE OUTPUT
# Question 13:     0.0
# Question 14:     1.0
# Question 15:     1.0
# Question 16:     ((0.5264000000000002, 0.5172000000000001), (0.44690000000000013, 0.4581000000000001))
# Question 17:     ((0.48409999999999975, 0.4700000000000002), (0.5097999999999999, 0.5143000000000001))
# Question 18:     0.0

