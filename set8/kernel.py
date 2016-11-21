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
    # In order to avoid running functions twice
    answer2 = None
    answer3 = None
    # Pull the data from the given files
    y_test, x_test = separate_data(get_data("features_test.txt"))
    y_train, x_train = separate_data(get_data("features_train.txt"))

    def question2():
        nonlocal y_test, x_test, y_train, x_train, answer2
        question2 = []
        for num in range(0, 10, 2):
            question2.append((experiment(x_test, y_test, x_train, y_train, C=0.01, kernel='poly', Q=2, gamma=1.0, coef0=1.0, num1=num, error_in=True), num))
        answer2 = max(question2, key=lambda x: x[0])[1]
        return answer2

    def question3():
        nonlocal y_test, x_test, y_train, x_train, answer3
        question3 = []
        for num in range(1, 10, 2):
            question3.append((experiment(x_test, y_test, x_train, y_train, C=0.01, kernel='poly', Q=2, gamma=1.0, coef0=1.0, num1=num, error_in=True), num))
        answer3 = min(question3, key=lambda x: x[0])[1]
        return answer3

    def question4():
        nonlocal y_test, x_test, y_train, x_train, answer2, answer3

        # In order to avoid running question2() and question3() twice
        if answer2 is None:
            answer2 = question2()
        if answer3 is None:
            answer3 = question3()
        sv = []
        for num in (answer2, answer3):
            sv.append(experiment(x_test, y_test, x_train, y_train, num1=num, num_sv=True)[0])
        return abs(sv[0] - sv[1])

    def question5():
        nonlocal y_test, x_test, y_train, x_train
        string = ""
        for c in (0.001, 0.01, 0.1, 1):
            result = experiment(x_test, y_test, x_train, y_train, C=c, kernel='poly', Q=2, gamma=1.0, coef0=1.0, num1=1, num2=5, error_in=True, error_out=True, num_sv=True)
            string += "C: %.3f" % c + "\t[e_in, e_out, num_sv]: " + str(result) + "\n"
        return string[:-1] # clip the new-line 

    def question6():
        nonlocal y_test, x_test, y_train, x_train
        string = ""
        for c in (0.0001, 0.001, 0.01, 1):
            for q in (2, 5):
                result = experiment(x_test, y_test, x_train, y_train, C=c, kernel='poly', Q=q, gamma=1.0, coef0=1.0, num1=1, num2=5, error_in=True, error_out=True, num_sv=True)
                string += "C: %.4f, " % c + "Q: %.1f" % q + "\t[e_in, e_out, num_sv]: " + str(result) + "\n"
        return string[:-1] # clip the new-line 

    def question7_8():
        nonlocal y_test, x_test, y_train, x_train
        dict_min_c = defaultdict(list) # Default values are an empty list.
        # 100 runs
        for _ in range(100):
            question7_8 = []
            # Per C, determine the C with the minimum E_cv
            for c in (0.0001, 0.001, 0.01, 0.1, 1):
                err = experiment(x_test, y_test, x_train, y_train, C=c, Q=2, num1=1, num2=5, error_cv=True)[0]
                question7_8.append((err, c))
            c_min_err = min(question7_8) # (Error_cv, C) sorted by error_cv and then by c
            dict_min_c[c_min_err[1]].append(c_min_err[0]) # Use a counter to keep track of the C with the minimum E_cv
            # shuffle the training data between runs
            data = list(zip(x_train, y_train))
            random.shuffle(data)
            x_train, y_train = zip(*data)
        # print(list(map(lambda x: (x[0], len(x[1])), dict_min_c.items())))
        best_c = max(dict_min_c.items(), key=lambda x: len(x[1]))[0] # Return the (C, number of counts)
        err_cv = dict_min_c[best_c] # Array of E_cv
        return "C: " + str(best_c) + ", avg E_cv: " + str(sum(err_cv)/len(err_cv)) # Return the (C, number of counts)

    def question9_10():
        string = ""
        # Per C, determine the C with the minimum E_cv
        for c in (0.01, 1, 100, 10**4, 10**6):
            err = experiment(x_test, y_test, x_train, y_train, C=c, kernel='rbf', gamma=1, num1=1, num2=5, error_in=True, error_out=True)
            string += "C: %11.2f, " % c + "\t[e_in, e_out]: " + str(err) + "\n"
        return string


    # Answer and print the questions that are flagged.
    if q2:
        print("Question 2: " + str(question2()))
    if q3:
        print("Question 3: " + str(question3()))
    if q4:
        print("Question 4: " + str(question4()))
    if q5:
        print("Question 5:\n" + str(question5()))
    if q6:
        print("Question 6:\n" + str(question6()))
    if q7 or q8:
        print("Question 7&8:\n" + str(question7_8()))
    if q9 or q10:
        print("Question 9&10:\n" + str(question9_10()))

# Answer questions by selectively flagging which ones should be answered.
answer(q7=True, q8=True)

# EXAMPLE OUTPUT FOR ALL QUESTIONS FLAGGED.
# Question 2: 0
# Question 3: 1
# Question 4: 1793
# Question 5:
# C: 0.001  [e_in, e_out, num_sv]: [0.004484304932735439, 0.01650943396226412, 76]
# C: 0.010  [e_in, e_out, num_sv]: [0.004484304932735439, 0.018867924528301883, 34]
# C: 0.100  [e_in, e_out, num_sv]: [0.004484304932735439, 0.018867924528301883, 24]
# C: 1.000  [e_in, e_out, num_sv]: [0.0032030749519538215, 0.018867924528301883, 24]
# Question 6:
# C: 0.0001, Q: 2.0 [e_in, e_out, num_sv]: [0.0089686098654708779, 0.01650943396226412, 236]
# C: 0.0001, Q: 5.0 [e_in, e_out, num_sv]: [0.004484304932735439, 0.018867924528301883, 26]
# C: 0.0010, Q: 2.0 [e_in, e_out, num_sv]: [0.004484304932735439, 0.01650943396226412, 76]
# C: 0.0010, Q: 5.0 [e_in, e_out, num_sv]: [0.004484304932735439, 0.021226415094339646, 25]
# C: 0.0100, Q: 2.0 [e_in, e_out, num_sv]: [0.004484304932735439, 0.018867924528301883, 34]
# C: 0.0100, Q: 5.0 [e_in, e_out, num_sv]: [0.0038436899423446302, 0.021226415094339646, 23]
# C: 1.0000, Q: 2.0 [e_in, e_out, num_sv]: [0.0032030749519538215, 0.018867924528301883, 24]
# C: 1.0000, Q: 5.0 [e_in, e_out, num_sv]: [0.0032030749519538215, 0.021226415094339646, 21]
# Question 7&8:
# C: 0.001, avg E_cv: 0.00452152014652
# Question 9&10:
# C:        0.01,   [e_in, e_out]: [0.0038436899423446302, 0.023584905660377409]
# C:        1.00,   [e_in, e_out]: [0.004484304932735439, 0.021226415094339646]
# C:      100.00,   [e_in, e_out]: [0.0032030749519538215, 0.018867924528301883]
# C:    10000.00,   [e_in, e_out]: [0.0025624599615631238, 0.023584905660377409]
# C:  1000000.00,   [e_in, e_out]: [0.00064061499039080871, 0.023584905660377409]
