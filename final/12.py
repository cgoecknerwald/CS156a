from sklearn import svm

X = [(1,0), (0,1), (0,-1), (-1,0), (0,2), (0,-2), (-2,0)]
Y = [-1, -1, -1, 1, 1, 1, 1]

machine = svm.SVC(C=1e100, kernel="poly", degree=2, coef0=1)
machine.fit(X=X, y=Y)

print(sum(list(machine.n_support_)))