from sys import argv
from sys import modules
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *
from numpy import sum


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def run_knn(points):
    m = KNN(5)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score)


"""
creates 1-NN classifier and returns its accuracy
"""
def ques_one(points):
    k = KNN(1)
    k.train(points)
    real = [0]*len(points)
    predicted = [0]*len(points)
    for i in range(len(points)):
        real[i] = points[i].label
        predicted[i] = k.predict(points[i])[0]
    print("question 1 answer: ",accuracy_score(real, predicted))


"""
creates KNN classifiers for 1<=k<=30.
for each one- run the cross-validation algorithm
and reports their accuracy
"""
def ques_two(points):
    max_accuracy = 0
    best_k = 0
    for k in range(1,31):
        m = KNN(k)
        m.train(points)
        cv = CrossValidation()
        # print("current k=", k ,"  ", end="")
        a = cv.run_cv(points, len(points), m, accuracy_score, False)
        if max_accuracy < a:
            max_accuracy = a
            best_k = k
    return best_k


"""
run the cross-validation for the best K,
using 2,10,20 folds, and print the accuracy parameters
"""
def ques_three(points):
    print("Question 3:")
    # best_k = ques_two(points)
    best_k = 19
    print("K={}".format(best_k))
    m = KNN(best_k)
    m.train(points)
    cv = CrossValidation()
    print("2-fold-cross-validation:")
    cv.run_cv(points, 2, m, accuracy_score,False,True)
    print("10-fold-cross-validation:")
    cv.run_cv(points, 10, m, accuracy_score,False,True)
    print("20-fold-cross-validation:")
    cv.run_cv(points, 20, m, accuracy_score,False,True)


"""
creates 5-NN and 7-NN and runs the 2-folds-cross-validation
using 4 ways to normalize the data, and print the accuracy for every case
"""
def ques_four(points):
    print("Question 4:")
    list_of_k = [5,7]
    for i in list_of_k:
        print("K={}".format(i))
        m = KNN(i)
        m.train(points)

        cv = CrossValidation()
        a = cv.run_cv(points, 2, m, accuracy_score,False,True)
        print("Accuracy of DummyNormalizer is", a)
        print()

        new_p = SumNormalizer()
        new_p.fit(points)
        new_points = new_p.transform(points)
        cv = CrossValidation()
        a = cv.run_cv(new_points, 2, m, accuracy_score,False,True)
        print("Accuracy of SumNormalizer is", a)
        print()

        new_p = MinMaxNormalizer()
        new_p.fit(points)
        new_points = new_p.transform(points)
        cv = CrossValidation()
        a = cv.run_cv(new_points, 2, m, accuracy_score,False,True)
        print("Accuracy of MinMaxNormalizer is", a)
        print()

        new_p = ZNormalizer()
        new_p.fit(points)
        new_points = new_p.transform(points)
        cv = CrossValidation()
        a = cv.run_cv(new_points, 2, m, accuracy_score,False,True)
        print("Accuracy of ZNormalizer is", a)
        if i==5: print()


if __name__ == '__main__':
    loaded_points = load_data()
    # run_knn(loaded_points)
    # ques_one(loaded_points)
    # ques_two(loaded_points)
    ques_three(loaded_points)
    ques_four(loaded_points)
