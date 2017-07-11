__author__ = 'eshan'

from math import sqrt
import operator
import pandas as pd


def euclidean_distance(example1, example2):
    return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(example1, example2)))


def get_prediction(test_instance):
    global training
    global k
    distances = []

    for _, r in training.iterrows():
        distance = euclidean_distance(r[:-1].tolist(), test_instance.tolist())
        neighbour = r[-1]
        distances.append((distance, neighbour))

    distances.sort(key=operator.itemgetter(0))

    candidates = [v for _, v in distances[:k]]
    return max(candidates, key=candidates.count)


training = pd.read_csv('iris_training.csv')
test = pd.read_csv('iris_test.csv')
correct_predictions = 0
k = 9                                             # number of neighbours to be considered in decision making

for index, row in test.iterrows():
    prediction = get_prediction(row[:-1])
    if prediction == row[-1]:
        correct_predictions += 1


print('Correct predictions : ' + str(correct_predictions))
print('Total   predictions : ' + str(test.shape[0]))
print('Accuracy ' + str(correct_predictions*100.0/test.shape[0]) + '%')


