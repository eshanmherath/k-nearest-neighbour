import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

training = pd.read_csv('data\iris\iris_training.csv')
test = pd.read_csv('data\iris\iris_test.csv')
correct_predictions = 0
k = 5  

knn = KNeighborsClassifier(n_neighbors=k)
X, y = training.iloc[:, :-1], training.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
knn.fit(X, y)
predictions = knn.predict(X_test)

print('Accuracy ' + str(accuracy_score(y_test.as_matrix(), predictions)) + '%')
