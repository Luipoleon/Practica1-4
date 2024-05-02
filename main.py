import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt


# Ignore ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


data = pd.read_csv('irisbin.csv', header=None)
data = data.replace(-1, 0)

# Proyección en dos dimensiones de la distribución de clases para el dataset Iris


X = data.iloc[:, :-3]  # Input features
y = data.iloc[:, -3:]  # Target labels

# Seleccionar las clases
setosa = data[data[4] == 1]
versicolor = data[data[5] == 1]
virginica = data[data[6] == 1]

# Graficar
plt.scatter(setosa[0], setosa[1], color='red', label='Setosa')
plt.scatter(versicolor[0], versicolor[1], color='blue', label='Versicolor')
plt.scatter(virginica[0], virginica[1], color='green', label='Virginica')
plt.xlabel('Longitud del sépalo')
plt.ylabel('Anchura del sépalo')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='relu', solver='adam', max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Graficar la proyección de las clases del resultado de la clasificación
y_pred = model.predict(X_test)
setosa = X_test[y_pred[:, 0] == 1]
versicolor = X_test[y_pred[:, 1] == 1]
virginica = X_test[y_pred[:, 2] == 1]

plt.scatter(setosa[0], setosa[1], color='red', label='Setosa')
plt.scatter(versicolor[0], versicolor[1], color='blue', label='Versicolor')
plt.scatter(virginica[0], virginica[1], color='green', label='Virginica')
plt.xlabel('Longitud del sépalo')
plt.ylabel('Anchura del sépalo')
plt.legend()
plt.show()


# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Precisión:", accuracy)



# Step 6: Leave-k-out validation
k = 5  # Number of splits
kf = KFold(n_splits=k)
errors = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    error = 1 - model.score(X_test, y_test)
    errors.append(error)

expected_error = np.mean(errors)
average = np.mean(errors)
std_deviation = np.std(errors)

print("\nLeave-k-out validación")
print("Error esperado:", expected_error)
print("Promedio:", average)
print("Desviación estandar:", std_deviation)

#Leave-one-out validation
loo = LeaveOneOut()
errors = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    error = 1 - model.score(X_test, y_test)
    errors.append(error)

expected_error = np.mean(errors)
average = np.mean(errors)
std_deviation = np.std(errors)

print("\nLeave-one-out validación")
print("Error esperado:", expected_error)
print("Promedio:", average)
print("Desviación estandar:", std_deviation)

