import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values #independant vars
y = dataset.iloc[:, -1].values #dependant vars

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_guess = regressor.predict(x_test)

print(y_test)
print(y_guess)
print(regressor.score(x_test, y_test))
print(regressor.intercept_)
print(regressor.coef_)
print(regressor.predict([[12]]))

pyplot.scatter(x_train, y_train, color='red')
pyplot.plot(x_train, regressor.predict(x_train), color='blue')
pyplot.title('Salary vs Experience (Training Set)')
pyplot.xlabel('Years of Experience')
pyplot.ylabel('Salary')
pyplot.show()

# pyplot.scatter(x_test, y_test, color='red')
# pyplot.plot(x_test, regressor.predict(x_test), color='blue')
# pyplot.title('Salary vs Experience (Test Set)')
# pyplot.xlabel('Years of Experience')
# pyplot.ylabel('Salary')
# pyplot.show()