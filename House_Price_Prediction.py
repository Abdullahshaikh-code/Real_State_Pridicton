
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


data=pd.read_csv("data.csv")
data


import matplotlib.pyplot as plt
import seaborn as sns


data.describe()


sns.pairplot(data)
plt.show()


data=data.dropna(axis=1,how="all")
data


sns.pairplot(data)
plt.show()


Xtrain=data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]


Ytrain=data['price']


X_train, X_test, y_train, y_test = train_test_split(Xtrain, Ytrain, test_size=0.45, random_state=42)



model = LinearRegression()

# Training the model
model.fit(X_train, y_train)


y_pred = model.predict(X_test)



score = model.score(X_test, y_test)
print("Model R^2 Score:", score)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)





