import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("sales.csv")

X = data[["TV", "Radio"]]
y = data["Sales"]

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[150, 30]])

print("Predicted Sales:", prediction[0])