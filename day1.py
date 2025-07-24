
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.DataFrame(
    np.array([[183,433],[175,393],[134,270],[170,364],[144,346],[183,399],[167,360],[114,361],[125,319],[187,376]]),
    columns=['College Test','University Test']
)

df

# Linear Regression - Slope and Intercept Calculation
# Using the formula: y = mx + b, where m is the slope and b is the intercept
# Slope (m) = (n * Σ(xy) - Σx * Σy) / (n * Σ(x^2) - (Σx)^2)
# Intercept (b) = (Σy - m * Σx) / n
# where n is the number of data points, Σ denotes summation, x is the independent variable, and y is the dependent variable.
def linear_eqn(df):
    x=df.iloc[:,0]
    y=df.iloc[:,1]
    n=len(x)

    x_total = np.sum(x)
    y_total = np.sum(y)
    xy_total = np.sum(x*y)
    x2_total = np.sum(x**2)

    slope = (n*xy_total - x_total*y_total)/(n*x2_total - x_total**2)
    intercept = (y_total - slope*x_total)/n

    plt.figure(figsize=(8,5))
    plt.scatter(x,y)
    plt.grid()
    plt.plot(x,slope*x+intercept,color='red')
    plt.show()

    return slope,intercept

m, b= linear_eqn(df)
print("Slope:",m)
print("Intercept:",b)

# Linear Model Prediction
def linear_model(x):
    y=m*x+b
    return y

linear_model(150) # OUTPUT = 352.4497313171707

# Mean Squared Error (MSE) Calculation
# MSE = (1/n) * Σ(y_i - ŷ_i)^2,
# where y_i is the actual value, ŷ_i is the predicted value, and n is the number of data points.
def mse():
  predictions = [linear_model(x) for x in df.iloc[:,0]]
  print("Predictions:",predictions)
  print("Actual Values:",df.iloc[:,1].values)
  mse = np.mean((predictions - df.iloc[:,1])**2)
  return mse

print("MSE:",mse()) #  OUTPUT = 927.8656804548871

# Visualizing the Linear Regression Model
plt.figure(figsize=(8,5))
plt.scatter(df.iloc[:,0],df.iloc[:,1])
plt.scatter(150,linear_model(150),color='green')
plt.xlabel('College Test')
plt.ylabel('University Test')
plt.grid()
plt.plot(df.iloc[:,0],m*df.iloc[:,0]+b,color='red')
plt.show()