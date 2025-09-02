import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RANGE = 20

df = pd.DataFrame({
    'hours_studied': np.arange(1, RANGE+1),
    'marks_scored': [71, 73, 75, 77, 79, 82, 84, 86, 88, 89, 90, 91, 92, 93, 94, 95, 95, 96, 96, 97]
})
df.to_csv('data.csv', index=False)
print(df)

df_copy = df.copy()

plt.figure(figsize=(6,4))
plt.scatter(df_copy['hours_studied'],df_copy['marks_scored'])
plt.xlabel('hours_studied')
plt.ylabel('marks_scored')
plt.title('Hours vs Marks')
plt.show()

x,y = df_copy['hours_studied'],df_copy['marks_scored']
m,c = np.polyfit(x,y,1)
print(f"Slope: {m}\nIntercept: {c}")

plt.figure(figsize=(6,4))
plt.scatter(df_copy['hours_studied'],df_copy['marks_scored'])
plt.plot(x,m*x+c,color='green',linestyle='--')
plt.xlabel('hours_studied')
plt.ylabel('marks_scored')
plt.title('Hours vs Marks (Raw Data)')
plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df_copy[['hours_studied']],df_copy['marks_scored'])

m1, c1 = model.coef_ , model.intercept_
print("Slope:",m1[0],"\nIntercept:",c1)

plt.figure(figsize=(6,4))
plt.scatter(df_copy['hours_studied'],df_copy['marks_scored'])
plt.plot(x,m1*x+c1,color='red',linestyle='--')
plt.xlabel('hours_studied')
plt.ylabel('marks_scored')
plt.title('Hours vs Marks (Predicted Data)')
plt.show()

test= pd.DataFrame({
    'hours_studied':[RANGE+1, RANGE+2]
})
print(test)

predict = model.predict(test)
test['marks_scored']=predict
print(test)

print("MSE --> ",np.mean((model.predict(df_copy[['hours_studied']])-df_copy['marks_scored'])**2))

new_df= pd.concat([df,test],ignore_index=True)
print(new_df)

model1 = LinearRegression()
model1.fit(new_df[['hours_studied']],new_df['marks_scored'])

m2, c2 = model1.coef_ , model1.intercept_
print("Slope:",m2[0],"\nIntercept:",c2)

x,y = new_df['hours_studied'],new_df['marks_scored']

plt.figure(figsize=(6,4))
plt.scatter(new_df['hours_studied'],new_df['marks_scored'])
plt.plot(x,m2*x+c2,color='red',linestyle='--')
plt.xlabel('hours_studied')
plt.ylabel('marks_scored')
plt.title('Hours vs Marks (New Data)')
plt.show()