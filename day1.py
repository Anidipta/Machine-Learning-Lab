import numpy as np
import pandas as pd

def nb_classifier(x, df, target_index):
  n=len(df)
  t=list(set(df.iloc[:,target_index]))
  print("Classes: ",t)

  prior=[]
  for i in t:
    prior.append(len(df[df.iloc[:,target_index]==i])/n)
  print("Prior: ",prior)

  likelihood=[]
  for i in range(len(x)):
    likelihood.append([])
    for j in t:
      num=len(df[(df.iloc[:,i]==x[i]) & (df.iloc[:,target_index]==j)])
      den=len(df[df.iloc[:,target_index]==j])
      likelihood[-1].append(num/den)
  print("Likelihood: ",likelihood)
  
  posterior=[]
  for i in range(len(t)):
    post=prior[i]
    for j in range(len(x)):
      post=post*likelihood[j][i]
    posterior.append(post)
  print("Posterior: ",posterior)
  
  return t[np.argmax(posterior)]

df=pd.DataFrame(
    np.array([[0,0,0],[0,1,0],
              [1,0,0],[1,1,1]]),
    columns=['X','Y','Output']
  )
print(df)

#    X  Y  Output
# 0  0  0       0
# 1  0  1       0
# 2  1  0       0
# 3  1  1       1


nb_classifier([0,0],df,2)
# Classes:  [0, 1]
# Prior:  [0.75, 0.25]
# Likelihood:  [[0.6666666666666666, 0.0], [0.6666666666666666, 0.0]]
# Posterior:  [0.3333333333333333, 0.0]
# 0

nb_classifier([0,1],df,2)
# Classes:  [0, 1]
# Prior:  [0.75, 0.25]
# Likelihood:  [[0.6666666666666666, 0.0], [0.3333333333333333, 1.0]]
# Posterior:  [0.16666666666666666, 0.0]
# 0

nb_classifier([1,0],df,2)
# Classes:  [0, 1]
# Prior:  [0.75, 0.25]
# Likelihood:  [[0.3333333333333333, 1.0], [0.6666666666666666, 0.0]]
# Posterior:  [0.16666666666666666, 0.0]
# 0

nb_classifier([1,1],df,2)
# Classes:  [0, 1]
# Prior:  [0.75, 0.25]
# Likelihood:  [[0.3333333333333333, 1.0], [0.3333333333333333, 1.0]]
# Posterior:  [0.08333333333333333, 0.25]
# 1
