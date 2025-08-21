"""## P1"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Path to dataset files:", path)

df= pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

print("Shape of the dataset:", df.shape)

print("Dataset information:")
print(df.info())

print(df.describe())

from sklearn.model_selection import train_test_split

X,y=df.drop("Outcome",axis=1),df["Outcome"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
acc=[]
bk,ba=1,0

for k in range(1,21):
  print("-"*90)
  print(f"Running for k={k}")
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train,y_train)
  print(f"Accuracy for k={k}: {knn.score(X_test,y_test):.6f}")

  cm = confusion_matrix(y_test,knn.predict(X_test))
  print(cm)

  tp, tn, fp, fn = cm.ravel()
  print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
  acc_val, rec, prec = (tp+tn)/(tp+tn+fp+fn), tp/(tp+fn), tp/(tp+fp)
  f1 = 2*rec*prec/(rec+prec)
  print(f"Accuracy: {acc_val:.6f}, Recall: {rec:.6f}, Precision: {prec:.6f}, F1 Score: {f1:.6f}")

  acc.append(knn.score(X_test,y_test))
  if knn.score(X_test,y_test)>ba:
    ba=knn.score(X_test,y_test)
    bk=k

print(f"\n\nBest k: {bk}, Accuracy: {ba:.6f}\n")

plt.figure(figsize=(5,4))
plt.scatter(bk,ba,color="red")
plt.plot(range(1,21),acc)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Classifier Accuracy")
plt.grid(alpha=0.5)
plt.show()

knn=KNeighborsClassifier(n_neighbors=bk)
knn.fit(X_train,y_train)

cm = confusion_matrix(y_test,knn.predict(X_test))
plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, knn.predict(X_test))
print("\n\nMSE:", mse)

"""## P2"""

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.DataFrame(
    np.array([
        ['A',160,50,0],
        ['B',165,55,0],
        ['C',170,65,1],
        ['D',175,70,1],
        ['E',180,80,1]
    ]),
    columns=['ID','Height(cm)','Weight(kg)','Class']
)

print(data)

def knn_predict(X, y, x, k):
  eucli_dist = np.sqrt(np.sum((X - x)**2, axis=1))
  sorted_dist_idx = np.argsort(eucli_dist)
  k_nearest_labels = y[sorted_dist_idx[:k]]
  unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
  predicted_label = unique_labels[np.argmax(label_counts)]
  return predicted_label

data['Height(cm)'] = pd.to_numeric(data['Height(cm)'])
data['Weight(kg)'] = pd.to_numeric(data['Weight(kg)'])

for k in range(1,5):
  print(f"For k={k}, Prediction --> {knn_predict(data[['Height(cm)','Weight(kg)']].values, data['Class'].values, np.array([172,66]), k)}")

