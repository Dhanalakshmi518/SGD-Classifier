# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data
2. Split Dataset into Training and Testing Sets
3. Train the Model Using Stochastic Gradient Descent (SGD)
4. Make Predictions and Evaluate Accuracy
5. Generate Confusion Matrix 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: m.p.dhanalakshmi
RegisterNumber:  212225040063
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
iris=load_iris() 
df=pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['target']=iris.target 
print(df.head())

X = df.drop('target',axis=1) 
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test,y_pred) 
print("Confusion Matrix:") 
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:
<img width="795" height="314" alt="Screenshot 2026-05-16 182729" src="https://github.com/user-attachments/assets/8f20f8fe-b9c6-4787-b33b-4a0ec76faf3a" />

<img width="162" height="34" alt="Screenshot 2026-05-16 183242" src="https://github.com/user-attachments/assets/24f7fe5c-dbbc-4488-a176-6a1e9e488b44" />

<img width="202" height="110" alt="Screenshot 2026-05-16 183250" src="https://github.com/user-attachments/assets/06d63291-bbdf-4443-b52b-07a5270e6227" />

<img width="644" height="511" alt="Screenshot 2026-05-16 182937" src="https://github.com/user-attachments/assets/a6625f77-2d59-4428-b64e-57df0b82bacb" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
