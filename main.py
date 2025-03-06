from Perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


wine_data = pd.read_csv("wine.data", header=0, encoding="utf-8")

y = wine_data.iloc[0:, 0].values
X = wine_data.iloc[0:, [1,2,3,4,6,8,9,10,11,12]].values
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train, test_size=0.35, random_state=42, stratify=y_train)

model = Perceptron(eta=0.01, n_iter=100, random_state=69)
model.fit_multiclass(X_train,y_train)

#<----------Validation Predictions---------------------->
validation_pred = model.predict_multiclass(X_validation)

confusion = confusion_matrix(y_validation, validation_pred)
classification = classification_report(y_validation, validation_pred)

print("Confusion Matrix (Validation):\n" )
print(confusion)
print("\nClassification Report(Validation):")  
print(classification)
#<----------------------------------------------------->

#<-----------Testing Set Predictions------------------->
test_pred = model.predict_multiclass(X_test)

confusion = confusion_matrix(y_test, test_pred)
classification = classification_report(y_test, test_pred)
print("Confusion Matrix (Test):\n" )
print(confusion)
print("\nClassification Report(Test):")  
print(classification)

#Graph updates over time for each class perceptron 
class_count = 0
colors = ["green", "red", "blue"]
for p in model.perceptrons_:
    plt.plot(range(1,len(p.errors_) + 1), p.errors_, marker="*", color=colors[class_count], label=f"Class {class_count+1}.",alpha=0.8)
    class_count += 1
    print(f"Class {class_count} Weights:")
    for w in p.w_:
        print(w, end=", ")       
    print(f"bias {p.b_}")

plt.xlabel("Epochs")
plt.ylabel("Updates")
plt.legend()
plt.show()

