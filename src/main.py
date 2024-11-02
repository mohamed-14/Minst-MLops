from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

digits = load_digits()
X = digits.data
y = digits.target 
model=RandomForestClassifier()
model.fit(X,y)
y_pred = model.predict(X)
accuracy=accuracy_score(y,y_pred)
print(f"Accuracy: {accuracy}")
cm=confusion_matrix(y,y_pred)
print(f"Confusion Matrix:\n{cm}")