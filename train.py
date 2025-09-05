import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("diabetes.csv")

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "CART": DecisionTreeClassifier(criterion="gini", random_state=42)
}

# Train & evaluate
best_model, best_score, best_scaler = None, 0, None
for name, model in models.items():
    if name == "KNN":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_score:
        best_model = model
        best_scaler = scaler if name == "KNN" else None
        best_name = name
        best_score = acc

# Save model
with open("trained_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "scaler": best_scaler, "model_name": best_name}, f)

print(f"Saved model: {best_name} with accuracy: {best_score:.4f}")
