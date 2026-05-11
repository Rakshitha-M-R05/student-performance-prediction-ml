import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'student_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Load dataset
data = pd.read_csv(DATA_PATH)

# Use a simple feature set for the web form and model comparison
FEATURES = [
    'StudyTimeWeekly',
    'Absences',
    'Tutoring'
]
TARGET = 'GPA'

X = data[FEATURES]
y = data[TARGET]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features for the models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
}

best_model = None
best_score = float('-inf')
best_predictions = None
linear_predictions = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    score = model.score(X_test_scaled, y_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f'{name} Score: {score:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}')

    if name == 'Linear Regression':
        linear_predictions = predictions

    if score > best_score:
        best_score = score
        best_model = model
        best_predictions = predictions

# Save the best model and scaler
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f'Best model saved: {best_model.__class__.__name__} with score {best_score:.4f}')

# Plot actual vs predicted for the best model
if best_predictions is not None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_predictions, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual GPA')
    plt.ylabel('Predicted GPA')
    plt.title(f'Actual vs Predicted GPA ({best_model.__class__.__name__})')
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, 'actual_vs_predicted.png')
    plt.savefig(plot_path)
    plt.show()
    print(f'Visualization saved to: {plot_path}')