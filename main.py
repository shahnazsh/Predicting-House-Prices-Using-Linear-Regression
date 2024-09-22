# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
url = 'https://raw.githubusercontent.com/your-repo/house-prices-dataset/main/train.csv'  # Update with your dataset URL or path
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Fill missing values
data.fillna(method='ffill', inplace=True)

# Feature Selection
features = ['OverallQual', 'GrLivArea', 'GarageCars']
X = data[features]
y = data['SalePrice']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'R-squared: {r2:.2f}')
print(f'Mean Absolute Error (MAE): ${mae:,.2f}')

# Plotting Actual vs Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.show()

# Feature Importance
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coefficients)

# Saving model
import joblib
joblib.dump(model, 'linear_regression_model.pkl')
