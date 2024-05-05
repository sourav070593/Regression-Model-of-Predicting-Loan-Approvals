import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data (replace 'loan_data.csv' with your actual file path)
data = pd.read_csv("loan_data.csv")

# Preprocess data (handle missing values, categorical variables, etc.)
# ... (Your data preprocessing )

# Feature selection (choose relevant features for prediction)
features = ["ApplicantIncome", "CreditHistory", "LoanAmount"]  # Example features
X = data[features]
y = data["Loan_Status"]  # Target variable (converted to binary: approved/rejected)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred > 0.5)  # Threshold for approval probability
confusion_matrix = confusion_matrix(y_test, y_pred > 0.5)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix)

# Feature importance analysis (optional)
# ... (Use libraries like scikit-learn to analyze feature coefficients)

# Prediction on new data point (example)
new_data = {"ApplicantIncome": 50000, "CreditHistory": 1, "LoanAmount": 100000}
new_data_df = pd.DataFrame([new_data])
predicted_approval = model.predict(new_data_df) > 0.5

print("Loan approval prediction for new data:", predicted_approval[0])

