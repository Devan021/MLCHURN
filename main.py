import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_scor e, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import numpy as np

# Load the dataset from Excel
excel_file = '/home/devan/Downloads/customer_churn_large_dataset.xlsx'  # Provide the path to your Excel file
data = pd.read_excel(excel_file)

# Data Preprocessing
# Encode categorical variables
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])
data['Location'] = encoder.fit_transform(data['Location'])

# Split the data into features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of columns to exclude from feature scaling
columns_to_exclude = ['CustomerID', 'Name']

# Exclude non-numeric columns from X_train and X_test
X_train = X_train.drop(columns=columns_to_exclude)
X_test = X_test.drop(columns=columns_to_exclude)

# Perform feature scaling on the remaining numerical columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Predict on the testing data
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")

# Model Saving (if needed)
joblib.dump(model, 'churn_prediction_model.pkl')

# Model Deployment (Loading and Using the Model)
loaded_model = joblib.load('churn_prediction_model.pkl')

# Example: Providing new customer data for prediction
new_customer_data = pd.DataFrame({
    'CustomerID': ['New_Customer'],
    'Name': ['New_Customer'],
    'Age': [45],
    'Gender': ['Female'],
    'Location': ['New York'],
    'Subscription_Length_Months': [6],
    'Monthly_Bill': [75.0],
    'Total_Usage_GB': [250]
})

# Ensure 'Unknown' is present in the encoder classes
encoder.classes_ = np.append(encoder.classes_, 'Unknown')

# Encoding categorical variables for new data
new_customer_data['Gender'] = encoder.transform(new_customer_data['Gender'].apply(lambda x: x if x in encoder.classes_ else 'Unknown'))
new_customer_data['Location'] = encoder.transform(new_customer_data['Location'].apply(lambda x: x if x in encoder.classes_ else 'Unknown'))

# Exclude non-numeric columns from new_customer_data
new_customer_data = new_customer_data.drop(columns=columns_to_exclude)

# Feature scaling for new data
new_customer_data_scaled = scaler.transform(new_customer_data)

# Making predictions for new data
new_customer_churn_prediction = loaded_model.predict(new_customer_data_scaled)

# Displaying the churn prediction for the new customer
if new_customer_churn_prediction[0] == 0:
    print("New customer is likely to stay.")
else:
    print("New customer is likely to churn.")
