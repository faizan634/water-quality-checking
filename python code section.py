import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Specify the file path
file_path = r"E:\\journey to python\water quality testing\\Water Quality Testing.csv"

# Load the data
data = load_data(file_path)

# Display the first few rows
print(data.head())
# Preprocess the data
def preprocess_data(df):
    # Select features and target variable
    features = ['pH', 'Temperature (°C)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)']
    target = 'Conductivity (µS/cm)'
    
    X = df[features]
    y = df[target]# Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    # Train the model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
# Make predictions
def predict_conductivity(model, scaler, new_data):
    # Ensure new_data has the same features as training data
    features = ['pH', 'Temperature (°C)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)']
     # Validate input data
    if not isinstance(new_data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if set(new_data.columns) != set(features):
        raise ValueError(f"Input must have columns: {features}")
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
     # Make predictions
    predictions = model.predict(new_data_scaled)
    
    return predictions
# Main execution
def main():
    # Load the data
    file_path = 'Water Quality Testing.csv'
    df = load_data(file_path)
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test_scaled, y_test)
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Example of making predictions
    # Create a sample new data point (replace with actual new measurements)
    new_data = pd.DataFrame({
        'pH': [7.2],
        'Temperature (°C)': [22.5],
        'Turbidity (NTU)': [4.0],
        'Dissolved Oxygen (mg/L)': [8.5]
    })# Make prediction
    prediction = predict_conductivity(model, scaler, new_data)
    print("\nPrediction Example:")
    print(f"Predicted Conductivity: {prediction[0]:.2f} µS/cm")
    # Feature importance
def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importances:")
    for f in range(len(feature_names)):
        print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

# Uncomment to run the main function
if __name__ == "__main__":
    main()