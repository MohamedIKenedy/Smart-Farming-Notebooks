import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingClassifier,
                            GradientBoostingClassifier, AdaBoostClassifier)

# Set MLflow experiment
mlflow.set_experiment("crop_recommendation")

# Data loading and preprocessing
df = pd.read_csv("Crop_recommendation.csv")

# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Prepare features and target
X = df.drop(['label', 'label_enc'], axis=1)
y = df['label_enc']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scalers
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Scale features
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

X_train_scaled = standard_scaler.fit_transform(X_train_minmax)
X_test_scaled = standard_scaler.transform(X_test_minmax)

# Save scalers
pickle.dump(minmax_scaler, open('minmaxscaler.pkl', 'wb'))
pickle.dump(standard_scaler, open('standscaler.pkl', 'wb'))

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

# Train and evaluate models with MLflow tracking
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Log model parameters
        params = model.get_params()
        mlflow.log_params(params)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, name)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Generate and log classification report
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Generate and log feature importance plot if available
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title(f'Feature Importance - {name}')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            mlflow.log_artifact('feature_importance.png')
            plt.close()

# Train and save the best performing model (Random Forest)
with mlflow.start_run(run_name="final_random_forest"):
    rfc = RandomForestClassifier()
    rfc.fit(X_train_scaled, y_train)
    
    # Log parameters
    mlflow.log_params(rfc.get_params())
    
    # Make predictions and log metrics
    y_pred = rfc.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log the model
    mlflow.sklearn.log_model(rfc, "random_forest_model")
    
    # Save model locally
    pickle.dump(rfc, open('Crop_Recommend_model.pkl', 'wb'))

def recommendation(N, P, k, temperature, humidity, ph, rainfall):
    with mlflow.start_run(run_name="prediction"):
        features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
        transformed_features = minmax_scaler.transform(features)
        transformed_features = standard_scaler.transform(transformed_features)
        prediction = rfc.predict(transformed_features).reshape(1, -1)
        
        # Log input parameters
        input_params = {
            'N': N, 'P': P, 'K': k, 
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        mlflow.log_params(input_params)
        
        # Log prediction
        predicted_crop = le.inverse_transform(prediction)[0]
        mlflow.log_param("predicted_crop", predicted_crop)
        
        return prediction[0]
