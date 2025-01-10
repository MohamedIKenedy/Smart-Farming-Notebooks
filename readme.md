# Crop Recommendation System

A machine learning system that recommends suitable crops based on soil and environmental parameters using multiple classification algorithms. The system leverages MLflow for experiment tracking, model versioning, and deployment management.

## Overview

This project implements a comprehensive machine learning pipeline for crop recommendation using various classification algorithms. It features extensive experimentation tracking using MLflow, automated model comparison, and a production-ready prediction system.

## Installation and Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd crop-recommendation-system
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install numpy pandas scikit-learn seaborn matplotlib mlflow
```

## Execution Instructions

1. **Prepare your data**
   - Ensure your `Crop_recommendation.csv` file is in the project root directory
   - The CSV should contain columns: N, P, K, temperature, humidity, ph, rainfall, and label

2. **Run the training script**
```bash
python train_model.py
```

3. **Launch MLflow UI**
```bash
mlflow ui --port 5000
```
   - Access the MLflow dashboard at `http://localhost:5000`
   - View experiments under the "crop_recommendation" experiment
   - Compare model performances
   - Access logged artifacts and parameters

4. **Track Experiments in MLflow**
   - Navigate to the experiment named "crop_recommendation"
   - View different runs for each model
   - Compare metrics across models
   - Access model artifacts and parameters
   - Download serialized models and preprocessors

### MLflow Dashboard Navigation

1. **Experiments View**
   - Lists all training runs
   - Shows accuracy metrics
   - Displays run status and timestamp

2. **Run Details**
   - Model parameters
   - Performance metrics
   - Feature importance plots
   - Classification reports

3. **Artifact Management**
   - Access saved models
   - View preprocessors
   - Download artifacts
   - Compare versions

4. **Model Registry**
   - Register best performing models
   - Track model versions
   - Manage deployment stages

## Features

## Features

- Multiple classifier implementations including:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Bagging
  - AdaBoost
  - Gradient Boosting
  - Extra Trees

- Automated MLflow experiment tracking for each model
- Feature importance visualization
- Model performance comparison
- Scalable prediction pipeline
- Serialized model and preprocessing components

## Technical Architecture

### Data Processing Pipeline

1. **Data Loading**: Loads crop recommendation dataset from CSV
2. **Label Encoding**: Converts crop labels to numerical format
3. **Feature Scaling**: Two-step scaling process
   - MinMax scaling
   - Standard scaling
4. **Train-Test Split**: 80-20 split with random state 42

### MLflow Integration

The project extensively uses MLflow for experiment tracking:

1. **Experiment Organization**
   - Each model runs in a separate MLflow run
   - Automatic parameter logging
   - Performance metrics tracking
   - Model artifact storage

2. **Tracked Components**
   - Model parameters
   - Training accuracy
   - Classification reports
   - Feature importance plots (where applicable)
   - Trained model artifacts

3. **Model Registry**
   - Best performing model (Random Forest) saved to MLflow registry
   - Version control for model iterations
   - Deployment metadata tracking

## Input Features

The system uses seven key parameters for crop recommendation:
- N: Nitrogen content in soil
- P: Phosphorous content in soil
- K: Potassium content in soil
- Temperature: Ambient temperature
- Humidity: Relative humidity
- pH: Soil pH
- Rainfall: Annual rainfall

## Model Artifacts

The following artifacts are generated and saved:

1. **Preprocessors**
   - `label_encoder.pkl`: For crop label encoding
   - `minmaxscaler.pkl`: For initial feature scaling
   - `standscaler.pkl`: For secondary feature scaling

2. **Models**
   - `Crop_Recommend_model.pkl`: Production Random Forest model
   - MLflow logged models for each classifier

3. **Visualization**
   - Feature importance plots for applicable models
   - Classification reports



## Project Structure

```
crop-recommendation-system/
│
├── train_model.py           # Main training script
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
├── Crop_recommendation.csv # Dataset
│
├── mlruns/                 # MLflow tracking directory
│   └── 0/                 # Experiment ID
│       └── runs/          # Individual run data
│
└── models/                 # Saved model artifacts
    ├── label_encoder.pkl
    ├── minmaxscaler.pkl
    ├── standscaler.pkl
    └── Crop_Recommend_model.pkl
```

## MLflow Commands Reference

### Basic Commands

```bash
# Start MLflow UI
mlflow ui --port 5000

# List experiments
mlflow experiments list

# Create new experiment
mlflow experiments create --experiment-name "new_experiment"

# Delete experiment
mlflow experiments delete --experiment-id <id>
```

### Advanced Usage

```bash
# Set tracking URI (if using remote server)
mlflow set-tracking-uri "http://your-server:5000"

# Run tracking with specific experiment
MLFLOW_EXPERIMENT_NAME="crop_recommendation" python train_model.py

# Export run artifacts
mlflow artifacts download -u "runs:/run_id/artifacts"
```

### Environment Variables

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Set experiment name
export MLFLOW_EXPERIMENT_NAME="crop_recommendation"

# Set tracking token (if using remote server)
export MLFLOW_TRACKING_TOKEN="your-token"
```

## Troubleshooting

Common issues and solutions:

1. **MLflow UI not starting**
   - Check if port 5000 is already in use
   - Try different port: `mlflow ui --port 5001`
   - Ensure MLflow is installed correctly

2. **Artifact logging failures**
   - Verify write permissions in the mlruns directory
   - Check available disk space
   - Ensure proper connection if using remote storage

3. **Model loading issues**
   - Confirm all required dependencies are installed
   - Check Python version compatibility
   - Verify model file paths

## MLflow Best Practices Implemented

1. **Experiment Organization**
   - Logical experiment naming
   - Consistent run naming
   - Hierarchical parameter tracking

2. **Artifact Management**
   - Systematic artifact logging
   - Version control for models
   - Preprocessor persistence

3. **Metric Tracking**
   - Performance metric logging
   - Classification report generation
   - Feature importance visualization

4. **Production Deployment**
   - Model serialization
   - Preprocessing pipeline preservation
   - Prediction function encapsulation

## ML Development Cycle

The project follows a structured ML development cycle:

1. **Data Preparation**
   - Data loading
   - Feature preprocessing
   - Train-test splitting

2. **Model Development**
   - Multiple model implementations
   - Hyperparameter specification
   - Training pipeline creation

3. **Experimentation**
   - Automated model training
   - Performance comparison
   - Feature importance analysis

4. **Model Selection**
   - Performance evaluation
   - Model comparison
   - Best model selection

5. **Production Deployment**
   - Model serialization
   - Preprocessing pipeline preservation
   - Prediction API creation

6. **Monitoring and Logging**
   - MLflow experiment tracking
   - Performance monitoring
   - Prediction logging

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib
- mlflow
- pickle

## Future Improvements

1. Add hyperparameter tuning using MLflow
2. Implement cross-validation
3. Add model explainability tools
4. Create a web interface for predictions
5. Add model monitoring dashboards
6. Implement A/B testing capabilities
