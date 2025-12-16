# Databricks notebook source
# MAGIC %md
# MAGIC # Titanic ML Pipeline - Model Training
# MAGIC 
# MAGIC This notebook trains the survival prediction model with hyperparameter optimization.
# MAGIC 
# MAGIC **Input:** Feature table
# MAGIC **Output:** Trained model registered in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Get parameters
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
    experiment_name = dbutils.widgets.get("experiment_name")
except:
    # Default values for interactive development
    catalog = "dbdemos_henryk"
    schema = "titanic_ml"
    experiment_name = "/Users/henryk.borzymowski@databricks.com/titanic_ml_experiment"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# Set MLflow experiment
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment set: {experiment_name}")

# Set catalog and schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Features

# COMMAND ----------

# Load features
df = spark.sql("SELECT * FROM titanic_features").toPandas()
print(f"Loaded {len(df)} rows")

# Define feature columns
feature_columns = [
    'pclass', 'sex_encoded', 'age_imputed', 'sibsp', 'parch', 
    'fare', 'embarked_encoded', 'family_size', 'is_alone', 'fare_per_person'
]

X = df[feature_columns]
y = df['survived']

print(f"Features: {feature_columns}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Splitting

# COMMAND ----------

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter Optimization

# COMMAND ----------

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    model_type = trial.suggest_categorical('model_type', ['gradient_boosting', 'random_forest'])
    
    if model_type == 'gradient_boosting':
        params = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('gb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True),
            'min_samples_split': trial.suggest_int('gb_min_samples_split', 2, 20),
            'random_state': 42
        }
        model = GradientBoostingClassifier(**params)
    else:
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    return scores.mean()

# Run optimization
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction='maximize', study_name='titanic_survival')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest trial accuracy: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Final Model

# COMMAND ----------

# Get best parameters
best_params = study.best_trial.params
model_type = best_params['model_type']

print(f"Training final {model_type} model...")

# Create model with best parameters
if model_type == 'gradient_boosting':
    final_params = {
        'n_estimators': best_params['gb_n_estimators'],
        'max_depth': best_params['gb_max_depth'],
        'learning_rate': best_params['gb_learning_rate'],
        'min_samples_split': best_params['gb_min_samples_split'],
        'random_state': 42
    }
    final_model = GradientBoostingClassifier(**final_params)
else:
    final_params = {
        'n_estimators': best_params['rf_n_estimators'],
        'max_depth': best_params['rf_max_depth'],
        'min_samples_split': best_params['rf_min_samples_split'],
        'min_samples_leaf': best_params['rf_min_samples_leaf'],
        'random_state': 42
    }
    final_model = RandomForestClassifier(**final_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Logging

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name="titanic_best_model") as run:
    run_id = run.info.run_id
    
    # Log parameters
    mlflow.log_params(final_params)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("features", feature_columns)
    
    # Train model
    final_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = final_model.predict(X_test_scaled)
    y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Create signature
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    signature = infer_signature(X_train_df, y_train)
    
    # Log model
    mlflow.sklearn.log_model(
        final_model, 
        "model",
        signature=signature,
        input_example=X_train_df.head(5)
    )
    
    # Log scaler
    import pickle
    scaler_path = "/tmp/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact(scaler_path)
    
    print(f"\n{'='*60}")
    print(f"FINAL MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Model Type: {model_type}")
    print(f"Run ID: {run_id}")
    print(f"\nTest Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model

# COMMAND ----------

# Register model in Unity Catalog
mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog}.{schema}.titanic_survival_model"
model_uri = f"runs:/{run_id}/model"

print(f"Registering model: {model_name}")

result = mlflow.register_model(model_uri=model_uri, name=model_name)

print(f"✅ Model registered!")
print(f"   Name: {result.name}")
print(f"   Version: {result.version}")

# Set champion alias
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(model_name, "champion", result.version)
print(f"   Alias 'champion' set to version {result.version}")

# COMMAND ----------

# Pass output to next task
dbutils.notebook.exit(f"Model registered: {model_name} version {result.version}, run_id: {run_id}")
