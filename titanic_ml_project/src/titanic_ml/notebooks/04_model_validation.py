# Databricks notebook source
# MAGIC %md
# MAGIC # Titanic ML Pipeline - Model Validation
# MAGIC 
# MAGIC This notebook validates the trained model before deployment.
# MAGIC 
# MAGIC **Input:** Registered model from Unity Catalog
# MAGIC **Output:** Validation report and approval for deployment

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
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import json

# Set registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Model name
model_name = f"{catalog}.{schema}.titanic_survival_model"
print(f"Validating model: {model_name}")

# Set catalog and schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Champion Model

# COMMAND ----------

# Get the champion model
try:
    model_version_info = client.get_model_version_by_alias(model_name, "champion")
    model_version = model_version_info.version
    print(f"Champion model version: {model_version}")
except Exception as e:
    # If no champion alias, get latest version
    versions = client.search_model_versions(f"name='{model_name}'")
    model_version = max([int(v.version) for v in versions])
    print(f"Latest model version: {model_version}")

# Load the model
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
print(f"✅ Model loaded from {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Validation Data

# COMMAND ----------

# Load feature data
df = spark.sql("SELECT * FROM titanic_features").toPandas()

# Define feature columns
feature_columns = [
    'pclass', 'sex_encoded', 'age_imputed', 'sibsp', 'parch', 
    'fare', 'embarked_encoded', 'family_size', 'is_alone', 'fare_per_person'
]

X = df[feature_columns].values
y = df['survived'].values

print(f"Validation data shape: {X.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Validation

# COMMAND ----------

# Make predictions
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred),
    "f1_score": f1_score(y, y_pred),
    "roc_auc": roc_auc_score(y, y_pred_proba)
}

print("=" * 60)
print("VALIDATION METRICS")
print("=" * 60)
for metric_name, metric_value in metrics.items():
    print(f"  {metric_name}: {metric_value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix

# COMMAND ----------

# Confusion matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Died  Survived")
print(f"Actual Died      {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"       Survived  {cm[1,0]:4d}     {cm[1,1]:4d}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification Report

# COMMAND ----------

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Died', 'Survived']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Thresholds

# COMMAND ----------

# Define minimum thresholds for deployment
THRESHOLDS = {
    "accuracy": 0.75,
    "precision": 0.70,
    "recall": 0.50,
    "f1_score": 0.60,
    "roc_auc": 0.75
}

# Check if model passes all thresholds
validation_passed = True
validation_results = {}

print("=" * 60)
print("VALIDATION THRESHOLD CHECK")
print("=" * 60)
for metric_name, threshold in THRESHOLDS.items():
    actual = metrics[metric_name]
    passed = actual >= threshold
    validation_results[metric_name] = {
        "actual": actual,
        "threshold": threshold,
        "passed": passed
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {metric_name}: {actual:.4f} >= {threshold:.2f} {status}")
    if not passed:
        validation_passed = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Validation Report

# COMMAND ----------

# Create validation report
validation_report = {
    "model_name": model_name,
    "model_version": model_version,
    "validation_passed": validation_passed,
    "metrics": metrics,
    "thresholds": THRESHOLDS,
    "validation_results": validation_results,
    "confusion_matrix": cm.tolist(),
    "sample_size": len(y)
}

# Save to Delta table
report_df = spark.createDataFrame([{
    "model_name": model_name,
    "model_version": int(model_version),
    "validation_passed": validation_passed,
    "accuracy": metrics["accuracy"],
    "precision": metrics["precision"],
    "recall": metrics["recall"],
    "f1_score": metrics["f1_score"],
    "roc_auc": metrics["roc_auc"],
    "validated_at": pd.Timestamp.now()
}])

report_df.write.format("delta").mode("append").saveAsTable("model_validation_log")
print(f"\n✅ Validation report saved to {catalog}.{schema}.model_validation_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Decision

# COMMAND ----------

if validation_passed:
    print("\n" + "=" * 60)
    print("✅ MODEL VALIDATION PASSED")
    print("=" * 60)
    print(f"Model {model_name} version {model_version} is approved for deployment.")
    result = "APPROVED"
else:
    print("\n" + "=" * 60)
    print("❌ MODEL VALIDATION FAILED")
    print("=" * 60)
    print(f"Model {model_name} version {model_version} did NOT meet all thresholds.")
    print("Please review the metrics and retrain if necessary.")
    result = "REJECTED"

# COMMAND ----------

# Pass output to next task
dbutils.notebook.exit(json.dumps({
    "model_name": model_name,
    "model_version": model_version,
    "validation_result": result,
    "metrics": metrics
}))
