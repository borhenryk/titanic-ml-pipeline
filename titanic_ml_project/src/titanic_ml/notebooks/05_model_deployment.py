# Databricks notebook source
# MAGIC %md
# MAGIC # Titanic ML Pipeline - Model Deployment
# MAGIC 
# MAGIC This notebook deploys the validated model to a serving endpoint.
# MAGIC 
# MAGIC **Input:** Validated model from Unity Catalog
# MAGIC **Output:** Model deployed to serving endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Get parameters
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
    endpoint_name = dbutils.widgets.get("endpoint_name")
except:
    # Default values for interactive development
    catalog = "dbdemos_henryk"
    schema = "titanic_ml"
    endpoint_name = "titanic-survival-endpoint"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import requests
import json
import time

# Set registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Model name
model_name = f"{catalog}.{schema}.titanic_survival_model"

# Get workspace URL and token
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print(f"Workspace: https://{host}")
print(f"Model: {model_name}")
print(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Champion Model Version

# COMMAND ----------

# Get the champion model version
try:
    model_version_info = client.get_model_version_by_alias(model_name, "champion")
    model_version = model_version_info.version
    print(f"Champion model version: {model_version}")
except Exception as e:
    # If no champion alias, get latest version
    versions = client.search_model_versions(f"name='{model_name}'")
    model_version = max([int(v.version) for v in versions])
    print(f"Latest model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Existing Endpoint

# COMMAND ----------

# Check if endpoint exists
endpoint_url = f"https://{host}/api/2.0/serving-endpoints/{endpoint_name}"
response = requests.get(endpoint_url, headers=headers)

endpoint_exists = response.status_code == 200

if endpoint_exists:
    print(f"✅ Endpoint '{endpoint_name}' exists")
    endpoint_info = response.json()
    state = endpoint_info.get('state', {}).get('ready', 'unknown')
    print(f"   Current state: {state}")
else:
    print(f"Endpoint '{endpoint_name}' does not exist")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy or Update Endpoint

# COMMAND ----------

# Entity name for serving (replace dots with underscores)
served_entity_name = f"{model_name.replace('.', '_')}_{model_version}"

if endpoint_exists:
    # Update existing endpoint
    print(f"Updating endpoint to model version {model_version}...")
    
    update_config = {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": str(model_version),
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": served_entity_name,
                    "traffic_percentage": 100
                }
            ]
        }
    }
    
    update_url = f"https://{host}/api/2.0/serving-endpoints/{endpoint_name}/config"
    update_response = requests.put(update_url, headers=headers, json=update_config)
    
    if update_response.status_code == 200:
        print(f"✅ Endpoint updated successfully")
    else:
        print(f"⚠️ Update response: {update_response.status_code}")
        print(update_response.text)
        
else:
    # Create new endpoint
    print(f"Creating new endpoint '{endpoint_name}'...")
    
    create_config = {
        "name": endpoint_name,
        "config": {
            "served_entities": [
                {
                    "entity_name": model_name,
                    "entity_version": str(model_version),
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True
                }
            ]
        }
    }
    
    create_url = f"https://{host}/api/2.0/serving-endpoints"
    create_response = requests.post(create_url, headers=headers, json=create_config)
    
    if create_response.status_code in [200, 201]:
        print(f"✅ Endpoint created successfully")
    else:
        print(f"❌ Error creating endpoint: {create_response.status_code}")
        print(create_response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint Ready

# COMMAND ----------

# Wait for endpoint to be ready (max 10 minutes)
max_wait_time = 600  # seconds
wait_interval = 30  # seconds
elapsed = 0

print(f"Waiting for endpoint to be ready (max {max_wait_time}s)...")

while elapsed < max_wait_time:
    response = requests.get(endpoint_url, headers=headers)
    
    if response.status_code == 200:
        state = response.json().get('state', {})
        ready = state.get('ready', 'NOT_READY')
        config_update = state.get('config_update', 'UNKNOWN')
        
        print(f"  [{elapsed}s] Ready: {ready}, Config Update: {config_update}")
        
        if ready == 'READY':
            print(f"\n✅ Endpoint is READY!")
            break
    
    time.sleep(wait_interval)
    elapsed += wait_interval

if elapsed >= max_wait_time:
    print(f"\n⚠️ Timeout waiting for endpoint. Please check manually.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Endpoint

# COMMAND ----------

# Test the endpoint with sample data
test_data = {
    "dataframe_records": [
        {
            "pclass": 1,
            "sex_encoded": 0,  # female
            "age_imputed": 35.0,
            "sibsp": 1,
            "parch": 0,
            "fare": 53.1,
            "embarked_encoded": 0,  # S
            "family_size": 2,
            "is_alone": 0,
            "fare_per_person": 26.55
        },
        {
            "pclass": 3,
            "sex_encoded": 1,  # male
            "age_imputed": 22.0,
            "sibsp": 1,
            "parch": 0,
            "fare": 7.25,
            "embarked_encoded": 0,  # S
            "family_size": 2,
            "is_alone": 0,
            "fare_per_person": 3.625
        }
    ]
}

invoke_url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"

print("Testing endpoint with sample data...")
test_response = requests.post(invoke_url, headers=headers, json=test_data)

if test_response.status_code == 200:
    predictions = test_response.json()
    print(f"\n✅ Endpoint test successful!")
    print(f"Predictions: {predictions}")
else:
    print(f"\n⚠️ Endpoint test failed: {test_response.status_code}")
    print(test_response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Deployment

# COMMAND ----------

# Log deployment to Delta table
import pandas as pd

deployment_log = spark.createDataFrame([{
    "model_name": model_name,
    "model_version": int(model_version),
    "endpoint_name": endpoint_name,
    "workspace_url": f"https://{host}",
    "deployed_at": pd.Timestamp.now(),
    "status": "SUCCESS" if test_response.status_code == 200 else "DEPLOYED_NOT_TESTED"
}])

deployment_log.write.format("delta").mode("append").saveAsTable("model_deployment_log")
print(f"\n✅ Deployment logged to {catalog}.{schema}.model_deployment_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("DEPLOYMENT SUMMARY")
print("=" * 60)
print(f"  Model: {model_name}")
print(f"  Version: {model_version}")
print(f"  Endpoint: {endpoint_name}")
print(f"  URL: https://{host}/serving-endpoints/{endpoint_name}")
print("=" * 60)

# COMMAND ----------

# Pass output to next task
dbutils.notebook.exit(json.dumps({
    "model_name": model_name,
    "model_version": model_version,
    "endpoint_name": endpoint_name,
    "endpoint_url": f"https://{host}/serving-endpoints/{endpoint_name}",
    "status": "SUCCESS"
}))
