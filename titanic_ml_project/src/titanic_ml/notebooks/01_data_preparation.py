# Databricks notebook source
# MAGIC %md
# MAGIC # Titanic ML Pipeline - Data Preparation
# MAGIC 
# MAGIC This notebook loads and prepares the Titanic dataset for ML training.
# MAGIC 
# MAGIC **Input:** Raw Titanic dataset from seaborn
# MAGIC **Output:** Delta table in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

# Get parameters
try:
    catalog = dbutils.widgets.get("catalog")
    schema = dbutils.widgets.get("schema")
except:
    # Default values for interactive development
    catalog = "dbdemos_henryk"
    schema = "titanic_ml"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import pandas as pd
import seaborn as sns
from pyspark.sql import functions as F

# Set catalog and schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Using: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Titanic Dataset

# COMMAND ----------

# Load the Titanic dataset from seaborn
titanic_df = sns.load_dataset('titanic')

print(f"Dataset shape: {titanic_df.shape}")
print(f"\nColumn names: {list(titanic_df.columns)}")
print(f"\nData types:")
print(titanic_df.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check for missing values
missing = titanic_df.isnull().sum()
missing_pct = (titanic_df.isnull().sum() / len(titanic_df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

print("Missing Values:")
print(missing_df)

# COMMAND ----------

# Check for duplicates
duplicates = titanic_df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Check target distribution
print(f"\nTarget distribution:")
print(titanic_df['survived'].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Table

# COMMAND ----------

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(titanic_df)

# Add metadata columns
spark_df = spark_df.withColumn("_loaded_at", F.current_timestamp())
spark_df = spark_df.withColumn("_source", F.lit("seaborn_dataset"))

# Save as Delta table
table_name = "titanic_raw"
spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

print(f"✅ Data saved to {catalog}.{schema}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Data

# COMMAND ----------

# Verify the table
verify_df = spark.sql(f"SELECT * FROM {table_name}")
print(f"Total rows: {verify_df.count()}")
verify_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

# Display summary statistics
display(verify_df.describe())

# COMMAND ----------

# Pass output to next task
dbutils.notebook.exit(f"Successfully loaded {verify_df.count()} rows to {catalog}.{schema}.{table_name}")
