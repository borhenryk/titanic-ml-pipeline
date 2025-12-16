# Databricks notebook source
# MAGIC %md
# MAGIC # Titanic ML Pipeline - Feature Engineering
# MAGIC 
# MAGIC This notebook creates features for ML training.
# MAGIC 
# MAGIC **Input:** Raw Titanic data from Delta table
# MAGIC **Output:** Feature table with engineered features

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

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# Set catalog and schema
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Raw Data

# COMMAND ----------

# Load raw data
raw_df = spark.sql("SELECT * FROM titanic_raw")
print(f"Loaded {raw_df.count()} rows")
raw_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Create features
features_df = raw_df

# 1. Family size feature
features_df = features_df.withColumn("family_size", F.col("sibsp") + F.col("parch") + 1)

# 2. Is alone feature
features_df = features_df.withColumn("is_alone", F.when(F.col("family_size") == 1, 1).otherwise(0))

# 3. Age groups
features_df = features_df.withColumn(
    "age_group",
    F.when(F.col("age") < 12, "child")
    .when(F.col("age") < 18, "teen")
    .when(F.col("age") < 35, "young_adult")
    .when(F.col("age") < 50, "adult")
    .otherwise("senior")
)

# 4. Fare per person
features_df = features_df.withColumn(
    "fare_per_person",
    F.when(F.col("family_size") > 0, F.col("fare") / F.col("family_size"))
    .otherwise(F.col("fare"))
)

# 5. Encode sex
features_df = features_df.withColumn(
    "sex_encoded",
    F.when(F.col("sex") == "male", 1).otherwise(0)
)

# 6. Encode embarked
features_df = features_df.withColumn(
    "embarked_encoded",
    F.when(F.col("embarked") == "S", 0)
    .when(F.col("embarked") == "C", 1)
    .when(F.col("embarked") == "Q", 2)
    .otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handle Missing Values

# COMMAND ----------

# Calculate median age by class for imputation
from pyspark.sql import Window

# Get median age by pclass
median_ages = features_df.groupBy("pclass").agg(
    F.percentile_approx("age", 0.5).alias("median_age")
)

# Join and fill missing ages
features_df = features_df.join(median_ages, on="pclass", how="left")
features_df = features_df.withColumn(
    "age_imputed",
    F.coalesce(F.col("age"), F.col("median_age"))
)

# Fill remaining missing ages with overall median
overall_median = features_df.select(F.percentile_approx("age", 0.5)).collect()[0][0]
features_df = features_df.withColumn(
    "age_imputed",
    F.coalesce(F.col("age_imputed"), F.lit(overall_median))
)

# Fill missing embarked with mode (S)
features_df = features_df.withColumn(
    "embarked_imputed",
    F.coalesce(F.col("embarked"), F.lit("S"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select Final Features

# COMMAND ----------

# Select features for modeling
feature_columns = [
    "survived",  # target
    "pclass",
    "sex_encoded",
    "age_imputed",
    "sibsp",
    "parch",
    "fare",
    "embarked_encoded",
    "family_size",
    "is_alone",
    "fare_per_person"
]

# Also keep original columns for reference
reference_columns = ["sex", "age", "embarked", "class", "who", "adult_male"]

final_df = features_df.select(
    feature_columns + reference_columns + ["_loaded_at", "_source"]
).withColumn("_feature_created_at", F.current_timestamp())

print(f"Final feature columns: {feature_columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Feature Table

# COMMAND ----------

# Save as feature table
table_name = "titanic_features"
final_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

print(f"✅ Features saved to {catalog}.{schema}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Features

# COMMAND ----------

# Verify the table
verify_df = spark.sql(f"SELECT * FROM {table_name}")
print(f"Total rows: {verify_df.count()}")
print(f"\nFeature statistics:")
display(verify_df.select(feature_columns).describe())

# COMMAND ----------

# Pass output to next task
dbutils.notebook.exit(f"Successfully created features: {catalog}.{schema}.{table_name}")
