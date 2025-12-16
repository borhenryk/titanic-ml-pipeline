# Titanic ML Pipeline

An end-to-end machine learning pipeline for predicting Titanic passenger survival, built with Databricks Asset Bundles (DABs) and deployed via CI/CD with GitHub Actions.

## 📋 Project Overview

This project demonstrates a production-ready ML pipeline that includes:

- **Data Preparation**: Load and clean the Titanic dataset
- **Feature Engineering**: Create predictive features
- **Model Training**: Train with hyperparameter optimization using Optuna
- **Model Validation**: Validate model performance against thresholds
- **Model Deployment**: Deploy to Databricks Model Serving
- **CI/CD**: Automated deployment via GitHub Actions

## 🏗️ Project Structure

```
titanic_ml_project/
├── databricks.yml                    # Bundle configuration
├── resources/
│   └── titanic_pipeline_job.yml     # Job definitions
├── src/titanic_ml/notebooks/
│   ├── 01_data_preparation.py       # Data loading and cleaning
│   ├── 02_feature_engineering.py    # Feature creation
│   ├── 03_model_training.py         # Model training with HPO
│   ├── 04_model_validation.py       # Model validation
│   └── 05_model_deployment.py       # Model deployment
├── reports/
│   └── eda_report.md                # EDA analysis report
├── .github/workflows/
│   └── databricks-deploy.yml        # CI/CD pipeline
└── README.md
```

## 🚀 Quick Start

### Prerequisites

1. **Databricks CLI** installed and configured
2. **GitHub account** with repository access
3. **Databricks workspace** with Unity Catalog enabled

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/borhenryk/titanic-ml-pipeline.git
cd titanic-ml-pipeline/titanic_ml_project
```

2. Configure Databricks CLI:
```bash
databricks auth login --host https://your-workspace.cloud.databricks.com
```

3. Validate the bundle:
```bash
databricks bundle validate -t dev
```

4. Deploy to development:
```bash
databricks bundle deploy -t dev
```

5. Run the pipeline:
```bash
databricks bundle run titanic_ml_pipeline -t dev
```

## ⚙️ Configuration

### Bundle Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `dbdemos_henryk` |
| `schema` | Schema name | `titanic_ml` |
| `experiment_name` | MLflow experiment path | `/Users/.../titanic_ml_experiment` |

### Deployment Targets

| Target | Description | Workspace |
|--------|-------------|------------|
| `dev` | Development | User workspace |
| `staging` | Pre-production | Shared workspace |
| `prod` | Production | Production workspace |

## 🔄 CI/CD Pipeline

### GitHub Secrets Required

Add these secrets to your GitHub repository:

| Secret | Description |
|--------|-------------|
| `DATABRICKS_HOST` | Databricks workspace URL (e.g., `https://dbc-xxxxx.cloud.databricks.com`) |
| `DATABRICKS_TOKEN` | Databricks Personal Access Token |

### Workflow Triggers

- **Push to `develop`**: Deploy to dev
- **Push to `main`**: Deploy to dev → staging → prod (with approval)
- **Pull Request**: Validate bundle only
- **Manual**: Select target and optionally run job

## 📊 Model Performance

The trained Gradient Boosting model achieves:

| Metric | Value |
|--------|-------|
| Accuracy | ~79% |
| Precision | ~80% |
| Recall | ~62% |
| F1 Score | ~70% |
| ROC AUC | ~84% |

### Key Features

1. `sex_encoded` - Gender (most predictive)
2. `pclass` - Passenger class
3. `age_imputed` - Age with imputation
4. `fare` - Ticket fare
5. `family_size` - Family size (engineered)

## 🔗 Resources

### Databricks Resources Created

- **Unity Catalog**:
  - Catalog: `dbdemos_henryk`
  - Schema: `titanic_ml`
  - Tables: `titanic_raw`, `titanic_features`

- **MLflow**:
  - Experiment: `titanic_ml_experiment`
  - Model: `titanic_survival_model`

- **Model Serving**:
  - Endpoint: `titanic-survival-endpoint`

### API Endpoint

```bash
# Test the serving endpoint
curl -X POST \
  https://your-workspace.cloud.databricks.com/serving-endpoints/titanic-survival-endpoint/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "pclass": 1,
      "sex_encoded": 0,
      "age_imputed": 35.0,
      "sibsp": 1,
      "parch": 0,
      "fare": 53.1,
      "embarked_encoded": 0,
      "family_size": 2,
      "is_alone": 0,
      "fare_per_person": 26.55
    }]
  }'
```

## 📖 Documentation

- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
- [Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)

## 📝 License

This project is for demonstration purposes.

## 👥 Contributors

- Henryk Borzymowski
