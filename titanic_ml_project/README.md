# Titanic ML Pipeline

End-to-end Machine Learning pipeline for Titanic survival prediction using Databricks Asset Bundles (DABs) with automated CI/CD deployment via GitHub Actions.

## 🚀 Quick Start

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- Databricks CLI configured
- GitHub account with repository secrets configured

### Deploy to Development
```bash
cd titanic_ml_project
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

### Run the Pipeline
```bash
databricks bundle run titanic_ml_pipeline -t dev
```

## 📁 Project Structure

```
titanic_ml_project/
├── databricks.yml              # Bundle configuration
├── resources/
│   └── titanic_pipeline_job.yml  # Job definition
├── src/titanic_ml/notebooks/
│   ├── 01_data_preparation.py    # Load & prepare data
│   ├── 02_feature_engineering.py # Feature creation
│   ├── 03_model_training.py      # Train with HPO
│   ├── 04_model_validation.py    # Validate model
│   └── 05_model_deployment.py    # Deploy endpoint
├── reports/
│   └── eda_report.md             # EDA findings
└── .github/workflows/
    └── databricks-deploy.yml     # CI/CD pipeline
```

## 🔧 Configuration

### Bundle Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `dbdemos_henryk` |
| `schema` | Schema for ML artifacts | `titanic_ml` |
| `experiment_name` | MLflow experiment path | `/Shared/titanic_ml_experiment` |

### Deployment Targets
- **dev**: Development environment (default)
- **staging**: Pre-production testing
- **prod**: Production deployment

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **On Push to `main`/`develop`**: Validates and deploys to dev
2. **On Push to `main`**: Promotes through staging → prod
3. **On Pull Request**: Validates bundle configuration
4. **Manual Trigger**: Deploy to any environment

### Required Secrets
- `DATABRICKS_HOST`: Workspace URL
- `DATABRICKS_TOKEN`: Personal Access Token

## 📊 Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| Data Preparation | Load Titanic dataset, quality checks | `titanic_raw` table |
| Feature Engineering | Create ML features | `titanic_features` table |
| Model Training | Hyperparameter optimization with Optuna | Registered model |
| Model Validation | Performance validation against thresholds | Validation report |
| Model Deployment | Deploy to serving endpoint | REST API endpoint |

## 🎯 Model Performance

Target metrics for deployment approval:
- Accuracy: ≥ 75%
- Precision: ≥ 70%
- Recall: ≥ 50%
- F1 Score: ≥ 60%
- ROC AUC: ≥ 75%

## 📈 MLflow Tracking

All experiments are logged to MLflow with:
- Hyperparameters
- Performance metrics
- Model artifacts
- Feature importance plots
- Confusion matrices

## 🔗 Resources

- [Databricks Asset Bundles Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)

## 📝 License

This project is for demonstration purposes.

---

*Last updated: December 16, 2025 - CI/CD pipeline ready* ✅
