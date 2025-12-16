# Titanic Dataset - Exploratory Data Analysis Report

**Generated from Databricks Cluster**  
**Dataset:** `dbdemos_henryk.titanic_ml.titanic_raw`

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| **Rows** | 891 |
| **Columns** | 15 |

### Column Details

| Column | Data Type |
|--------|------------|
| survived | int64 |
| pclass | int64 |
| sex | object |
| age | float64 |
| sibsp | int64 |
| parch | int64 |
| fare | float64 |
| embarked | object |
| class | object |
| who | object |
| adult_male | bool |
| deck | object |
| embark_town | object |
| alive | object |
| alone | bool |

---

## 2. Descriptive Statistics (Numerical Features)

| Statistic | survived | pclass | age | sibsp | parch | fare |
|-----------|----------|--------|-----|-------|-------|------|
| **count** | 891.00 | 891.00 | 714.00 | 891.00 | 891.00 | 891.00 |
| **mean** | 0.38 | 2.31 | 29.70 | 0.52 | 0.38 | 32.20 |
| **std** | 0.49 | 0.84 | 14.53 | 1.10 | 0.81 | 49.69 |
| **min** | 0.00 | 1.00 | 0.42 | 0.00 | 0.00 | 0.00 |
| **25%** | 0.00 | 2.00 | 20.12 | 0.00 | 0.00 | 7.91 |
| **50%** | 0.00 | 3.00 | 28.00 | 0.00 | 0.00 | 14.45 |
| **75%** | 1.00 | 3.00 | 38.00 | 1.00 | 0.00 | 31.00 |
| **max** | 1.00 | 3.00 | 80.00 | 8.00 | 6.00 | 512.33 |

---

## 3. Target Variable Analysis (Survived)

### Survival Distribution

```
Died (0):     ████████████████████████░░░░░░░░░░░░░░░░  549 (61.6%)
Survived (1): ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░  342 (38.4%)
```

| Outcome | Count | Percentage |
|---------|-------|------------|
| Died (0) | 549 | 61.6% |
| Survived (1) | 342 | 38.4% |

---

## 4. Missing Values Analysis

| Column | Missing Count | Percentage |
|--------|---------------|------------|
| deck | 688 | 77.22% |
| age | 177 | 19.87% |
| embarked | 2 | 0.22% |
| embark_town | 2 | 0.22% |

### Recommendations for Missing Data:
- **Age (~20%):** Impute using median grouped by class and sex
- **Deck (~77%):** Consider dropping or using Cabin info for extraction
- **Embarked (<1%):** Use mode imputation (Southampton)

---

## 5. Categorical Features Analysis

### Sex Distribution
| Sex | Count | Percentage |
|-----|-------|------------|
| male | 577 | 64.8% |
| female | 314 | 35.2% |

### Passenger Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Third | 491 | 55.1% |
| First | 216 | 24.2% |
| Second | 184 | 20.7% |

### Embarkation Port Distribution
| Port | Count | Percentage |
|------|-------|------------|
| Southampton (S) | 644 | 72.3% |
| Cherbourg (C) | 168 | 18.9% |
| Queenstown (Q) | 77 | 8.6% |

### Who (Age Group) Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| man | 537 | 60.3% |
| woman | 271 | 30.4% |
| child | 83 | 9.3% |

### Deck Distribution
| Deck | Count | Percentage |
|------|-------|------------|
| C | 59 | 6.6% |
| B | 47 | 5.3% |
| D | 33 | 3.7% |
| E | 32 | 3.6% |
| A | 15 | 1.7% |
| F | 13 | 1.5% |
| G | 4 | 0.4% |

---

## 6. Survival Rates by Key Features

### By Sex
```
Female: ████████████████████████████████████████████░░░░░░  74.2%
Male:   ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  18.9%
```

### By Passenger Class
```
1st Class: ████████████████████████████████░░░░░░░░░░░░░░░░  63.0%
2nd Class: ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  47.3%
3rd Class: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  24.2%
```

### By Embarkation Port
```
Cherbourg (C):   ████████████████████████████░░░░░░░░░░░░░░  55.4%
Queenstown (Q):  ████████████████████░░░░░░░░░░░░░░░░░░░░░░  39.0%
Southampton (S): █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  33.7%
```

### By Age Group
```
Child (0-12):       █████████████████████████████░░░░░░░░░░░  58.0%
Teen (12-18):       █████████████████████░░░░░░░░░░░░░░░░░░░  42.9%
Young Adult (18-35):████████████████████░░░░░░░░░░░░░░░░░░░░  38.3%
Adult (35-50):      ████████████████████░░░░░░░░░░░░░░░░░░░░  39.9%
Senior (50+):       █████████████████░░░░░░░░░░░░░░░░░░░░░░░  34.4%
```

---

## 7. Correlation Matrix

| | survived | pclass | age | sibsp | parch | fare |
|---|----------|--------|-----|-------|-------|------|
| **survived** | 1.000 | -0.338 | -0.077 | -0.035 | 0.082 | 0.257 |
| **pclass** | -0.338 | 1.000 | -0.369 | 0.083 | 0.018 | -0.549 |
| **age** | -0.077 | -0.369 | 1.000 | -0.308 | -0.189 | 0.096 |
| **sibsp** | -0.035 | 0.083 | -0.308 | 1.000 | 0.415 | 0.160 |
| **parch** | 0.082 | 0.018 | -0.189 | 0.415 | 1.000 | 0.216 |
| **fare** | 0.257 | -0.549 | 0.096 | 0.160 | 0.216 | 1.000 |

### Key Correlations:
- **pclass ↔ survived:** -0.338 (Strong negative - lower class = higher survival)
- **fare ↔ pclass:** -0.549 (Strong negative - higher fare = better class)
- **fare ↔ survived:** 0.257 (Moderate positive - higher fare = higher survival)

---

## 8. Key Insights & Recommendations

### 🔑 Key Findings

1. **Class is the Strongest Predictor**
   - First-class passengers had **63.0%** survival rate
   - Third-class passengers had only **24.2%** survival rate
   - Correlation: -0.338 (highest correlation with survival)

2. **Gender is Critical**
   - Female passengers: **74.2%** survival rate
   - Male passengers: **18.9%** survival rate
   - This reflects the "women and children first" evacuation policy

3. **Age Matters**
   - Children (0-12) had **58.0%** survival rate
   - Seniors (50+) had **34.4%** survival rate
   - Age has ~20% missing values requiring imputation

4. **Embarkation Port Impact**
   - Cherbourg passengers had highest survival (55.4%)
   - May correlate with higher class passengers boarding there

### 🛠️ Feature Engineering Recommendations

| Feature | Formula | Rationale |
|---------|---------|------------|
| `family_size` | sibsp + parch + 1 | Traveling alone vs. with family |
| `is_alone` | family_size == 1 | Binary indicator |
| `fare_per_person` | fare / family_size | Normalized fare |
| `age_bin` | Categorical bins | Handle missing values better |
| `title` | Extract from Name | Mr., Mrs., Miss, Master, etc. |

### 📊 Model Recommendations

- **Algorithm Candidates:** Random Forest, Gradient Boosting, Logistic Regression
- **Handle Class Imbalance:** 38.4% survived (consider SMOTE or class weights)
- **Key Features to Include:** sex, pclass, age, fare, embarked, family_size
- **Cross-Validation:** Use stratified K-fold to maintain class balance

---

## Summary Statistics Table

| Metric | Value |
|--------|-------|
| Total Passengers | 891 |
| Survivors | 342 (38.4%) |
| Casualties | 549 (61.6%) |
| Male Passengers | 577 (64.8%) |
| Female Passengers | 314 (35.2%) |
| Mean Age | 29.7 years |
| Mean Fare | $32.20 |
| Missing Age Values | 177 (19.87%) |

---

*Report generated from Databricks Unity Catalog*  
*Catalog: dbdemos_henryk | Schema: titanic_ml | Table: titanic_raw*
