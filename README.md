# ML Modeling Challenge B
Data Science Assessment

## Challenge
You are presented with a binary classification problem related to NBA player performance. Your task is to build, train, and evaluate a model capable of predicting whether a player will perform Above Average (1) or Below Average (0) in a given game based on historical game statistics.

The assessment consists of:
• A training dataset containing 9,000 samples.
• A blind test dataset containing 1,000 samples.

## Preprocessing 
proprocessing for the best performance dataset 'ds4'

### 1. Identifier Variables
| Variable   | Preprocessing |
|------------|---------------|
| row_id     | None          |
| player_id  | None          |

### 2. Categorical Variables
- **Frequency Encoding** applied to all categorical variables except `rest_days`.

| Variable            | Preprocessing |
|--------------------|---------------|
| position            | Freq Enc      |
| team                | Freq Enc      |
| opponent            | Freq Enc      |
| game_location       | Freq Enc      |
| high_usage_scorer   | Freq Enc      |
| high_eff_min        | Freq Enc      |
| high_eff_scorer     | Freq Enc      |
| rest_days           | None          |

### 3. Continuous & Discrete Variables
- **Quantile binning** applied according to predefined bin counts.
- **Raw numeric columns** (`steals`, `blocks`, `turnovers`) are left unprocessed.

| Variable            | Preprocessing       |
|--------------------|-------------------|
| minutes_played      | Binning Quant 5   |
| fg_pct              | Binning Quant 8   |
| three_pct           | Binning Quant 5   |
| ft_pct              | Binning Quant 6   |
| age                 | Binning Quant 9   |
| plus_minus          | Binning Quant 7   |
| efficiency          | Binning Quant 20  |
| points              | Binning Quant 6   |
| rebounds            | Binning Quant 7   |
| assists             | Binning Quant 4   |
| eff_per_point       | Binning Quant 6   |
| eff_per_min         | Binning Quant 6   |
| points_per_min      | Binning Quant 6   |
| scoring_impact      | Binning Quant 6   |
| eff_times_minutes   | Binning Quant 6   |
| scoring_volume      | Binning Quant 6   |
| steals              | None              |
| blocks              | None              |
| turnovers           | None              |

### 4. Engineered Features
Engineered numerical features are treated like continuous variables and **quantile-binned**:
  - `eff_per_point`:
  Efficiency per point scored. Computed as `efficiency / points`, with division-by-zero handled by replacing infinities and NaNs with 0.

  - `eff_per_min`:
  Efficiency per minute played. Computed as `efficiency / minutes_played`, with infinities and NaNs replaced with 0.

  - `points_per_min`:
  Scoring rate per minute played. Computed as `points / minutes_played`, handling division-by-zero similarly.

  - `scoring_impact`:
  Player scoring impact. Computed as `efficiency * points`. Captures how efficient scoring translates into impact.

  - `eff_times_minutes`:
  Total contribution. Computed as `efficiency * minutes_played`. Captures overall contribution across playing time.

  - `scoring_volume`:
  Total scoring volume. Computed as `points * minutes_played`. Reflects heavy usage and productivity.

  - `high_eff_scorer`:
  Binary flag for high-efficiency scorers. Players flagged if `efficiency >= 20` and `points >= 15`.

  - `high_eff_min`:
  Binary flag for high-intensity players. Players flagged if `eff_per_min > 0.8` and `minutes_played > 30`.

  - `high_usage_scorer`:
  Binary flag for high-usage scorers. Players flagged if `points >= 20` and `minutes_played >= 30`


### 5. Target Variable
| Variable | Preprocessing |
|----------|---------------|
| target   | None          |

### Data set engineering

Frequency Encoding:

- Categorical variables are converted to numeric values based on category frequency.
  Applied to the following columns:
  position, team, opponent, game_location, high_usage_scorer, high_eff_min, high_eff_scorer
- The rest_days column is excluded in DS4 to avoid potential sparsity or redundancy.

Raw Numeric Features:

- Certain numeric features are kept in their original form without scaling or binning.
- Columns retained as raw numeric in DS4:
steals, blocks, turnovers
- These statistics are relatively low-dimensional and directly informative for player performance.

Quantile-Based Binning:

- Continuous features are transformed into quantile bins
- Features binned in DS4 include:
Minutes and efficiency metrics: minutes_played, efficiency, eff_per_point, eff_per_min, points_per_min, eff_times_minutes, scoring_impact, scoring_volume
- Shooting percentages: fg_pct, three_pct, ft_pct
- Game contributions: points, rebounds, assists
- Other statistics: plus_minus, age

### Feature importance
To identify the most predictive variables, I implemented a two-step feature importance ranking using a Random Forest model for each dataset:
 - `Tree-based importance`: 
  Measures how much each feature reduces the node impurity across all trees in the forest.

 - `Permutation importance`:
  Measures the decrease in model accuracy when the feature’s values are randomly permuted, providing a robust estimate of a feature’s predictive contribution.

### Feature selection
The 6 top features whith the highest prediction power were automatically ranked by importance and automatically sellected:

- `efficiency_binning_quantile`	
- `ff_times_minutes_binning_quantile`	
- `scoring_impact_binning_quantile`	
- `eff_per_point_binning_quantile`	
- `eff_per_min_binning_quantile`	
- `points_binning_quantile`

## Model
Three supervised classification models were evaluated to predict player performance outcomes, using **cross-validated ROC-AUC** and **test ROC-AUC** for comparison.

---
Three supervised classification models were evaluated on DS4 using **cross-validated ROC-AUC** and **test ROC-AUC**.

| Model            | CV ROC-AUC | Test ROC-AUC | Notes                                           |
|-----------------|------------|--------------|------------------------------------------------|
| Decision Tree    | 0.917      | 0.921        | Simple, interpretable, captures non-linear patterns; depth limited to avoid overfitting |
| Random Forest    | 0.914      | 0.915        | Ensemble reduces variance; stable predictions |
| Gradient Boosting| 0.920      | 0.924        | Sequential ensemble capturing complex patterns; highest predictive performance |
---

Gradient Boosting is the top-performing model on DS4, achieving the highest ROC-AUC and strong generalization.


## Hyperparameters
Key hyperparameters for each model:

---
| Model            | Key Hyperparameters                                      | Notes                                   |
|-----------------|---------------------------------------------------------|----------------------------------------|
| Decision Tree    | max_depth=5, min_samples_leaf=1, min_samples_split=5   | Small, interpretable tree; avoids overfitting |
| Random Forest    | max_depth=10, min_samples_leaf=2, min_samples_split=5, n_estimators=200 | Ensemble of trees; reduces variance    |
| Gradient Boosting| learning_rate=0.05, max_depth=3, n_estimators=100      | Sequential boosting; captures complex patterns |
---

Gradient Boosting balances depth and learning rate effectively, achieving superior performance with a moderate number of estimators, while Random Forest uses deeper trees and more estimators for stability. Decision Tree is simplest but slightly less predictive.


## Validation metrics

---
| Metric            | Value  | Interpretation                             |
|-------------------|--------|--------------------------------------------|
| Accuracy          | 85.2%  | Overall correct predictions                |
| Precision         | 83.2%  | Correct positive predictions               |
| Recall / TPR      | 78.3%  | Proportion of actual positives detected    |
| Specificity / TNR | 89.7%  | Proportion of actual negatives detected    |
| FPR               | 10.3%  | Rate of false positives                    |
| FNR               | 21.7%  | Rate of false negatives                    |
| F1-score          | 80.7%  | Balance between precision and recall       |
| ROC-AUC           | 0.924  | Excellent discrimination across thresholds |
---

The model demonstrates strong predictive performance with high discrimination between positive and negative outcomes, low misclassification rates, and reliable overall accuracy.


## Performance
| Model              | CV ROC-AUC | Test ROC-AUC | Notes |
|--------------------|------------|--------------|-------|
| Decision Tree      | 0.917      | 0.921        | Simple tree, interpretable, slightly lower performance |
| Random Forest      | 0.914      | 0.915        | Ensemble reduces variance, good generalization        |
| Gradient Boosting  | 0.920      | 0.924        | Best performance, captures complex patterns          |

Gradient Boosting is the top-performing model, achieving the highest ROC-AUC and robust generalization, making it the recommended choice for production.

