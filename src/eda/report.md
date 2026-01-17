# EDA Report

##  Column overview
*(See summary statistics table below for data types, ranges, and missing values.)*

### Categorical Columns
---
| Column         | Type   | Non-Null Count | Missing | % Missing | Unique Values | Mode | Mode Count |
|----------------|--------|----------------|---------|-----------|---------------|------|------------|
| game_location  | object | 7000           | 0       | 0.0       | 2             | Home | 3501       |
| position       | object | 7000           | 0       | 0.0       | 5             | SG   | 1475       |
| team           | object | 7000           | 0       | 0.0       | 30            | NYK  | 263        |
| opponent       | object | 7000           | 0       | 0.0       | 30            | GSW  | 269        |
| rest_days      | int64  | 7000           | 0       | 0.0       | 6             | 1    | 0          |
---

### Binary Numeric Columns
---
| Column            | Type  | Non-Null Count | Unique | Min | Max | Mean     | Std     |
|-------------------|-------|----------------|--------|-----|-----|----------|---------|
| high_eff_scorer   | int64 | 7000           | 2      | 0   | 1   | 0.4631   | 0.4987  |
| high_eff_min      | int64 | 7000           | 2      | 0   | 1   | 0.2033   | 0.4025  |
| high_usage_scorer | int64 | 7000           | 2      | 0   | 1   | 0.1461   | 0.3533  |
---

### Continuous Numeric Columns

Integer Continuous

---
| Column          | Type  | Non-Null Count | Unique | Min   | Max   | Mean     | Std     |
|-----------------|-------|----------------|--------|-------|-------|----------|---------|
| steals          | int64 | 7000           | 4      | 0     | 3     | 0.8279   | 0.6810  |
| blocks          | int64 | 7000           | 4      | 0     | 3     | 0.6714   | 0.7267  |
| turnovers       | int64 | 7000           | 6      | 0     | 5     | 1.8226   | 1.0861  |
| assists         | int64 | 7000           | 14     | 0     | 13    | 3.0246   | 2.4167  |
| rebounds        | int64 | 7000           | 18     | 0     | 17    | 5.1433   | 2.9126  |
| age             | int64 | 7000           | 20     | 19    | 39    | 26.5069  | 3.1725  |
| points          | int64 | 7000           | 31     | 0     | 30    | 14.3451  | 5.0570  |
| efficiency      | int64 | 7000           | 45     | 1     | 45    | 22.1897  | 7.1663  |
| plus_minus      | int64 | 7000           | 55     | -26   | 30    | 0.0276   | 8.0310  |
| scoring_impact  | int64 | 7000           | 369    | 0     | 1276  | 350.2890 | 214.8598|
---

Float Continuous

---
| Column            | Type    | Non-Null Count | Unique | Min     | Max     | Mean      | Std      |
|-------------------|---------|----------------|--------|---------|---------|-----------|----------|
| eff_per_point     | float64 | 7000           | 295    | 0.0     | 8.0     | 1.6016    | 0.3792   |
| minutes_played    | float64 | 7000           | 341    | 7.2     | 43.0    | 26.7976   | 7.4013   |
| fg_pct            | float64 | 7000           | 344    | 30.0    | 69.7    | 47.3263   | 6.0355   |
| ft_pct            | float64 | 7000           | 395    | 45.0    | 95.0    | 77.5985   | 7.9644   |
| three_pct         | float64 | 7000           | 429    | 10.0    | 60.0    | 35.0147   | 7.5350   |
| points_per_min    | float64 | 7000           | 2295   | 0.0     | 1.1538  | 0.5354    | 0.1324   |
| scoring_volume    | float64 | 7000           | 2663   | 0.0     | 1275.0  | 413.7422  | 233.2975 |
| eff_per_min       | float64 | 7000           | 2733   | 0.0704  | 1.7778  | 0.8317    | 0.1747   |
| eff_times_minutes | float64 | 7000           | 3113   | 14.2    | 1878.8  | 637.7827  | 345.6669 |
---

### Identifier Variables
These variables identify records and entities but do not carry predictive or analytical meaning by themselves.

- **row_id**: Unique identifier for each row (game-level observation).
- **player_id**: Unique identifier assigned to each player.

---

### Categorical Nominal Variables

- **position**: Player’s on-court position (e.g., SG, PG, SF).
- **team**: Team the player belongs to.
- **opponent**: Opposing team in the game.
- **game_location**: Indicates whether the game was played at *Home* or *Away*.

---

### Categorical Ordinal Variables
- **rest_days**: Number of rest days before the game.

---

### Numerical Continuous Variables

- **minutes_played**: Total minutes played by the player in a game.
- **fg_pct**: Field goal percentage for the game.
- **three_pct**: Three-point shooting percentage for the game.
- **ft_pct**: Free-throw shooting percentage for the game.
- **plus_minus**: Point differential while the player was on the court.

---

### Numerical Discrete Variables (Player & Game Context)
These variables take integer values and describe player context or game impact.

- **age**: Player’s age at the time of the game.
- **efficiency**: Composite performance metric summarizing player contribution.

---

### Numerical Discrete Variables (Box Score Statistics)
These variables represent count-based in-game performance statistics.

- **points**: Total points scored by the player.
- **rebounds**: Total rebounds collected.
- **assists**: Total assists made.
- **steals**: Total steals recorded.
- **blocks**: Total shots blocked.
- **turnovers**: Total turnovers committed.

---

### Binary Target Variable
This variable represents the prediction target for modeling tasks.

- **target**: Binary outcome variable (0 or 1), representing the modeled event or class, whether a player will perform Above Average (1) or Below Average (0) in a given game based

---

### Correlation Analisys

One hot encoding uses drop first colunn to avoid the dummy variable trap and avoid perfect multicollinearity in linear models caused by this encoding.

### Correlation and heatmap for Continuous Variables

Continuous Variables (minutes_played, fg_pct, three_pct, ft_pct.) show near 0 correlation between them.

### Correlation and heatmap for Continuous Variables in data sets 1 and 2

Variable pairs correlations (efficiency,minutes played) = 0.81,(efficiency,points) = 0.88 and  (points,minutes played) = 0.78 shows strong possitive correlation.
Variable pairs correlations (rebounds,possition) = -0.55 and (blocks,possition) = -0.40 shows moderate negative correlation.
Variable pair correlation (age,minutes played) has weak correlation near 0.

The target is moderate correlated possitivelly with minutes playes, efficiency and points.

## Plot suggested pairs and choosen pairs to engineer

(age,minutes played)
(efficiency,minutes played)
(efficiency,points)
(points,minutes played)


## Feature Engineering

Age–Minutes Relationship Proposed Features

1. Age-Adjusted Minutes (Residuals)
2. Minutes per Age Ratio
5. Combined Age-Usage Profile
6. Minutes Percentile Within Age Group

Efficiency–Minutes Relationship Proposed Features

1. Efficiency Per Minute
2. Total contribution
3. High-intensity player flag 

Efficiency–Points Relationship Proposed Features

1. Efficiency Per Point
2. Scoring Impact
3. High-Efficiency Scorer Flag

Points-Minutes Relationship Proposed Features

1. Points Per Minute
2. Scoring Volume
3. High Usage Scorer Flag

## Preprocessing

Abbreviations for Preprocessing Methods:

Standardize - Std
Normalize - Nor
Binning Standard - Binning Std
Binning Quantile X - Binning Quant X 
One-Hot Encoding - OHE
Frequency Encoding - Freq Enc
No preprocessing - None

| **Variable Name**   | **Variable** | Unique  | **Data Set** | **Data Set** | **Data Set**| **Data Set** | **Data Set** | **Data Set**  |
|                     | **Type**     | Values  | **1**        | **2**        | **3**       | **4**        | **5**        | **6**         |
|---------------------|--------------|---------|--------------|--------------|-------------|--------------|------------- |---------------|
| **Identifier Variables** |         |         |              |              |             |              |              |               |
| row_id              | Identifier   | 7200    | None         | None         | None        | None         | None         | None          |
| player_id           | Identifier   | 500     | None         | None         | None        | None         | None         | None          |
| **Categorical Variables**|         |         |              |              |             |              |              |               |
| position            | Nominal      | 5       | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| team                | Nominal      | 30      | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| opponent            | Nominal      | 30      | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| game_location       | Nominal      | 2       | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| **New Categorical Variables** |    |         |              |              |             |              |              |               |
| high_usage_scorer   | Nominal      | 2       | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| high_eff_min        | Nominal      | 2       | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| high_eff_scorer     | Nominal      | 2       | Freq Enc     | Freq Enc     | Freq Enc    | Freq Enc     | OHE          | OHE           |
| **Ordinal Variables**    |         |         |              |              |             |              |              |               |
| rest_days           | Ordinal      | 6       | Freq Enc     | Freq Enc     | None        | None         | OHE          | OHE           |
| **Continuous Variables** |         |         |              |              |             |              |              |               |
| minutes_played      | Continuous   | 341     | Std          | Norm         | Bin Std 5   | Bin Quant 5  | OHE Bin Std  | OHE Bin Quant |
| fg_pct              | Continuous   | 344     | Std          | Norm         | Bin Std 8   | Bin Quant 8  | OHE Bin Std  | OHE Bin Quant |
| three_pct           | Continuous   | 431     | Std          | Norm         | Bin Std 5   | Bin Quant 5  | OHE Bin Std  | OHE Bin Quant |
| ft_pct              | Continuous   | 396     | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| **New Continuous Variables** |     |         |              |              |             |              |              |               |
| eff_per_point       | Continuous   | 296     | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| eff_per_min         | Continuous   | 2758    | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| points_per_min      | Continuous   | 2303    | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| scoring_impact      | Continuous   | 369     | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| eff_times_minutes   | Continuous   | 3106    | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| scoring_volume      | Continuous   | 2636    | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| **Discrete Variables**   |         |         |              |              |             |              |              |               |
| age                 | Discrete     | 20      | Std          | Norm         | Bin Std 9   | Bin Quant 9  | OHE Bin Std  | OHE Bin Quant |
| plus_minus          | Discrete     | 55      | Std          | Norm         | Bin Std 7   | Bin Quant 7  | OHE Bin Std  | OHE Bin Quant |
| efficiency          | Discrete     | 45      | Std          | Norm         | Bin Std 21  | Bin Quant 21 | OHE Bin Std  | OHE Bin Quant |
| points              | Discrete     | 31      | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std  | OHE Bin Quant |
| rebounds            | Discrete     | 18      | Std          | Norm         | Bin Std 7   | Bin Quant 7  | OHE Bin Std  | OHE Bin Quant |
| assists             | Discrete     | 14      | Std          | Norm         | Bin Std 4   | Bin Quant 4  | OHE Bin Std  | OHE Bin Quant |
| steals              | Discrete     | 4       | Std          | Norm         | None        | None         | OHE          | OHE           |
| blocks              | Discrete     | 4       | Std          | Norm         | None        | None         | OHE          | OHE           |
| turnovers           | Discrete     | 6       | Std          | Norm         | None        | None         | OHE          | OHE           |
| **Target Variable**      |         |         |              |              |             |              |              |               |
| target              | Binary Class | 2       | None         | None         | None        | None         | None         | None          |

## Feature Importance - Feature Selection

Random forest classifier for ds1, ds3, and ds5

| Feature          | ds1 | ds2 | ds3 | ds4 | class   | ds5 | ds6 |
|------------------|:---:|:---:|:---:|:---:|:------: |:---:|:---:|
| eff_per_min      |  X  |  X  |     |     | 6       |  X  |  X  |
| efficency        |  X  |  X  |  X  |  X  | 10      |  X  |  X  |
| scoring_impact   |  X  |  X  |  X  |  X  | 2, 3    |  X  |  X  |
| eff_per_point    |  X  |  X  |  X  |  X  | 2       |  X  |  X  |
| eff_time_minutes |  X  |  X  |  X  |  X  | 4, 2, 3 |  X  |  X  |
| plus_minus       |  X  |  X  |  X  |  X  |         |     |     |
| turnovers        |  X  |  X  |  X  |  X  |         |     |     |
| scoring_volume   |  X  |  X  |  X  |  X  |         |     |     |
| opponent         |     |     |  X  |  X  |         |     |     |
| team             |     |     |  X  |  X  |         |     |     |
| high_eff_min     |     |     |     |     | 0.7963  |  X  |  X  |
| high_eff_scorer  |     |     |     |     | 0.5361  |  X  |  X  |
| assit            |     |     |     |     | 4       |  X  |  X  |
| PCA              |  X  |  X  |  X  |  X  |         |     |     | 

<!--
 Recursive Feature Elimination (RFE)
-->
## Training Datasets

Six datasets with the most important features selected based on the feature importances computed above, and four PCA-reduced datasets.

## model training