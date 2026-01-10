# EDA Report

##  Column overview
*(See summary statistics table below for data types, ranges, and missing values.)*

---
| Column          | Type       | Unique Values | Missing | % Missing | Min   | Max   | Mode |
|-----------------|------------|---------------|---------|-----------|-------|-------|------|
| row_id          | int64      | 7200          | 0       | 0.0       | 2.0   | 8998.0| 6318 |
| player_id       | int64      | 500           | 0       | 0.0       | 1000.0| 1499.0| 1110 |
| age             | int64      | 20            | 0       | 0.0       | 19.0  | 39.0  | 27   |
| position        | object     | 5             | 0       | 0.0       |       |       | SG   |
| team            | object     | 30            | 0       | 0.0       |       |       | NYK  |
| opponent        | object     | 30            | 0       | 0.0       |       |       | GSW  |
| minutes_played  | float64    | 341           | 0       | 0.0       | 7.2   | 43.0  | 24.0 |
| points          | int64      | 31            | 0       | 0.0       | 0.0   | 30.0  | 13   |
| rebounds        | int64      | 18            | 0       | 0.0       | 0.0   | 17.0  | 5    |
| assists         | int64      | 14            | 0       | 0.0       | 0.0   | 13.0  | 2    |
| steals          | int64      | 4             | 0       | 0.0       | 0.0   | 3.0   | 1    |
| blocks          | int64      | 4             | 0       | 0.0       | 0.0   | 3.0   | 0    |
| turnovers       | int64      | 6             | 0       | 0.0       | 0.0   | 5.0   | 2    |
| fg_pct          | float64    | 344           | 0       | 0.0       | 30.0  | 69.7  | 47.4 |
| three_pct       | float64    | 431           | 0       | 0.0       | 10.0  | 60.0  | 32.9 |
| ft_pct          | float64    | 396           | 0       | 0.0       | 45.0  | 95.0  | 95.0 |
| plus_minus      | int64      | 55            | 0       | 0.0       | -26.0 | 30.0  | -2   |
| efficiency      | int64      | 45            | 0       | 0.0       | 1.0   | 45.0  | 23   |
| game_location   | object     | 2             | 0       | 0.0       |       |       | Away |
| rest_days       | int64      | 6             | 0       | 0.0       | 0.0   | 5.0   | 1    |
| target          | int64      | 2             | 0       | 0.0       | 0.0   | 1.0   | 0    |

---

### Identifier Variables
These variables uniquely identify records and entities but do not carry predictive or analytical meaning by themselves.

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

### Variable Analysis

Variables will be analyzed according to the groups defined above, based on their statistical properties.

- **Identifier variables** will be excluded from statistical analysis and modeling, as they serve only as unique references.
- **Categorical variables** will be examined through frequency distributions and class balance, with attention to potential high-cardinality effects.
- **Numerical continuous variables** will be analyzed using summary statistics, distribution plots, and outlier detection methods.
- **Numerical discrete variables** will be evaluated using count distributions and comparative statistics across categories.
- **The binary target variable** will be assessed for class balance and its relationship with key explanatory variables.

Abbreviations for Preprocessing Methods:

Standardize - Std
Normalize - Nor
Binning Standard - Binning Std
Binning Quantile X - Binning Quant X 
One-Hot Encoding - OHE
Frequency Encoding - Freq Enc
No preprocessing - None

| **Variable Name**   | **Variable** | Unique  | **Data Set** | **Data Set** | **Data Set**| **Data Set** | **Data Set**    | **Data Set**      |
|                     | **Type**     | Values  | **1**        | **2**        | **3**       | **4**        | **5**           | **6**             |
|---------------------|--------------|---------|--------------|--------------|-------------|--------------|-----------------|-------------------|
| **Identifier Variables** |         |         |              |              |             |              |                 |                   |
| row_id              | Identifier   | 7200    | None         | None         | None        | None         | None            | None              |
| player_id           | Identifier   | 500     | None         | None         | None        | None         | None            | None              |
| **Categorical Variables**|         |         |              |              |             |              |                 |                   |
| position            | Nominal      | 5       | Freq Enc     | Freq Enc     | None        | None         | OHE PCA         | OHE PCA           |
| team                | Nominal      | 30      | Freq Enc     | Freq Enc     | None        | None         | OHE PCA         | OHE PCA           |
| opponent            | Nominal      | 30      | Freq Enc     | Freq Enc     | None        | None         | OHE PCA         | OHE PCA           |
| game_location       | Nominal      | 2       | Freq Enc     | Freq Enc     | None        | None         | OHE PCA         | OHE PCA           |
| **Ordinal Variables**    |         |         |              |              |             |              |                 |                   |
| rest_days           | Ordinal      | 6       | Freq Enc     | Freq Enc     | None        | None         | OHE PCA         | OHE PCA           |
| **Continuous Variables** |         |         |              |              |             |              |                 |                   |
| minutes_played      | Continuous   | 341     | Std          | Norm         | Bin Std 5   | Bin Quant 5  | OHE Bin Std PCA | OHE Bin Quant PCA |
| fg_pct              | Continuous   | 344     | Std          | Norm         | Bin Std 8   | Bin Quant 8  | OHE Bin Std PCA | OHE Bin Quant PCA |
| three_pct           | Continuous   | 431     | Std          | Norm         | Bin Std 5   | Bin Quant 5  | OHE Bin Std PCA | OHE Bin Quant PCA |
| ft_pct              | Continuous   | 396     | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std PCA | OHE Bin Quant PCA |
| **Discrete Variables**   |         |         |              |              |             |              |                 |                   |
| age                 | Discrete     | 20      | Std          | Norm         | Bin Std 9   | Bin Quant 9  | OHE Bin Std PCA | OHE Bin Quant PCA |
| plus_minus          | Discrete     | 55      | Std          | Norm         | Bin Std 7   | Bin Quant 7  | OHE Bin Std PCA | OHE Bin Quant PCA |
| efficiency          | Discrete     | 45      | Std          | Norm         | Bin Std 21  | Bin Quant 21 | OHE Bin Std PCA | OHE Bin Quant PCA |
| points              | Discrete     | 31      | Std          | Norm         | Bin Std 6   | Bin Quant 6  | OHE Bin Std PCA | OHE Bin Quant PCA |
| rebounds            | Discrete     | 18      | Std          | Norm         | Bin Std 7   | Bin Quant 7  | OHE Bin Std PCA | OHE Bin Quant PCA |
| assists             | Discrete     | 14      | Std          | Norm         | Bin Std 4   | Bin Quant 4  | OHE Bin Std PCA | OHE Bin Quant PCA |
| steals              | Discrete     | 4       | Std          | Norm         | None        | None         | OHE PCA         | OHE PCA           |
| blocks              | Discrete     | 4       | Std          | Norm         | None        | None         | OHE PCA         | OHE PCA           |
| turnovers           | Discrete     | 6       | Std          | Norm         | None        | None         | OHE PCA         | OHE PCA           |
| **Target Variable**      |         |         |              |              |             |              |                 |                   |
| target              | Binary       | 2       | None         | None         | None        | None         | None            | None              |



