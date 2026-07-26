# 🏀 NBA Player Performance Prediction

## 📌 Project Overview

This project predicts an NBA player's **Game Score (GmSc)** using Machine Learning techniques.

The project demonstrates a complete Machine Learning lifecycle including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Comparison
- MLflow Experiment Tracking
- MLflow Model Registry
- Flask REST API for inference
- Docker containerization

---

# 📂 Project Structure

```
NBA-Player-Performance-Prediction
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
│
├── data
│   ├── raw
│   │     nba_stats.csv
│   └── processed
│         nba_processed.csv
│
├── models
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── notebooks
│   └── EDA.ipynb
│
├── src
│   ├── preprocessing.py
│   └── train.py
│
└── mlruns
```

---

# 📊 Dataset

Dataset: **NBA Player Statistics**

The dataset contains player game statistics including:

- Minutes Played
- Field Goals
- Field Goal Attempts
- 3 Point Statistics
- Free Throws
- Rebounds
- Assists
- Steals
- Blocks
- Turnovers
- Fouls
- Team Information
- Opponent Information

Target Variable:

**Game Score (GmSc)**

---

# ⚙ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Flask
- Joblib
- Docker
- Git & GitHub

---

# 🔍 Exploratory Data Analysis

EDA includes:

- Dataset inspection
- Missing value checking
- Data types
- Summary statistics
- Correlation analysis
- Feature distribution

Notebook:

```
notebooks/EDA.ipynb
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline performs:

- Removing unnecessary columns
- Label Encoding categorical variables
- Date conversion
- Feature selection
- Train/Test split
- Feature Scaling using StandardScaler

Processed dataset is stored in:

```
data/processed/nba_processed.csv
```

---

# 🤖 Machine Learning Models

Three regression models were trained and compared.

### 1. Linear Regression

Baseline model.

### 2. Decision Tree Regressor

Captures non-linear relationships.

### 3. Random Forest Regressor

Ensemble model used for better prediction accuracy.

---

# 📈 Model Evaluation

Models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The model with the highest R² score is selected as the final model.

---

# 📊 MLflow Experiment Tracking

MLflow is used to:

- Track experiments
- Store evaluation metrics
- Save trained models
- Compare multiple runs
- Register the best model

Tracked metrics include:

- MAE
- MSE
- RMSE
- R² Score

---

# 📦 Model Registry

The best performing model is registered in the MLflow Model Registry as:

```
NBA_Player_Performance_Model
```

Multiple model versions can be managed and deployed.

---

# 🌐 Flask API

The trained model is deployed as a REST API.

Run:

```bash
python app.py
```

API Endpoint

```
POST /predict
```

Example JSON Request

```json
{
  "Player":372,
  "Tm":13,
  "Opp":9,
  "MP":40.98,
  "FG":17,
  "FGA":27,
  "FG%":0.63,
  "3P":2,
  "3PA":10,
  "3P%":0.20,
  "FT":7,
  "FTA":10,
  "FT%":0.70,
  "ORB":0,
  "DRB":12,
  "TRB":12,
  "AST":9,
  "STL":2,
  "BLK":1,
  "TOV":3,
  "PF":1,
  "Date":0
}
```

Example Response

```json
{
    "Predicted_GmSc": 42.92
}
```

---

# 🐳 Docker

Build the Docker image

```bash
docker build -t nba-player-performance .
```

Run the container

```bash
docker run -p 5001:5001 nba-player-performance
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Bhav615/NBA-Player-Performance-Prediction.git
```

Move into the project

```bash
cd NBA-Player-Performance-Prediction
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Run Project

Preprocess data

```bash
python src/preprocessing.py
```

Train model

```bash
python src/train.py
```

Start Flask API

```bash
python app.py
```

Launch MLflow UI

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

---

# 📌 Results

✔ Data preprocessing completed

✔ Multiple regression models trained

✔ Best model selected automatically

✔ Experiment tracking using MLflow

✔ Model Registry implemented

✔ Flask API deployed

✔ Docker support added

---

# 👨‍💻 Author

**Bhavadeeswar Reddy P**

Master's in Data Science & Artificial Intelligence

SRH University Hamburg

GitHub:

https://github.com/Bhav615
