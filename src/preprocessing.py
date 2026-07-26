import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("data/raw/nba_stats.csv")

print("Original Shape:", df.shape)

# Remove unwanted columns
df.drop(columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"], inplace=True)

# Fill missing numeric values
numeric_columns = df.select_dtypes(include=np.number).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Convert MP from MM:SS to decimal minutes
def convert_minutes(mp):
    if isinstance(mp, str) and ":" in mp:
        minutes, seconds = mp.split(":")
        return int(minutes) + int(seconds) / 60
    return np.nan

df["MP"] = df["MP"].apply(convert_minutes)
df["MP"] = df["MP"].fillna(df["MP"].mean())

# Encode categorical columns
categorical_columns = ["Player", "Tm", "Opp", "Result", "Date"]

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

# Define features and target
X = df.drop("PTS", axis=1)
y = df["PTS"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed dataset
df.to_csv("data/processed/nba_processed.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)
print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)