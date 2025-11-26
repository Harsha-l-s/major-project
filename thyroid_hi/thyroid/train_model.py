import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('data/hypothyroid.csv')

# Simple preprocessing example
df = df.dropna()
cat_cols = df.select_dtypes(include=['object']).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Selecting features and target
target_col = 'target'  # Change to actual target column name
X = df.drop(columns=[target_col])
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'models/model_rf.pkl')
joblib.dump(encoders, 'models/encoders.pkl')
joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')

print('Model training completed and saved!')
