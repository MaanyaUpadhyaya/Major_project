import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset
dataset_path = "dataset.csv"
df = pd.read_csv(dataset_path)

# Separate features (symptoms) and target variable (diseases)
X = df.iloc[:, 1:]  # Features (symptoms)
y = df.iloc[:, 0]   # Target variable (diseases)

# 1. Convert diseases to numerical using one-hot encoding
y_encoded = pd.get_dummies(y, prefix='disease')

# 2. Concatenate one-hot encoded diseases with symptoms
df_encoded = pd.concat([y_encoded, X], axis=1)

# 3. Create a binary column indicating whether the disease is present or not
df_encoded['is_disease_present'] = df_encoded.iloc[:, 0:].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

# 4. Save the final DataFrame to a new CSV file
final_csv_path = "final_dataset.csv"
df_encoded.to_csv(final_csv_path, index=False)

# 5. Data Splitting (if needed)
X_train, X_test, y_train, y_test = train_test_split(X, df_encoded['is_disease_present'], test_size=0.2, random_state=42)

# 6. Feature Scaling (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
