import pandas as pd
import numpy as np
import joblib #used to train the ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../dataset/traffic.csv')#reads the csv file and loads it into memory

# remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# 🔥 FIX infinity & NaN values
# infinite values caused by division-by-zero in flow-based features were handled by replacing them with NaN and removing affected records.”
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True) 

#this was done because the dataset required preprocessing ie had yo be made more clean and organised to be used for machine learning
# Convert labels to binary
df['BinaryLabel'] = df['Label'].apply(
    lambda x: 0 if x == 'BENIGN' else 1
)
#If it’s BENIGN → normal → 0
#Anything else → attack → 1

# 4. Separate features and target
X = df.drop(['Label', 'BinaryLabel'], axis=1)
# Save feature names
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

y = df['BinaryLabel']

# 5. Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'ids_model.pkl')


