import os
import sys
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.draw_fig import draw_fig
from utils.preprocessing import Preprocessing
from utils.metrics import similarity, nmae, rmse, r2, fa2, fb, fsd

# Constants
METHOD = "SVR"
DATA = "SVR"
SAVE_RESULTS = True

if SAVE_RESULTS:
    # Create directories
    folder_path = f'outputs/predicts/data_{DATA}/{METHOD}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Load data
dataframe = pd.read_csv(f'data/processed/{DATA}/30/CAF003_imputed.csv')

# Preprocess data
processed_df = pd.DataFrame()
for col in dataframe.columns:
    univariate_raw_data = dataframe[[col]]
    # Preprocessing data
    preprocessing = Preprocessing()
    preprocessed_data = preprocessing.flow(univariate_raw_data)
    univariate_processed_data = preprocessed_data[col].to_numpy()
    processed_df[col] = univariate_processed_data
df = deepcopy(processed_df)

# Prepare the data
X = df.drop('VW_30cm', axis=1)  # Features (all columns except VW_30cm)
y = df['VW_30cm']  # Target variable

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (SVR works better with scaled features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVR model
svr_model = SVR(
    kernel='rbf',  # Radial Basis Function kernel
    C=100,         # Regularization parameter
    gamma='scale', # Kernel coefficient
    epsilon=0.1    # Epsilon in the epsilon-SVR model
)

# Fit the model
svr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svr_model.predict(X_test_scaled)

# Calculate metrics
metrics = {
    'Similarity': similarity(y_test, y_pred),
    'NMAE': nmae(y_test, y_pred),
    'R2': r2(y_test, y_pred),
    'RMSE': rmse(y_test, y_pred),
    'FSD': fsd(y_test, y_pred),
    'FB': fb(y_test, y_pred),
    'FA2': fa2(y_test, y_pred)
}
# Save metrics to CSV
metrics_df = pd.DataFrame([metrics])
print("Metrics:")
print(metrics_df)
if SAVE_RESULTS:
    metrics_df.to_csv(f'{folder_path}/metrics.csv', index=False)

# Save predictions to CSV for further analysis
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
if SAVE_RESULTS:
    predictions_df.to_csv(f'{folder_path}/predictions.csv', index=False)
    draw_fig(Actual=predictions_df["Actual"], Predicted=predictions_df["Predicted"], title="SVR Predictions", save_path=f'{folder_path}/predictions.png', is_show_fig=False)