import os
import sys
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.draw_fig import draw_fig
from utils.preprocessing import Preprocessing
from utils.metrics import similarity, nmae, rmse, r2, fa2, fb, fsd

# Constants
METHOD = "AdaBoost"
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

# Train the AdaBoost model
base_estimator = DecisionTreeRegressor(max_depth=4)
adaboost_model = AdaBoostRegressor(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.001,
    random_state=42
)

# Fit the model
adaboost_model.fit(X_train, y_train)

# Make predictions
y_pred = adaboost_model.predict(X_test)

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
    draw_fig(Actual=predictions_df["Actual"], Predicted=predictions_df["Predicted"], title="AdaBoost Predictions", save_path=f'{folder_path}/predictions.png', is_show_fig=False)