import os
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
# Remove StandardScaler import since we're not using it

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.draw_fig import draw_fig
from utils.preprocessing import Preprocessing
from utils.metrics import similarity, nmae, rmse, r2, fa2, fb, fsd

# Constants
METHOD = "Attention"
DATA = "SVR"
SAVE_RESULTS = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

class Attention(nn.Module):
    """
    Attention-based neural network for time series imputation.
    """
    @property
    def name(self):
        return "Attention"
    
    def __init__(self, input_size, hidden_channels=64):
        """
        Initialize the Attention model.
        
        Args:
            input_size (int): Size of the input features.
            hidden_channels (int): Number of hidden channels.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        
        # Embedding layer
        self.embedding = nn.Linear(input_size, hidden_channels)
        self.attention = nn.MultiheadAttention(hidden_channels, num_heads=1)
        self.fc_1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc_2 = nn.Linear(hidden_channels, 1)
            
    def forward(self, x):
        """
        Forward pass for the Attention model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size].
        """
        # [batch_size, input_size] -> [batch_size, hidden_channels]
        x = self.embedding(x)
        # Reshape for attention: [batch_size, 1, hidden_channels]
        x = x.unsqueeze(1)
        # Apply self-attention (requires shape [seq_len, batch_size, hidden_channels])
        x = x.transpose(0, 1)  # [1, batch_size, hidden_channels]
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)  # [batch_size, 1, hidden_channels]
        x = x.squeeze(1)  # [batch_size, hidden_channels]
        x = self.fc_1(x)
        x = nn.ReLU()(x)
        x = self.fc_2(x)
        return x.squeeze(-1)
    
    def get_model(self):
        """
        Get a copy of the model for training.
        
        Returns:
            Attention: A copy of the model.
        """
        return deepcopy(self)

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

# Convert DataFrames to numpy arrays
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values

# Convert numpy arrays to PyTorch tensors directly (without scaling)
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the Attention model
input_size = X_train_np.shape[1]
model = Attention(input_size=input_size, hidden_channels=64).to(DEVICE)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
print(f"Training Attention model on {DEVICE}...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.cpu().numpy()

# Calculate metrics
metrics = {
    'Similarity': similarity(y_test_np, y_pred),
    'NMAE': nmae(y_test_np, y_pred),
    'R2': r2(y_test_np, y_pred),
    'RMSE': rmse(y_test_np, y_pred),
    'FSD': fsd(y_test_np, y_pred),
    'FB': fb(y_test_np, y_pred),
    'FA2': fa2(y_test_np, y_pred)
}
# Save metrics to CSV
metrics_df = pd.DataFrame([metrics])
print("Metrics:")
print(metrics_df)
if SAVE_RESULTS:
    metrics_df.to_csv(f'{folder_path}/metrics.csv', index=False)

# Save predictions to CSV for further analysis
predictions_df = pd.DataFrame({
    'Actual': y_test_np,
    'Predicted': y_pred
})
if SAVE_RESULTS:
    predictions_df.to_csv(f'{folder_path}/predictions.csv', index=False)
    draw_fig(Actual=predictions_df["Actual"], Predicted=predictions_df["Predicted"], title="Attention Model Predictions", save_path=f'{folder_path}/predictions.png', is_show_fig=False)

# Save model
if SAVE_RESULTS:
    torch.save(model.state_dict(), f'{folder_path}/model.pt')