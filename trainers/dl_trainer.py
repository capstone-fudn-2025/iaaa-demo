import numpy as np
import torch
from copy import deepcopy
from trainers.base_trainer import BaseTrainer

class DLTrainer(BaseTrainer):
    """
    Trainer for deep learning models (PyTorch).
    """
    def __init__(self, window_size=7, batch_size=32, num_epochs=50, learning_rate=0.001, weight_decay=1e-5, min_epochs=10, patience=5):
        """
        Initialize the DL trainer with parameters.
        
        Args:
            window_size (int): Size of the sliding window for prediction models.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.
            weight_decay (float): Weight decay for optimizer.
            min_epochs (int): Minimum number of epochs before early stopping.
            patience (int): Number of epochs to wait before early stopping.
        """
        super().__init__(window_size)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_epochs = min_epochs
        self.patience = patience
        
    def train_forward(self, data, model):
        """
        Train a PyTorch model in the forward direction.
        
        Args:
            data (np.ndarray): The training data.
            model: The PyTorch model to train.
            
        Returns:
            model: The trained model.
        """
        # Create windowed data
        X, y = self.create_windows(data)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # Create a deep copy of the model to avoid modifying the original
        model_copy = deepcopy(model)
        model_copy = model_copy.to(self.device)
        
        # Define loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        best_loss = float('inf')
        patience_counter = 0
        
        # Train the model
        model_copy.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            
            for inputs, targets in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model_copy(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
            
            # Early stopping
            if epoch+1 >= self.min_epochs:
                epoch_loss = running_loss / len(dataloader)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return model_copy
    
    def train_backward(self, data, model):
        """
        Train a PyTorch model in the backward direction.
        
        Args:
            data (np.ndarray): The training data.
            model: The PyTorch model to train.
            
        Returns:
            model: The trained model.
        """
        # Reverse the data for backward training
        reversed_data = data[::-1].copy()
        
        # Create windowed data
        X, y = self.create_windows(reversed_data)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # Create a deep copy of the model to avoid modifying the original
        model_copy = deepcopy(model)
        model_copy = model_copy.to(self.device)
        
        # Define loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        best_loss = float('inf')
        patience_counter = 0
        
        # Train the model
        model_copy.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            
            for inputs, targets in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model_copy(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
            
            # Early stopping
            if epoch+1 >= self.min_epochs:
                epoch_loss = running_loss / len(dataloader)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        return model_copy
    
    def predict_forward(self, data_before_gap, gap_size, model):
        """
        Predict the gap using a forward PyTorch model.
        
        Args:
            data_before_gap (np.ndarray): Data before the gap.
            gap_size (int): Size of the gap to predict.
            model: The trained PyTorch model.
            
        Returns:
            np.ndarray: Predicted values for the gap.
        """
        model.eval()  # Set model to evaluation mode
        predictions = []
        current_window = data_before_gap[-self.window_size:]
        
        with torch.no_grad():  # No need to track gradients
            for _ in range(gap_size):
                # Convert to tensor for prediction
                X_pred = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Make prediction
                pred = model(X_pred).cpu().item()
                predictions.append(pred)
                
                # Update window for next prediction
                current_window = np.append(current_window[1:], pred)
        
        return np.array(predictions)
    
    def predict_backward(self, data_after_gap, gap_size, model):
        """
        Predict the gap using a backward PyTorch model.
        
        Args:
            data_after_gap (np.ndarray): Data after the gap.
            gap_size (int): Size of the gap to predict.
            model: The trained PyTorch model.
            
        Returns:
            np.ndarray: Predicted values for the gap (in correct order).
        """
        model.eval()  # Set model to evaluation mode
        # The model was trained on reversed data, so we need to reverse data_after_gap
        reversed_data = data_after_gap[::-1].copy()
        
        predictions = []
        current_window = reversed_data[:self.window_size].copy()
        
        with torch.no_grad():  # No need to track gradients
            for _ in range(gap_size):
                # Convert to tensor for prediction
                X_pred = torch.FloatTensor(current_window).unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Make prediction
                pred = model(X_pred).cpu().item()
                predictions.append(pred)
                
                # Update window for next prediction
                current_window = np.append(current_window[1:], pred)
        
        # Reverse the predictions to get the correct order
        return np.array(predictions[::-1])