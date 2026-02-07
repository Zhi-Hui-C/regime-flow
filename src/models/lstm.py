"""LSTM neural network for time series forecasting."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LSTMForecaster(nn.Module):
    """
    PyTorch LSTM module for financial time series forecasting.
    
    Attributes:
        input_size (int): Number of input features.
        hidden_size (int): Number of LSTM hidden units.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout probability.
        output_size (int): Number of output features.
    """
    
    def __init__(
        self, 
        input_size: int = 2, 
        hidden_size: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize the LSTM forecaster.
        
        Args:
            input_size: Number of input features (default: 2).
            hidden_size: Number of hidden units in LSTM layers (default: 64).
            num_layers: Number of stacked LSTM layers (default: 2).
            dropout: Dropout probability for regularization (default: 0.2).
            output_size: Number of output features (default: 1).
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            hidden: Optional tuple of (h_0, c_0) hidden states.
            
        Returns:
            Tuple of (output, (h_n, c_n)) where:
                - output: Predictions of shape (batch_size, output_size).
                - (h_n, c_n): Final hidden and cell states.
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(last_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.
        
        Args:
            batch_size: Size of the batch.
            device: Device to create tensors on (CPU or CUDA).
            
        Returns:
            Tuple of (h_0, c_0) initialized hidden states.
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)


class LSTMTrainer:
    """
    Trainer class for LSTM model with standard training loop.
    """
    
    def __init__(
        self,
        model: LSTMForecaster,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the LSTM trainer.
        
        Args:
            model: LSTMForecaster model to train.
            learning_rate: Learning rate for optimizer (default: 0.001).
            device: Device to train on (default: auto-detect).
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader with training data.
            
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(batch_x)
            
            # Compute loss
            loss = self.criterion(output, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader with validation data.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output, _ = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            x: Input array of shape (n_samples, sequence_length, input_size).
            
        Returns:
            Predictions as numpy array.
        """
        self.model.eval()
        
        x_tensor = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(x_tensor)
        
        return output.cpu().numpy()
