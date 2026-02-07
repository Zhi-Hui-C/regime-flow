"""Hidden Markov Model for regime detection."""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict, Optional, Tuple


class RegimeDetector:
    """
    Detects market regimes (Bull/Bear) using Gaussian Hidden Markov Models.
    
    Attributes:
        n_states (int): Number of hidden states (default: 2 for Bull/Bear).
        model: Trained GaussianHMM model.
        regime_names (Dict): Mapping of state indices to regime names.
    """
    
    def __init__(self, n_states: int = 2, random_state: int = 42):
        """
        Initialize the RegimeDetector.
        
        Args:
            n_states: Number of hidden states (default: 2 for Bull/Bear).
            random_state: Random seed for reproducibility.
        """
        self.n_states = n_states
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.regime_names: Dict[int, str] = {}
        
    def fit(self, features: np.ndarray, n_iter: int = 100) -> 'RegimeDetector':
        """
        Fit the HMM model to the features.
        
        Args:
            features: Feature matrix (n_samples, n_features). First column should contain
                     returns for proper regime labeling (Bull/Bear assignment).
            n_iter: Maximum number of iterations for training.
            
        Returns:
            Self for method chaining.
        """
        print(f"Training HMM with {self.n_states} states...")
        
        # Initialize and train the model
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=self.random_state
        )
        
        self.model.fit(features)
        
        # Assign regime names based on mean returns
        hidden_states = self.model.predict(features)
        mean_returns = []
        
        for state in range(self.n_states):
            state_mask = hidden_states == state
            state_returns = features[state_mask, 0]  # Assuming first column is returns
            mean_returns.append(state_returns.mean())
        
        # Sort states by mean returns (descending)
        sorted_states = np.argsort(mean_returns)[::-1]
        
        # Assign names
        regime_labels = ["Bull", "Bear"] if self.n_states == 2 else [f"Regime_{i}" for i in range(self.n_states)]
        self.regime_names = {sorted_states[i]: regime_labels[i] for i in range(self.n_states)}
        
        print(f"Training complete. Regime names: {self.regime_names}")
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict hidden states for given features.
        
        Args:
            features: Feature matrix (n_samples, n_features).
            
        Returns:
            Array of predicted state indices.
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities for each state.
        
        Args:
            features: Feature matrix (n_samples, n_features).
            
        Returns:
            Array of probabilities (n_samples, n_states).
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        return self.model.predict_proba(features)
    
    def decode_regimes(self, features: np.ndarray) -> pd.Series:
        """
        Decode hidden states to regime names.
        
        Args:
            features: Feature matrix (n_samples, n_features).
            
        Returns:
            Series with regime names.
        """
        states = self.predict(features)
        regimes = pd.Series([self.regime_names[state] for state in states])
        return regimes
    
    def get_regime_statistics(self, features: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Compute statistics for each detected regime.
        
        Args:
            features: Feature matrix (n_samples, n_features). Expected to have returns in
                     column 0 and volatility in column 1 (if available).
            dates: Optional datetime index for the features.
            
        Returns:
            DataFrame with regime statistics.
        """
        states = self.predict(features)
        regimes = self.decode_regimes(features)
        
        stats = []
        for state in range(self.n_states):
            state_mask = states == state
            regime_name = self.regime_names[state]
            
            state_data = features[state_mask]
            
            stats.append({
                'regime': regime_name,
                'count': state_mask.sum(),
                'mean_return': state_data[:, 0].mean(),
                'std_return': state_data[:, 0].std(),
                'mean_volatility': state_data[:, 1].mean() if state_data.shape[1] > 1 else None
            })
        
        return pd.DataFrame(stats)
