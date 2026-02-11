"""Main pipeline entry point for the regime-flow system."""

import numpy as np
import pandas as pd
import warnings
# Filter specific warnings to avoid hiding important issues
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from src.data.loader import DataLoader
from src.models.hmm import RegimeDetector
from src.models.lstm import LSTMForecaster, LSTMTrainer


def main():
    """
    Execute the regime-flow pipeline.
    
    Pipeline steps:
    1. Load SPY data using DataLoader
    2. Calculate log returns and volatility
    3. Detect market regimes using HMM
    4. Display regime statistics
    5. Initialize LSTM forecaster for future predictions
    """
    print("=" * 60)
    print("REGIME-FLOW: Market Regime Detection & Forecasting Pipeline")
    print("=" * 60)
    print()
    
    # Step 1: Load Data
    print("Step 1: Loading SPY data...")
    loader = DataLoader(ticker="SPY", start_date="2020-01-01")
    data = loader.fetch_data()
    print(f"Loaded data shape: {data.shape}")
    print()
    
    # Step 2: Prepare Features
    print("Step 2: Calculating features (log returns & volatility)...")
    features_df = loader.prepare_features(window=20)
    features_df['Close'] = loader.data['Close']
    print(f"Features shape: {features_df.shape}")
    print(f"Features preview:\n{features_df.head()}")
    print()
    
    # Step 3: Detect Regimes
    print("Step 3: Detecting market regimes using HMM...")
    detector = RegimeDetector(n_states=2, random_state=42)
    features_array = features_df.values
    detector.fit(features_array, n_iter=100)
    print()
    
    # Step 4: Get Regime Predictions
    print("Step 4: Predicting regimes...")
    regimes = detector.decode_regimes(features_array)
    features_df['regime'] = regimes.values
    
    print(f"Regime distribution:")
    print(regimes.value_counts())
    print()
    
    # Step 5: Display Regime Statistics
    print("Step 5: Regime Statistics:")
    stats = detector.get_regime_statistics(features_array)
    print(stats.to_string(index=False))
    print()
    
    # Step 6: Train LSTM Model (The Neural Network)
    print("\nStep 6: Training LSTM Forecaster (Option 2)...")
    
    # 1. Setup the data for the Neural Net
    from src.models.lstm import LSTMForecaster
    import torch
    
    # Create sequences (History -> Target)
    X, y = [], []
    window_size = 20
    data_values = features_df[['log_returns', 'volatility']].values
    
    for i in range(len(data_values) - window_size):
        X.append(data_values[i : i+window_size]) # The 20-day window
        y.append(data_values[i+window_size, 0])  # The Next Day's Return (Target)
        
    # FIX 2: Convert to numpy array first (Much faster)
    X_tensor = torch.FloatTensor(np.array(X))
    y_tensor = torch.FloatTensor(np.array(y)).view(-1, 1)
    
    # 2. Train the Model
    model = LSTMForecaster(input_size=2, hidden_size=64, num_layers=2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training neural network...")
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        output, _ = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # 3. Predict Tomorrow
    last_window = torch.FloatTensor(data_values[-window_size:]).unsqueeze(0)
    prediction, _ = model(last_window)
    predicted_return = prediction.item()
    
    print("\n============================================================")
    print(f"LSTM PREDICTION (Neural Net):")
    print(f"Predicted Return: {predicted_return:.4%}")
    if predicted_return > 0:
        print(">>> Model Predicts: UP 🟢")
    else:
        print(">>> Model Predicts: DOWN 🔴")
    print("============================================================")
    
    # Step 7: Save Results
    print("Step 7: Saving results...")
    output_file = "regime_analysis_results.csv"
    features_df.to_csv(output_file)
    print(f"Results saved to: {output_file}")
    print()
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    # Display sample of final results
    print("\nSample of results (last 10 days):")
    print(features_df.tail(10).to_string())
    

if __name__ == "__main__":
    main()
