# regime-flow

A regime-aware algorithmic trading system that uses Hidden Markov Models (HMM) to detect market volatility and dynamically switch between specialized LSTM agents.

## Overview

**regime-flow** is a quantitative finance project that combines statistical modeling and deep learning to analyze and forecast financial markets. The system detects market regimes (Bull/Bear markets) using Hidden Markov Models and leverages LSTM neural networks for time series forecasting.

## Key Features

- **Data Loading**: Automated fetching of historical stock data (SPY) using `yfinance`
- **Feature Engineering**: Calculation of log returns and rolling volatility
- **Regime Detection**: Hidden Markov Model (HMM) to identify Bull and Bear market states
- **Time Series Forecasting**: PyTorch LSTM module for predictive modeling
- **Type-Safe**: Comprehensive type hints throughout the codebase
- **Well-Documented**: Detailed docstrings for all classes and methods

## Project Structure

```
regime-flow/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # DataLoader class for fetching & preprocessing
│   └── models/
│       ├── __init__.py
│       ├── hmm.py             # RegimeDetector using GaussianHMM
│       └── lstm.py            # LSTMForecaster with PyTorch
├── main.py                     # Pipeline entry point
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Zhi-Hui-C/regime-flow.git
cd regime-flow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main pipeline:
```bash
python main.py
```

The pipeline will:
1. Fetch SPY historical data from Yahoo Finance
2. Calculate log returns and volatility features
3. Train a Hidden Markov Model to detect market regimes
4. Classify each time period as Bull or Bear market
5. Display regime statistics and save results to CSV

## HMM Strategy Explained

### What is a Hidden Markov Model?

A Hidden Markov Model (HMM) is a statistical model that assumes the system being modeled is a Markov process with hidden (unobservable) states. In financial markets, we can observe returns and volatility, but the underlying market regime (Bull/Bear) is hidden.

### How It Works

1. **Observable Features**: 
   - Log returns: `log(P_t / P_{t-1})`
   - Rolling volatility: Standard deviation of returns over a 20-day window (annualized)

2. **Hidden States**:
   - **Bull Market**: Characterized by positive mean returns and lower volatility
   - **Bear Market**: Characterized by negative mean returns and higher volatility

3. **Model Training**:
   - The HMM learns the probability of transitioning between states
   - It estimates emission probabilities (likelihood of observing certain features in each state)
   - Uses the Gaussian distribution to model feature distributions in each regime

4. **Regime Detection**:
   - The Viterbi algorithm finds the most likely sequence of hidden states
   - States are labeled as Bull/Bear based on their mean return characteristics

### Why Use HMM for Regime Detection?

- **Probabilistic Framework**: Provides confidence scores for regime predictions
- **Captures Regime Persistence**: Models the tendency of markets to stay in the same regime
- **Temporal Dependencies**: Accounts for the sequential nature of financial data
- **Adaptability**: Can detect regime changes without predefined rules

### Trading Applications

- **Risk Management**: Adjust position sizing based on detected regime
- **Strategy Switching**: Use different trading strategies for Bull vs Bear markets
- **Dynamic Asset Allocation**: Rebalance portfolio based on regime predictions
- **Volatility Forecasting**: Predict future volatility based on current regime

## LSTM Forecasting Module

The project includes a PyTorch LSTM module designed for time series forecasting:

- **Multi-layer LSTM**: Stacked LSTM layers for complex pattern recognition
- **Dropout Regularization**: Prevents overfitting on financial data
- **Flexible Architecture**: Configurable input/output dimensions and hidden units
- **Training Utilities**: Built-in trainer class with standard training loop

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **yfinance**: Financial data retrieval from Yahoo Finance
- **hmmlearn**: Hidden Markov Models implementation
- **torch**: PyTorch for deep learning (LSTM)

## Example Output

```
REGIME-FLOW: Market Regime Detection & Forecasting Pipeline
============================================================

Step 1: Loading SPY data...
Fetched 1258 records.

Step 2: Calculating features (log returns & volatility)...
Features shape: (1237, 2)

Step 3: Detecting market regimes using HMM...
Training HMM with 2 states...
Training complete. Regime names: {0: 'Bull', 1: 'Bear'}

Step 4: Predicting regimes...
Regime distribution:
Bull    1089
Bear     148

Step 5: Regime Statistics:
  regime  count  mean_return  std_return  mean_volatility
    Bull   1089     0.000612    0.007123         0.183421
    Bear    148    -0.001234    0.025678         0.412356
```

## Future Enhancements

- Real-time data streaming
- Multiple asset support
- Backtesting framework
- Advanced LSTM training with regime-specific models
- Portfolio optimization based on regime predictions
- Web dashboard for visualization

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Author

Zhi-Hui-C

## Acknowledgments

- Yahoo Finance for providing free financial data
- hmmlearn library for HMM implementation
- PyTorch team for the deep learning framework
