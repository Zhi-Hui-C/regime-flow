"""Data loader for fetching and preprocessing financial data."""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Optional
from datetime import datetime


class DataLoader:
    """
    Fetches stock data using yfinance and calculates log returns and volatility.
    
    Attributes:
        ticker (str): The stock ticker symbol.
        start_date (str): Start date for data retrieval.
        end_date (str): End date for data retrieval.
        data (pd.DataFrame): Raw stock data.
    """
    
    def __init__(self, ticker: str = "SPY", start_date: str = "2020-01-01", 
                 end_date: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            ticker: Stock ticker symbol (default: "SPY").
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format (default: today).
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data: Optional[pd.DataFrame] = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        """
        print(f"Fetching {self.ticker} data from {self.start_date} to {self.end_date}...")
            
        # ADDED: auto_adjust=True ensures we get clean data
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
            
        # ADDED: If yfinance gives us a MultiIndex (e.g. Price, Ticker), flatten it
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = self.data.columns.get_level_values(0)

        print(f"Fetched {len(self.data)} records.")
        return self.data    
    
    def calculate_log_returns(self) -> pd.Series:
        """
        Calculate log returns from close prices.
        """
        if self.data is None:
            self.fetch_data()
        
        # CHANGED: 'Adj Close' -> 'Close'
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1))
        return log_returns.dropna()
    
    def calculate_volatility(self, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility using a specified window.
        
        Args:
            window: Rolling window size in days (default: 20).
            
        Returns:
            Series of rolling volatility.
        """
        log_returns = self.calculate_log_returns()
        volatility = log_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility.dropna()
    
    def prepare_features(self, window: int = 20) -> pd.DataFrame:
        """
        Prepare feature matrix for regime detection.
        
        Args:
            window: Rolling window size for volatility calculation.
            
        Returns:
            DataFrame with log returns and volatility features.
        """
        log_returns = self.calculate_log_returns()
        volatility = self.calculate_volatility(window=window)
        
        # Align the data
        features = pd.DataFrame({
            'log_returns': log_returns,
            'volatility': volatility
        }).dropna()
        
        return features
