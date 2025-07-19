'''
This is the main file where the code should be run from. This file contains the StatArb class which is the main class that 
controls the running of the code.
'''

# First starting with the necessary imports
import sys
import os
import yfinance as yf
import time
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from Data.data_fetch import DataFetch
from Pair_Identification.PairSelection import Pairs
from Features.Labelling import FeatureLabelling  
from Features.Labelling import Combiner
from Features.Feature_Importance import FeatureImportance 
from ML_Models import LogisticRegressionClassifier, Neural_Network
from ML_Model_Trainer import Purged_K_CV
from Backtest import Backtest, PortfolioBacktest

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)

from scipy.stats import linregress

class StatArb:

    """
    A class to run the statistical arbitrage strategy. 
    This class orchestrates the entire process from data loading, pair identification,
    feature engineering, model training, and backtesting.

    Attributes:

        file_name : str
            Path to the CSV file containing pricing data.
        pricing : pd.DataFrame
            DataFrame holding the raw pricing data after processing.
        returns : pd.DataFrame
            DataFrame holding percentage returns computed from pricing data.
        tickers_list : list
            List of unique tickers found in the dataset.
        pairs : list
            List of tuples containing cointegrated stock pairs.
        normalized_df : pd.DataFrame
            DataFrame holding the normalized features for model training.

    Methods:

        load_data(): Loads the data from the specified file and identifies pairs for analysis.
        pairs_dataframes(): Creates DataFrames for each identified pair with relevant features.
        get_technical_indicators(pair_df, stock1, stock2): Generates technical indicators for the given pair DataFrame.
        normalize_features(pairs, globals_dict): Normalizes features for the identified pairs using rolling normalization.
        feature_importance_analysis(pairs, globals_dict):Computes feature importance for the identified pairs using XGBoost.
        train_model(pairs, globals_dict): Trains a machine learning model on the normalized features of the pairs.
        run_base_model(): Runs the base model training and evaluation process.
        Run_NN(): Runs the neural network model training and evaluation process.
        Combine_models(): Combines the predictions from multiple models for improved accuracy.
        run_backtest(): Runs the backtest on the trained model predictions.


    """


    def __init__(self, file_name: str):
        """
        Initialize the StatArb class with the given file name where our data is stored.
        """
        self.file_name = file_name
        self.pricing = None
        self.returns = None
        self.tickers_list = None
        self.pairs = None
        self.normalized_df = None
    
    def load_data(self):
        """
        Calls the pairs method which loads the data and runs the pairs method to find 
        our pairs for analysis.
        """
        pairs = Pairs(self.file_name)
        pairs.Run_pairs()
        self.pricing = pairs.pricing
        self.returns = pairs.returns
        self.pairs = pairs.pairs

    def pairs_dataframes(self):
        """
        Uses the loaded data and pairs to create dataframes for each pair.
        """
    
        pricing = self.pricing
        returns = self.returns.reindex(self.pricing.index)
        print("pricing Index Range:", pricing.index.min(), "-", pricing.index.max())

        for pair in self.pairs:
            stock1 = pair[0].replace(' ', '_')
            stock2 = pair[1].replace(' ', '_')

            pair_df = pd.DataFrame({
                stock1: self.pricing[pair[0]],
                stock2: self.pricing[pair[1]],
                f'{stock1}_returns': returns[pair[0]],
                f'{stock2}_returns': returns[pair[1]],
                f'{stock1}_log_returns': np.log(self.pricing[pair[0]] / self.pricing[pair[0]].shift(1)),
                f'{stock2}_log_returns': np.log(self.pricing[pair[1]] / self.pricing[pair[1]].shift(1)),
                f'{stock1}_20std': returns[pair[0]].rolling(window=20).std(),
                f'{stock2}_20std': returns[pair[1]].rolling(window=20).std()
            })

            pair_df.index = pd.to_datetime(self.pricing.index)  # Ensure the index is datetime
            pair_df = pair_df.dropna().sort_index()  # Drop NaN values and sort by index (date) in ascending order
            pair_df = self.get_technical_indicators(pair_df, stock1, stock2)  # This calls the method to generate technical indicators
            # pair_df = pair_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            globals()[f'{stock1}_{stock2}_total_df'] = pair_df
            print(f'{stock1}_{stock2}_total_df created successfully.')

    def get_technical_indicators(self, pair_df: pd.DataFrame, stock1: str, stock2: str) -> pd.DataFrame:
        """
        This method creates a focused set of reliable technical indicators using pricing data
        with robust handling of NaN and extreme values.
        """
        # Import required libraries
        import pandas as pd
        import numpy as np
        from scipy.stats import linregress
        
        # Define simpler window sizes
        short_windows = [3, 4, 5, 7, 10, 12, 20, 24]
        medium_windows = [30, 50, 60]
        long_windows = [100, 200, 400]
        all_windows = short_windows + medium_windows + long_windows
        
        # Basic spread calculation
        spread = pair_df[stock1] - pair_df[stock2]
        pair_df['spread'] = spread                                                                                              

        # Z-Score (simplified and robust)
        spread = np.log(pair_df[stock1]) - np.log(pair_df[stock2])
        pair_df['spread'] = spread
        pair_df['spread_mean'] = spread.rolling(window=5).mean() # Change window sizes for different result.
        pair_df['spread_std'] = spread.rolling(window=5).std()
        pair_df['z_score'] = (spread - pair_df['spread_mean']) / pair_df['spread_std']
        pair_df['z_score'] = pair_df['z_score'].bfill()
        z_score = pair_df['z_score']
        pair_df['z_score_returns'] = np.log(z_score) - np.log(z_score.shift(1))
        pair_df['z_score_returns'] = pair_df['z_score_returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Simple Moving Averages for both stocks
        for window in short_windows + medium_windows + long_windows:
            pair_df[f'{stock1}_sma_{window}'] = pair_df[stock1].rolling(window=window).mean().fillna(pair_df[stock1])
            pair_df[f'{stock2}_sma_{window}'] = pair_df[stock2].rolling(window=window).mean().fillna(pair_df[stock2])
            pair_df[f'spread_sma_{window}'] = pair_df['spread'].rolling(window=window).mean().fillna(pair_df['spread'])
        
        # Simple Volatility (Standard Deviation)
        for window in short_windows + medium_windows:
            pair_df[f'{stock1}_vol_{window}'] = pair_df[stock1].rolling(window=window).std().fillna(0)
            pair_df[f'{stock2}_vol_{window}'] = pair_df[stock2].rolling(window=window).std().fillna(0)
            pair_df[f'spread_vol_{window}'] = pair_df['spread'].rolling(window=window).std().fillna(0)
        
        # Simple returns (no extreme transforms)
        pair_df[f'{stock1}_return'] = pair_df[stock1].pct_change().fillna(0)
        pair_df[f'{stock2}_return'] = pair_df[stock2].pct_change().fillna(0)
        
        # Clip extreme return values
        pair_df[f'{stock1}_return'] = np.clip(pair_df[f'{stock1}_return'], -0.2, 0.2)
        pair_df[f'{stock2}_return'] = np.clip(pair_df[f'{stock2}_return'], -0.2, 0.2)
        
        # Basic momentum indicators (price changes over different periods)
        for window in [3, 5, 10, 20]:
            pair_df[f'{stock1}_momentum_{window}'] = (pair_df[stock1] - pair_df[stock1].shift(window)).fillna(0)
            pair_df[f'{stock2}_momentum_{window}'] = (pair_df[stock2] - pair_df[stock2].shift(window)).fillna(0)
            pair_df[f'spread_momentum_{window}'] = (pair_df['spread'] - pair_df['spread'].shift(window)).fillna(0)

        # Calculate z-scores for all window sizes
        for window in all_windows:
            # Calculate mean and standard deviation of the spread over the window
            pair_df[f'spread_mean_{window}'] = pair_df['spread'].rolling(window=window).mean()
            pair_df[f'spread_std_{window}'] = pair_df['spread'].rolling(window=window).std()
            
            # Calculate z-score for this window
            pair_df[f'z_score_{window}'] = (pair_df['spread'] - pair_df[f'spread_mean_{window}']) / pair_df[f'spread_std_{window}']
            
            # Backfill any NaN values
            pair_df[f'z_score_{window}'] = pair_df[f'z_score_{window}'].bfill()
            
            # Optional: Calculate z-score returns (log differences)
            pair_df[f'z_score_{window}_returns'] = np.log(pair_df[f'z_score_{window}']) - np.log(pair_df[f'z_score_{window}'].shift(1))
            pair_df[f'z_score_{window}_returns'] = pair_df[f'z_score_{window}_returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Simple RSI calculation
        def calculate_rsi(series, window=14):
            delta = series.diff().fillna(0)
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            avg_gain = gains.rolling(window=window).mean().fillna(0)
            avg_loss = losses.rolling(window=window).mean().fillna(0)
            
            # Avoid division by zero
            rs = np.where(avg_loss == 0, 100, avg_gain / np.maximum(avg_loss, 0.0001))
            rsi = 100 - (100 / (1 + rs))
            return np.clip(rsi, 0, 100)  # Ensure RSI is in the valid range
        
        pair_df[f'{stock1}_rsi'] = calculate_rsi(pair_df[stock1])
        pair_df[f'{stock2}_rsi'] = calculate_rsi(pair_df[stock2])
        
        # Half-life of mean reversion (simplified)
        def half_life(spread, window=100):
            if len(spread) < window:
                return 30  # Default value
            
            # Get just the last window values
            spread_window = spread[-window:].dropna()
            
            if len(spread_window) < 2:
                return 30
            
            spread_lag = spread_window.shift(1).iloc[1:]
            spread_diff = spread_window.diff().iloc[1:]
            
            if len(spread_lag) < 2:
                return 30
            
            try:
                beta = linregress(spread_lag, spread_diff)[0]
                if beta < 0:
                    hl = -np.log(2) / beta
                    return np.clip(hl, 1, 100)  # Clip to reasonable range
                else:
                    return 30  # Default for non-mean-reverting series
            except:
                return 30  # Default value in case of calculation errors
        
        # Calculate half-life once for the whole series
        pair_df['half_life'] = half_life(pair_df['spread'])
        
        # Bollinger Bands (simplified)
        pair_df['upper_band'] = pair_df['spread_mean'] + (2 * pair_df['spread_std'])
        pair_df['lower_band'] = pair_df['spread_mean'] - (2 * pair_df['spread_std'])
        
        # Simple overbought/oversold indicators
        pair_df['z_score_overbought'] = np.where(pair_df['z_score'] > 2, 1, 0)
        pair_df['z_score_oversold'] = np.where(pair_df['z_score'] < -2, 1, 0)
        
        # Create a few simple ratio features
        pair_df['stock_ratio'] = pair_df[stock1] / np.maximum(pair_df[stock2], 0.0001)
        
        # Price relative to moving averages
        for window in [20, 50]:
            pair_df[f'{stock1}_rel_to_ma_{window}'] = pair_df[stock1] / np.maximum(pair_df[f'{stock1}_sma_{window}'], 0.0001) - 1
            pair_df[f'{stock2}_rel_to_ma_{window}'] = pair_df[stock2] / np.maximum(pair_df[f'{stock2}_sma_{window}'], 0.0001) - 1
            pair_df[f'spread_rel_to_ma_{window}'] = pair_df['spread'] / np.maximum(pair_df[f'spread_sma_{window}'], 0.0001) - 1
        
        from scipy import stats

        # Cointegration features
        # Rolling cointegration test
        windows = [30, 60, 90]
        for window in windows:
            pair_df[f'cointegration_pval_{window}'] = np.nan
            
            # Rolling calculation of cointegration p-value
            for i in range(window, len(pair_df)):
                s1 = pair_df[stock1].iloc[i-window:i]
                s2 = pair_df[stock2].iloc[i-window:i]
                try:
                    result = stats.coint(s1, s2)
                    pair_df.loc[pair_df.index[i], f'cointegration_pval_{window}'] = result[1]
                except:
                    pass
                    
            # Fill NaN values
            pair_df[f'cointegration_pval_{window}'] = pair_df[f'cointegration_pval_{window}'].fillna(method='ffill').fillna(0.5)
            
            # Binary signal for strong cointegration (p < 0.05)
            pair_df[f'strong_cointegration_{window}'] = (pair_df[f'cointegration_pval_{window}'] < 0.05).astype(int)
        
        # Z-score cross features
        # Tracking when z-score crosses significant thresholds
        thresholds = [-2, -1, 0, 1, 2]
        for i in range(len(thresholds)-1):
            lower, upper = thresholds[i], thresholds[i+1]
            pair_df[f'z_in_range_{lower}_{upper}'] = ((pair_df['z_score'] > lower) & 
                                                    (pair_df['z_score'] <= upper)).astype(int)
        
        # Z-score regime changes
        pair_df['z_regime_change'] = (pair_df['z_score'].shift(1) * pair_df['z_score'] < 0).astype(int)
        pair_df['z_sign'] = np.sign(pair_df['z_score']).astype(int)
        
        # Spread Acceleration (2nd derivative)
        for window in [5, 10, 20]:
            # First calculate momentum (1st derivative)
            momentum = pair_df[f'spread_momentum_{window}']
            # Then calculate acceleration (2nd derivative)
            pair_df[f'spread_acceleration_{window}'] = momentum.diff(window).fillna(0)
            
            # Normalize acceleration relative to volatility
            spread_vol = np.maximum(pair_df[f'spread_vol_{window}'], 0.0001)
            pair_df[f'spread_acceleration_norm_{window}'] = pair_df[f'spread_acceleration_{window}'] / spread_vol
        
        # Exponential features of z-score (enhances sensitivity to extreme values)
        pair_df['z_score_squared'] = pair_df['z_score'] ** 2
        pair_df['z_score_cubed'] = pair_df['z_score'] ** 3
        pair_df['z_score_exp'] = np.exp(np.clip(pair_df['z_score'], -5, 5)) / np.exp(5)
        
        # Convergence velocity - how fast z-score is returning to mean
        pair_df['z_convergence_velocity'] = -pair_df['z_score'] * pair_df['z_score'].diff().fillna(0)
        pair_df['z_convergence_velocity'] = np.where(
            pair_df['z_convergence_velocity'] > 0, 
            pair_df['z_convergence_velocity'], 
            0
        )
        
        # Divergence velocity - how fast z-score is moving away from mean
        pair_df['z_divergence_velocity'] = pair_df['z_score'] * pair_df['z_score'].diff().fillna(0)
        pair_df['z_divergence_velocity'] = np.where(
            pair_df['z_divergence_velocity'] > 0, 
            pair_df['z_divergence_velocity'], 
            0
        )
        
        # Correlation features
        for window in [10, 20, 30, 60]:
            # Rolling correlation between stocks
            pair_df[f'correlation_{window}'] = pair_df[stock1].rolling(window).corr(pair_df[stock2]).fillna(0)
            
            # Correlation change
            pair_df[f'correlation_change_{window}'] = pair_df[f'correlation_{window}'].diff(5).fillna(0)
            
            # Correlation * z-score interaction feature
            pair_df[f'corr_z_interaction_{window}'] = pair_df[f'correlation_{window}'] * pair_df['z_score']
        
        # Beta features (systematic risk)
        for window in [20, 60]:
            # Calculate rolling beta
            cov = pair_df[stock1].rolling(window).cov(pair_df[stock2]).fillna(0)
            var = pair_df[stock2].rolling(window).var().fillna(0.0001)
            pair_df[f'beta_{window}'] = cov / var
            
            # Beta stability (lower is more stable)
            pair_df[f'beta_stability_{window}'] = pair_df[f'beta_{window}'].rolling(window).std().fillna(0)
        
        # Volatility ratio features
        for window in [10, 20, 50]:
            vol1 = np.maximum(pair_df[f'{stock1}_vol_{window}'], 0.0001)
            vol2 = np.maximum(pair_df[f'{stock2}_vol_{window}'], 0.0001)
            
            # Ratio of volatilities
            pair_df[f'vol_ratio_{window}'] = vol1 / vol2
            
            # Normalized volatilities
            pair_df[f'{stock1}_norm_vol_{window}'] = vol1 / np.maximum(pair_df[stock1], 0.0001)
            pair_df[f'{stock2}_norm_vol_{window}'] = vol2 / np.maximum(pair_df[stock2], 0.0001)
            
            # Volatility spread
            pair_df[f'vol_spread_{window}'] = vol1 - vol2
        
        # Trend strength indicators
        for window in [20, 50]:
            # ADX-like indicator for trend strength (simplified)
            price_diff1 = pair_df[stock1].diff().fillna(0)
            price_diff2 = pair_df[stock2].diff().fillna(0)
            
            up_move1 = np.maximum(price_diff1, 0)
            down_move1 = np.abs(np.minimum(price_diff1, 0))
            
            up_move2 = np.maximum(price_diff2, 0)
            down_move2 = np.abs(np.minimum(price_diff2, 0))
            
            avg_up1 = up_move1.rolling(window).mean().fillna(0)
            avg_down1 = down_move1.rolling(window).mean().fillna(0)
            
            avg_up2 = up_move2.rolling(window).mean().fillna(0)
            avg_down2 = down_move2.rolling(window).mean().fillna(0)
            
            # Trend strength as ratio of directional movement
            pair_df[f'{stock1}_trend_strength_{window}'] = np.abs(avg_up1 - avg_down1) / (avg_up1 + avg_down1 + 0.0001)
            pair_df[f'{stock2}_trend_strength_{window}'] = np.abs(avg_up2 - avg_down2) / (avg_up2 + avg_down2 + 0.0001)
            
            # Spread trend strength
            spread_diff = pair_df['spread'].diff().fillna(0)
            up_move_spread = np.maximum(spread_diff, 0)
            down_move_spread = np.abs(np.minimum(spread_diff, 0))
            avg_up_spread = up_move_spread.rolling(window).mean().fillna(0)
            avg_down_spread = down_move_spread.rolling(window).mean().fillna(0)
            
            pair_df[f'spread_trend_strength_{window}'] = np.abs(avg_up_spread - avg_down_spread) / (avg_up_spread + avg_down_spread + 0.0001)
        
        # Seasonality features (time-based patterns)
        if isinstance(pair_df.index, pd.DatetimeIndex):
            # Day of week
            pair_df['day_of_week'] = pair_df.index.dayofweek
            
            # Month
            pair_df['month'] = pair_df.index.month
            
            # Quarter
            pair_df['quarter'] = pair_df.index.quarter
            
            # Create cyclical features using sine and cosine transformations
            # Day of week (cycle of 5 for trading days)
            pair_df['day_sin'] = np.sin(2 * np.pi * pair_df['day_of_week'] / 5)
            pair_df['day_cos'] = np.cos(2 * np.pi * pair_df['day_of_week'] / 5)
            
            # Month (cycle of 12)
            pair_df['month_sin'] = np.sin(2 * np.pi * pair_df['month'] / 12)
            pair_df['month_cos'] = np.cos(2 * np.pi * pair_df['month'] / 12)
        
        # Feature interactions
        # Z-score * volatility interaction
        for window in [20, 50]:
            pair_df[f'z_vol_interaction_{window}'] = pair_df['z_score'] * pair_df[f'spread_vol_{window}']
        
        # Z-score historical extremes features
        for window in [60, 120]:
            pair_df[f'z_historic_max_{window}'] = pair_df['z_score'].rolling(window).max().fillna(0)
            pair_df[f'z_historic_min_{window}'] = pair_df['z_score'].rolling(window).min().fillna(0)
            pair_df[f'z_historic_range_{window}'] = pair_df[f'z_historic_max_{window}'] - pair_df[f'z_historic_min_{window}']
            
            # Z-score percentile in historical window
            z_rank = pair_df['z_score'].rolling(window).rank(pct=True).fillna(0.5)
            pair_df[f'z_percentile_{window}'] = z_rank
            
            # Distance from historical extremes
            pair_df[f'z_dist_from_max_{window}'] = pair_df[f'z_historic_max_{window}'] - pair_df['z_score']
            pair_df[f'z_dist_from_min_{window}'] = pair_df['z_score'] - pair_df[f'z_historic_min_{window}']
        
        # Stationarity features
        for window in [30, 60]:
            # Calculate rolling mean and std of first differences (approximation of stationarity)
            diff_mean = pair_df['spread'].diff().rolling(window).mean().fillna(0)
            diff_std = pair_df['spread'].diff().rolling(window).std().fillna(0.0001)
            
            # Coefficient of variation of the differenced series (lower values suggest more stationarity)
            pair_df[f'spread_stationarity_{window}'] = np.abs(diff_mean) / diff_std
        
        # Mean reversion probability estimated from historical behavior
        for window in [60, 120]:
            # Get sign changes in z-score
            z_sign_change = (pair_df['z_score'].shift(1) * pair_df['z_score'] <= 0).astype(int)
            
            # Mean reversion probability as frequency of sign changes
            pair_df[f'mean_reversion_prob_{window}'] = z_sign_change.rolling(window).mean().fillna(0.5)
            
            # Expected time to mean reversion based on historical behavior
            crossover_idx = pair_df.index[z_sign_change == 1]
            
            # Initialize with default value
            pair_df[f'expected_reversion_time_{window}'] = window / 2
            
            # Calculate for each point
            if len(crossover_idx) > 1:
                for i in range(1, len(crossover_idx)):
                    duration = (crossover_idx[i] - crossover_idx[i-1]).days if hasattr(crossover_idx[i], 'days') else 1
                    # Apply to all points between these crossovers
                    if i < len(crossover_idx) - 1:
                        between_idx = pair_df.index[(pair_df.index > crossover_idx[i]) & 
                                                (pair_df.index <= crossover_idx[i+1])]
                        if len(between_idx) > 0:
                            pair_df.loc[between_idx, f'expected_reversion_time_{window}'] = duration
                
            # Ensure this feature is non-negative and reasonably bounded
            pair_df[f'expected_reversion_time_{window}'] = np.clip(
                pair_df[f'expected_reversion_time_{window}'].fillna(window/2), 
                1, 
                window
            )


         # Z-score transformations (power transformations)
        power_transforms = [0.5, 1.5, 2.5]
        for power in power_transforms:
            # Sign-preserving power transform
            pair_df[f'z_score_pow_{power}'] = np.sign(pair_df['z_score']) * np.abs(pair_df['z_score']) ** power
        
        # Z-score moving features
        for window in [3, 5, 10]:
            # Z-score momentum (rate of change)
            pair_df[f'z_score_momentum_{window}'] = pair_df['z_score'] - pair_df['z_score'].shift(window)
            
            # Z-score acceleration (momentum of momentum)
            pair_df[f'z_score_acceleration_{window}'] = pair_df[f'z_score_momentum_{window}'] - pair_df[f'z_score_momentum_{window}'].shift(window)
            
            # Z-score jerk (rate of change of acceleration)
            pair_df[f'z_score_jerk_{window}'] = pair_df[f'z_score_acceleration_{window}'] - pair_df[f'z_score_acceleration_{window}'].shift(window)
            
            # Fill NaN values
            for col in [f'z_score_momentum_{window}', f'z_score_acceleration_{window}', f'z_score_jerk_{window}']:
                pair_df[col] = pair_df[col].fillna(0)
        
        # ===== Spread Envelope Features =====
        
        # Calculate various quantiles of the spread for different windows
        for window in [30, 60, 120]:
            # Calculate various quantiles to create an envelope
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99]
            for q in quantiles:
                pair_df[f'spread_q{int(q*100)}_{window}'] = pair_df['spread'].rolling(window).quantile(q).fillna(method='ffill').fillna(method='ffill')
            
            # Calculate relative position within the envelope
            low_q = pair_df[f'spread_q1_{window}']
            high_q = pair_df[f'spread_q99_{window}']
            range_q = high_q - low_q + 1e-10  # Avoid division by zero
            pair_df[f'spread_envelope_position_{window}'] = (pair_df['spread'] - low_q) / range_q
            
            # Extreme value indicators
            pair_df[f'spread_is_extreme_high_{window}'] = (pair_df['spread'] > pair_df[f'spread_q95_{window}']).astype(int)
            pair_df[f'spread_is_extreme_low_{window}'] = (pair_df['spread'] < pair_df[f'spread_q5_{window}']).astype(int)
        
        # ===== Non-linear Transformations =====
        
        # Apply sigmoid transformation to z-score for probabilistic interpretation
        pair_df['z_score_sigmoid'] = 1 / (1 + np.exp(-pair_df['z_score']))
        
        # Log-based features (safely handling positive and negative values)
        for col in ['spread', 'z_score']:
            # Apply sign-preserving log transformation
            pair_df[f'{col}_log'] = np.sign(pair_df[col]) * np.log1p(np.abs(pair_df[col]))
        
        # Hyperbolic transformations for z-score
        pair_df['z_score_tanh'] = np.tanh(pair_df['z_score'])
        pair_df['z_score_arctan'] = np.arctan(pair_df['z_score']) * 2 / np.pi  # Scaled to [-1, 1]
        
        # ===== Cross-Asset Feature Interactions =====
        
        # Ratio momentum features
        for window in [5, 10, 20]:
            # Calculate ratio momentum (differential asset performance)
            ratio = pair_df[stock1] / (pair_df[stock2] + 1e-10)
            pair_df[f'ratio_momentum_{window}'] = (ratio / ratio.shift(window) - 1).fillna(0)
            
            # Clip extreme values
            pair_df[f'ratio_momentum_{window}'] = np.clip(pair_df[f'ratio_momentum_{window}'], -0.5, 0.5)
        
        # Performance disparity features
        for window in [5, 10, 20]:
            # Calculate normalized returns
            ret1 = pair_df[f'{stock1}_return'].rolling(window).sum().fillna(0)
            ret2 = pair_df[f'{stock2}_return'].rolling(window).sum().fillna(0)
            
            # Performance differential (which asset is outperforming)
            pair_df[f'perf_differential_{window}'] = ret1 - ret2
            
            # Relative performance (ratio of performances)
            pair_df[f'rel_performance_{window}'] = (1 + ret1) / (1 + ret2 + 1e-10) - 1
        
        # Calculate mean reversion velocity considering z-score level
        for window in [5, 10, 20]:
            # Traditional mean reversion would expect negative relationship between
            # z-score level and subsequent change (high z-score -> decrease)
            z_level = pair_df['z_score'].shift(window)
            z_change = pair_df['z_score'] - z_level
            
            # Mean reversion score (negative = mean reverting, positive = trending away)
            pair_df[f'mean_reversion_score_{window}'] = (z_level * z_change).fillna(0)
            
            # Mean reversion strength (higher = stronger mean reversion)
            pair_df[f'mean_reversion_strength_{window}'] = -np.clip(pair_df[f'mean_reversion_score_{window}'], -10, 0)
            
            # Trend continuation strength (higher = stronger trend)
            pair_df[f'trend_continuation_strength_{window}'] = np.clip(pair_df[f'mean_reversion_score_{window}'], 0, 10)
        
        # Create autoregressive features for spread and z-score
        for target in ['spread', 'z_score']:
            # AR features at different lags
            for lag in [1, 2, 3, 5, 10]:
                pair_df[f'{target}_lag_{lag}'] = pair_df[target].shift(lag).fillna(method='ffill')
                
            # Multiple lag differences (capturing short-term dynamics)
            for lag1, lag2 in [(1, 3), (1, 5), (3, 10)]:
                col_name = f'{target}_diff_{lag1}_{lag2}'
                pair_df[col_name] = pair_df[f'{target}_lag_{lag1}'] - pair_df[f'{target}_lag_{lag2}']
        
        # Calculate higher moments for both stocks and spread
        for col in [stock1, stock2, 'spread']:
            for window in [20, 50, 100]:
                # Skewness (asymmetry of the distribution)
                pair_df[f'{col}_skew_{window}'] = pair_df[col].rolling(window).skew().fillna(0)
                
                # Kurtosis (heavy-tailedness of the distribution)
                pair_df[f'{col}_kurt_{window}'] = pair_df[col].rolling(window).kurt().fillna(0)
                
                # Clip extreme values for numerical stability
                pair_df[f'{col}_skew_{window}'] = np.clip(pair_df[f'{col}_skew_{window}'], -10, 10)
                pair_df[f'{col}_kurt_{window}'] = np.clip(pair_df[f'{col}_kurt_{window}'], -10, 10)
        
        # ===== Volatility Dynamics =====
        
        # Enhanced volatility features
        for window in [10, 20, 50]:
            # Volatility of volatility (vol of vol)
            for asset in [stock1, stock2, 'spread']:
                vol_col = f'{asset}_vol_{window}'
                pair_df[f'{vol_col}_of_vol'] = pair_df[vol_col].rolling(window).std().fillna(0)
            
            # Volatility ratio stability
            vol_ratio_col = f'vol_ratio_{window}'
            if vol_ratio_col in pair_df.columns:
                pair_df[f'{vol_ratio_col}_stability'] = pair_df[vol_ratio_col].rolling(window).std().fillna(0)
            
            # Volatility regime change detector
            for asset in [stock1, stock2, 'spread']:
                vol_col = f'{asset}_vol_{window}'
                # Compare current volatility to historical
                vol_avg = pair_df[vol_col].rolling(window*2).mean().fillna(pair_df[vol_col])
                pair_df[f'{asset}_vol_regime_{window}'] = pair_df[vol_col] / (vol_avg + 1e-10) - 1
        
        # ===== Complex Feature Interactions =====
        
        # Z-score / volatility interaction matrix
        z_windows = [20, 50]
        vol_windows = [10, 30]
        
        for z_window in z_windows:
            for vol_window in vol_windows:
                # Z-score volatility ratio
                z_col = f'z_score_{z_window}' if f'z_score_{z_window}' in pair_df.columns else 'z_score'
                vol_col = f'spread_vol_{vol_window}'
                
                # Z-score scaled by volatility (volatility adjusted z-score)
                pair_df[f'z_vol_scaled_{z_window}_{vol_window}'] = pair_df[z_col] / (pair_df[vol_col] + 1e-10)
                
                # Z-score / volatility interaction features
                pair_df[f'z_vol_interact_{z_window}_{vol_window}'] = pair_df[z_col] * pair_df[vol_col]
        
        # ===== Regime Detection Features =====
        
        # HMM-like state detection using z-score and volatility
        for window in [30, 60]:
            # Compute z-score histogram features for regime detection
            hist_bins = [-float('inf'), -2, -1, 0, 1, 2, float('inf')]
            
            for i in range(len(hist_bins)-1):
                lower, upper = hist_bins[i], hist_bins[i+1]
                if lower == -float('inf'):
                    bin_name = f'z_lt_{upper}_{window}'
                elif upper == float('inf'):
                    bin_name = f'z_gt_{lower}_{window}'
                else:
                    bin_name = f'z_{lower}_to_{upper}_{window}'
                
                # Calculate percentage of time z-score spent in this bin
                in_bin = ((pair_df['z_score'] > lower) & (pair_df['z_score'] <= upper)).astype(float)
                pair_df[bin_name] = in_bin.rolling(window).mean().fillna(0)
            
            # Detect regime shifts using histogram changes
            hist_cols = [col for col in pair_df.columns if col.startswith(f'z_') and col.endswith(f'_{window}')]
            if hist_cols:
                # Calculate Euclidean distance between current histogram and past histogram
                current_hist = pair_df[hist_cols].values
                past_hist = pair_df[hist_cols].shift(window//2).fillna(method='ffill').values
                
                # Calculate row-wise Euclidean distance
                distances = np.sqrt(np.sum((current_hist - past_hist)**2, axis=1))
                pair_df[f'regime_shift_magnitude_{window}'] = distances
                
                # Normalize and detect significant shifts
                mean_dist = np.nanmean(distances)
                std_dist = np.nanstd(distances) if np.nanstd(distances) > 0 else 1
                pair_df[f'regime_shift_zscore_{window}'] = (distances - mean_dist) / std_dist
                pair_df[f'regime_shift_detected_{window}'] = (pair_df[f'regime_shift_zscore_{window}'] > 2).astype(int)
        
        # ===== Wavelet-inspired Multi-timeframe Features =====
        
        # Create features that blend information across timeframes
        timeframes = [(5, 20), (10, 50), (20, 100)]
        
        for short_window, long_window in timeframes:
            # Cross-timeframe z-score comparison
            short_z = f'z_score_{short_window}' if f'z_score_{short_window}' in pair_df.columns else 'z_score'
            long_z = f'z_score_{long_window}' if f'z_score_{long_window}' in pair_df.columns else 'z_score'
            
            pair_df[f'z_timeframe_diff_{short_window}_{long_window}'] = pair_df[short_z] - pair_df[long_z]
            
            # Z-score alignment feature (agreement between timeframes)
            pair_df[f'z_alignment_{short_window}_{long_window}'] = np.sign(pair_df[short_z] * pair_df[long_z]).astype(float)
            
            # Z-score divergence feature (short vs long timeframe)
            pair_df[f'z_divergence_{short_window}_{long_window}'] = np.abs(pair_df[short_z]) - np.abs(pair_df[long_z])
        
        # ===== Trading Signal Combination Features =====
        
        # Create composite trading signals
        windows = [20, 50]
        
        for window in windows:
            # Combine z-score, momentum, and mean reversion signals
            z_col = f'z_score_{window}' if f'z_score_{window}' in pair_df.columns else 'z_score'
            mom_col = f'spread_momentum_{window}' if f'spread_momentum_{window}' in pair_df.columns else f'spread_momentum_5'
            
            # Normalized signals between -1 and 1
            z_norm = np.clip(pair_df[z_col] / 3, -1, 1)  # Normalize z-score
            
            # Normalize momentum
            mom_std = pair_df[mom_col].rolling(window).std().fillna(1)
            mom_norm = np.clip(pair_df[mom_col] / (mom_std * 3), -1, 1)
            
            # Combined signal (equal weighted)
            pair_df[f'combined_signal_{window}'] = -z_norm * 0.7 - mom_norm * 0.3
            
            # Signal strength (absolute value)
            pair_df[f'signal_strength_{window}'] = np.abs(pair_df[f'combined_signal_{window}'])
            
            # Signal consistency (agreement between z-score and momentum)
            pair_df[f'signal_consistency_{window}'] = (np.sign(-z_norm) == np.sign(-mom_norm)).astype(float)
        
        # ===== Feature Normalization and Scaling =====
        
        # Standardize key features relative to their own history
        for feature in ['z_score', 'spread', f'{stock1}_return', f'{stock2}_return']:
            for window in [50, 100]:
                # Compute historical mean and std
                feat_mean = pair_df[feature].rolling(window).mean().fillna(0)
                feat_std = pair_df[feature].rolling(window).std().fillna(1)
                
                # Create standardized feature
                pair_df[f'{feature}_standardized_{window}'] = (pair_df[feature] - feat_mean) / (feat_std + 1e-10)
                
                # Clip extreme values
                pair_df[f'{feature}_standardized_{window}'] = np.clip(
                    pair_df[f'{feature}_standardized_{window}'], -5, 5
                )
        
        # Add short-term volatility features with smaller windows
        short_term_windows = [2, 3, 4]
        for window in short_term_windows:
            # Short-term volatility
            pair_df[f'{stock1}_st_vol_{window}'] = pair_df[stock1].rolling(window).std().fillna(0)
            pair_df[f'{stock2}_st_vol_{window}'] = pair_df[stock2].rolling(window).std().fillna(0)
            pair_df[f'spread_st_vol_{window}'] = pair_df['spread'].rolling(window).std().fillna(0)
            
            # Short-term volatility ratios
            pair_df[f'st_vol_ratio_{window}'] = pair_df[f'{stock1}_st_vol_{window}'] / np.maximum(pair_df[f'{stock2}_st_vol_{window}'], 0.0001)

        # Apply additional non-linear transformations to z-score
        # Asymmetric transformations that emphasize different z-score ranges
        pair_df['z_score_asymm_pos'] = np.where(pair_df['z_score'] > 0, 
                                            pair_df['z_score']**2, 
                                            pair_df['z_score'])
        pair_df['z_score_asymm_neg'] = np.where(pair_df['z_score'] < 0, 
                                            pair_df['z_score']**2, 
                                            pair_df['z_score'])

        # Logarithmic scaling for extreme z-scores (preserves sign)
        pair_df['z_score_log_scaled'] = np.sign(pair_df['z_score']) * np.log1p(np.abs(pair_df['z_score']))

        # Special transformations for mean-reversion signals
        pair_df['z_score_mean_rev'] = -np.sign(pair_df['z_score']) * np.abs(pair_df['z_score'])**0.7

        # Add extremely short-term mean reversion signals
        for window in [2, 3, 4]:
            # Calculate micro z-scores
            pair_df[f'micro_spread_mean_{window}'] = pair_df['spread'].rolling(window).mean()
            pair_df[f'micro_spread_std_{window}'] = pair_df['spread'].rolling(window).std()
            pair_df[f'micro_z_score_{window}'] = (pair_df['spread'] - pair_df[f'micro_spread_mean_{window}']) / np.maximum(pair_df[f'micro_spread_std_{window}'], 0.0001)
            
            # Short-term mean reversion velocity
            pair_df[f'micro_z_velocity_{window}'] = pair_df[f'micro_z_score_{window}'].diff().fillna(0)
            
            # Mean reversion acceleration
            pair_df[f'micro_z_accel_{window}'] = pair_df[f'micro_z_velocity_{window}'].diff().fillna(0)


        # Z-score crossing indicators (short-term signal triggers)
        for threshold in [0, 0.5, 1.0, 1.5, 2.0]:
            # Positive crossings
            pair_df[f'z_cross_pos_{threshold}'] = ((pair_df['z_score'].shift(1) < threshold) & 
                                                (pair_df['z_score'] >= threshold)).astype(int)
            # Negative crossings
            pair_df[f'z_cross_neg_{threshold}'] = ((pair_df['z_score'].shift(1) > -threshold) & 
                                                (pair_df['z_score'] <= -threshold)).astype(int)

        # Divergence indicators for stocks (when they move in opposite directions)
        pair_df['stocks_diverge'] = ((pair_df[f'{stock1}_return'] * pair_df[f'{stock2}_return']) < 0).astype(int)
        
        # Create adaptive lookback windows based on recent volatility
        for base_window in [5, 10, 20]:
            # Dynamic window size based on volatility
            vol_ratio = pair_df[f'spread_vol_{base_window}'] / pair_df[f'spread_vol_{base_window}'].rolling(100).mean().fillna(1)
            # Higher volatility = shorter window, lower volatility = longer window
            dynamic_window = np.clip(base_window / np.clip(vol_ratio, 0.5, 2), base_window/2, base_window*2).astype(int)
            
            # Apply exponential weights for more responsive mean
            weights = np.exp(np.linspace(-1, 0, base_window))
            weights = weights / weights.sum()
            
            # Calculate weighted stats for spread
            pair_df[f'spread_weighted_mean_{base_window}'] = pair_df['spread'].rolling(base_window).apply(
                lambda x: np.sum(x * weights[-len(x):] / np.sum(weights[-len(x):])), raw=True
            ).fillna(pair_df['spread'])

        # Create interaction features between z-score and other relevant indicators
        for indicator in ['spread_vol_5', f'{stock1}_rsi', f'{stock2}_rsi']:
            if indicator in pair_df.columns:
                # Z-score interaction features
                pair_df[f'z_score_{indicator}_interact'] = pair_df['z_score'] * pair_df[indicator]
                
                # Non-linear interactions
                pair_df[f'z_score_{indicator}_interact_nonlin'] = np.sign(pair_df['z_score']) * pair_df[indicator]

        # Create features based on combinations of z-scores at different timeframes
        for window1 in [3, 5, 10]:
            for window2 in [20, 30, 50]:
                if window1 >= window2:
                    continue
                    
                z1_col = f'z_score_{window1}' if f'z_score_{window1}' in pair_df.columns else 'z_score'
                z2_col = f'z_score_{window2}' if f'z_score_{window2}' in pair_df.columns else 'z_score'
                
                # Z-score agreement signals
                pair_df[f'z_agreement_{window1}_{window2}'] = (np.sign(pair_df[z1_col]) == np.sign(pair_df[z2_col])).astype(int)
                
                # Z-score relative strength
                pair_df[f'z_rel_strength_{window1}_{window2}'] = np.abs(pair_df[z1_col]) - np.abs(pair_df[z2_col])

        # Z-score oscillator features
        for window in [3, 5, 7]:
            # Calculate z-score momentum
            z_mom = pair_df['z_score'].diff(window).fillna(0)
            
            # Z-score RSI-like indicator
            up_z = np.maximum(z_mom, 0)
            down_z = np.maximum(-z_mom, 0)
            
            avg_up = up_z.rolling(window=window*2).mean().fillna(0)
            avg_down = down_z.rolling(window=window*2).mean().fillna(0)
            
            rs = avg_up / np.maximum(avg_down, 0.0001)
            pair_df[f'z_rsi_{window}'] = 100 - (100 / (1 + rs))
            
        # Spread rate of change features
        for window in [1, 2, 3, 5]:
            # Calculate various rates of change
            pair_df[f'spread_roc_{window}'] = (pair_df['spread'] / pair_df['spread'].shift(window) - 1).fillna(0)
            pair_df[f'z_score_roc_{window}'] = (pair_df['z_score'] / pair_df['z_score'].shift(window) - 1).fillna(0)
            
            # Replace infinities with 0
            pair_df[f'spread_roc_{window}'] = pair_df[f'spread_roc_{window}'].replace([np.inf, -np.inf], 0)
            pair_df[f'z_score_roc_{window}'] = pair_df[f'z_score_roc_{window}'].replace([np.inf, -np.inf], 0)

        # Apply rank transformations to make features more robust for ML
        for col in ['z_score', 'spread', f'{stock1}_return', f'{stock2}_return']:
            # Rolling rank transformation (percentile)
            for window in [50, 100]:
                pair_df[f'{col}_rank_{window}'] = pair_df[col].rolling(window).rank(pct=True).fillna(0.5)
        
        # Fill any remaining NaN values
        pair_df = pair_df.fillna(0)
        
        return pair_df

    def EDA(self):
        """
        Perform Exploratory Data Analysis (EDA) on the pairs data and save the plots.
        From the plots we notice that the log returns are almost gaussian normally distributed,
        and that the returns are correlated with the volatility and with the returns of the other stock.
        (Additional validation for the pairs)
        """
        # Ensure the plots directory exists
        plots_dir = 'plots/EDA'
        os.makedirs(plots_dir, exist_ok=True)

        for pair in self.pairs:
            stock1 = pair[0].replace(' ', '_')
            stock2 = pair[1].replace(' ', '_')
            pair_name = f'{stock1}_{stock2}_total_df'
            pair_df = globals()[pair_name]

            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(14, 20))

            # KDE Plot of Log Returns
            ax[0, 0].hist(pair_df[f'{stock1}_log_returns'], bins=50, edgecolor='black', alpha=0.5, label=stock1, color='royalblue')
            ax[0, 0].hist(pair_df[f'{stock2}_log_returns'], bins=50, edgecolor='black', alpha=0.5, label=stock2, color='orange')
            ax[0, 0].set_title(f'KDE Plot of Log Returns for {stock1} and {stock2}')
            ax[0, 0].set_xlabel('Log Returns')
            ax[0, 0].set_ylabel('Frequency')
            ax[0, 0].legend()

            # Scatter Plot of Returns
            ax[0, 1].scatter(pair_df[f'{stock1}_returns'], pair_df[f'{stock2}_returns'], label=f'{stock1} vs {stock2}', color='royalblue')
            ax[0, 1].set_title(f'Scatter Plot of Returns for {stock1} and {stock2}')
            ax[0, 1].set_xlabel(f'{stock1} Returns')
            ax[0, 1].set_ylabel(f'{stock2} Returns')
            ax[0, 1].legend()

            # Box Plot of Log Returns
            ax[1, 0].boxplot([pair_df[f'{stock1}_log_returns'], pair_df[f'{stock2}_log_returns']], vert=False, labels=[stock1, stock2])
            ax[1, 0].set_title(f'Box Plot of Log Returns for {stock1} and {stock2}')

            # Scatter Plot of Returns vs Volatility
            ax[1, 1].scatter(pair_df[f'{stock1}_20std'], pair_df[f'{stock1}_returns'], label=f'{stock1} Returns vs Volatility', color='royalblue', edgecolors='black')
            ax[1, 1].scatter(pair_df[f'{stock2}_20std'], pair_df[f'{stock2}_returns'], label=f'{stock2} Returns vs Volatility', color='lightblue', edgecolors='black')
            ax[1, 1].set_title(f'Scatter Plot of Returns vs Volatility for {stock1} and {stock2}')
            ax[1, 1].set_xlabel('20-Period Standard Deviation')
            ax[1, 1].set_ylabel('Returns')
            ax[1, 1].legend()

            # Plot of Prices Against Time
            ax[2, 0].plot(pair_df.index, pair_df[stock1], label=stock1, color='royalblue')
            ax[2, 0].plot(pair_df.index, pair_df[stock2], label=stock2, color='lightblue')
            ax[2, 0].set_title(f'Price of {stock1} and {stock2} Over Time')
            ax[2, 0].set_xlabel('Date')
            ax[2, 0].set_ylabel('Price')
            ax[2, 0].legend()

            # Plot of Spread
            ax[2, 1].plot(pair_df.index, pair_df['spread'], label='Spread', color='royalblue')
            ax[2, 1].hlines(pair_df['spread_mean'].mean(), pair_df.index[0], pair_df.index[-1], colors='red', linestyles='--', label='Mean')
            ax[2, 1].hlines(pair_df['spread_mean'].mean() + pair_df['spread_std'].mean(), pair_df.index[0], pair_df.index[-1], colors='green', linestyles='--', label='+1 Std')
            ax[2, 1].hlines(pair_df['spread_mean'].mean() - pair_df['spread_std'].mean(), pair_df.index[0], pair_df.index[-1], colors='green', linestyles='--', label='-1 Std')
            ax[2, 1].set_title('Spread Over Time')
            ax[2, 1].set_xlabel('Date')
            ax[2, 1].set_ylabel('Spread')
            ax[2, 1].legend()

            # Plot of Z-Score
            ax[3, 0].plot(pair_df.index, pair_df['z_score'], label='Z-Score', color='royalblue')
            ax[3, 0].hlines(0, pair_df.index[0], pair_df.index[-1], colors='red', linestyles='--', label='Mean')
            ax[3, 0].hlines(1, pair_df.index[0], pair_df.index[-1], colors='green', linestyles='--', label='+1 Std')
            ax[3, 0].hlines(-1, pair_df.index[0], pair_df.index[-1], colors='green', linestyles='--', label='-1 Std')
            ax[3, 0].set_title('Z-Score Over Time')
            ax[3, 0].set_xlabel('Date')
            ax[3, 0].set_ylabel('Z-Score')
            ax[3, 0].legend()

            # Check if Z-Score is stationary
            adf_result = adfuller(pair_df['z_score'].dropna())
            ax[3, 1].text(0.1, 0.5, f'ADF Statistic: {adf_result[0]:.2f}\nP-Value: {adf_result[1]:.2f}', fontsize=12)
            ax[3, 1].set_title('Z-Score Stationarity Test')
            ax[3, 1].axis('off')

            plt.tight_layout()
            plt.savefig(f'plots\\EDA\\{stock1}_{stock2}_EDA.png', bbox_inches='tight')
            plt.close()

    def perform_fft(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method performs Fast Fourier Transform (FFT) on the z-score returns and returns as features.
        """

        z_score_returns = pair_df['z_score_returns']
        
        # Make sure we have enough data for FFT
        if len(z_score_returns) < 2:
            print("Warning: Not enough data points for FFT")
            return pd.DataFrame(index=pair_df.index)
        
        # Perform FFT on the z-score returns
        close_fft = np.fft.fft(np.asarray(z_score_returns.tolist()))
        
        # Create FFT DataFrame with the correct index
        fft_df = pd.DataFrame({'fft': close_fft}, index=z_score_returns.index)
        
        # Compute additional features
        fft_df['absolute'] = np.abs(fft_df['fft'])  # FFT absolute value
        fft_df['angle'] = np.angle(fft_df['fft'])  # FFT angle
        
        # Select the relevant features
        fft_features = pd.DataFrame(index=z_score_returns.index)
        fft_features['z_score_fft_absolute'] = fft_df['absolute']
        fft_features['z_score_fft_angle'] = fft_df['angle']
        
        # Apply Inverse FFT with different frequency cutoffs
        fft_list = np.asarray(fft_df['fft'].tolist())  # Convert FFT results to numpy array
        
        # Limit the range based on the length of your data
        max_cutoff = min(104, len(fft_list) // 2)
        
        for num_ in np.arange(0, max_cutoff, 4):  # Range from 0 to max_cutoff with step size 4
            if 2 * num_ >= len(fft_list):
                # Skip cases where we'd filter out everything
                continue
                
            fft_filtered = np.copy(fft_list)
            if num_ > 0:  # Only filter if num_ > 0
                fft_filtered[num_:-num_] = 0  # Zero out frequencies except the first and last 'num_'
            
            # Compute inverse FFT
            reconstructed_series = np                                                                                                                                                                                                                                                                                                                                                                                       .fft.ifft(fft_filtered).real

            # Add to DataFrame
            fft_features[f'z_score_fft_recon_{num_}'] = reconstructed_series
        
        # Fill any remaining NaNs with 0
        fft_features.fillna(0, inplace=True)
        
        return fft_features

    def arima_model(self, pair_df: pd.DataFrame, order=(1, 0, 0)) -> pd.DataFrame: # 1,0,0
        """
        Fit an ARIMA model to the z-score series and return a DataFrame with combined fitted and predicted values for machine learning
        model training. The 1,0,0 model was tested and found to be the best in general, based on summary statistics and statistical plots.
        """

        z_score_returns = pair_df['z_score_returns']

        adf_result = adfuller(z_score_returns) # Checking if the z_score returns is stationary

        if adf_result[0] < 0.05:
            # Fitting the ARIMA model
            model = ARIMA(z_score_returns, order=order)
            model_fit = model.fit(method='innovations_mle')

        else: 
            print('📌 z_score returns not stationary')
            # Fit the differenced ARIMA model (order=(1,1,0) for differenced series)
            model = ARIMA(z_score_returns, order=(1,1,0))
            model_fit = model.fit(method='innovations_mle')

        print(model_fit.summary())

        # Get in-sample (training) fitted values
        fitted_values = model_fit.fittedvalues  # These correspond to the training set
        
        # Train-test split
        X = z_score_returns.values
        size = int(len(X) * 0.95)  # Define train-test split
        train, test = X[:size], X[size:]

        history = list(train)
        predictions = []

        # Forecast values for the test set
        for t in tqdm(range(len(test)), desc='Processing ARIMA for z_score'):
            model = ARIMA(history, order=order)
            model_fit = model.fit(method='innovations_mle')
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])

        # Convert predictions into a Series with the correct index
        predictions_series = pd.Series(predictions, index=z_score_returns.index[size:])
        
        # Merge fitted and predicted values into one column
        arima_predicted = fitted_values.combine_first(predictions_series)

        # Return as a DataFrame
        return pd.DataFrame({'arima_predicted': arima_predicted})

    def process_pair(self, pair_df):
        print(f"Processing DataFrame with shape: {pair_df.shape}")

        stock1 = pair_df.columns[0]  # First stock ticker
        stock2 = pair_df.columns[1]  # Second stock ticker
        pair_name = f'{stock1}_{stock2}_total_df'

        pair_df = globals().get(pair_name, pair_df)
        print(f"✅ Found {pair_name} in globals with shape {pair_df.shape}")
        print(f"📊 Columns in {pair_name}: {list(pair_df.columns)}")
        # Filter columns with NaNs
        nan_columns = pair_df.columns[pair_df.isna().any()]

        print("Columns with NaNs:", list(nan_columns))

        def process_stock():
            print(f"📌 Performing FFT & ARIMA for z_score...")
            fft_features = self.perform_fft(pair_df)
            arima_features = self.arima_model(pair_df)
            return fft_features, arima_features

        # Process stocks 
        fft_features, arima_features = process_stock()

        # Concatenating FFT and ARIMA features
        print(f"🔗 Concatenating features for {pair_name}...")
        pair_df = pd.concat([pair_df, fft_features, arima_features], axis=1)
        print(f"✅ Concatenation successful. New shape: {pair_df.shape}")

        # Update the global variable with the new DataFrame
        globals()[pair_name] = pair_df
        print(f"✅ {pair_name} updated in globals.")
        print(pair_df.head())

    def normalize_features(self, pairs):
        '''
        This method calls a method from the FeatureLabelling class to normalize the features, using a rolling standard scaler. 
        '''
        normalize = FeatureLabelling()
        self.normalized_df = normalize.normalize_features(pairs, globals())
        df = self.normalized_df

        # Replacing inf value issues (to prevent XGBoost errors that were occuring)
        df['arima_predicted'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['arima_predicted'].fillna(df.mean(), inplace=True)

        return df

    def feature_importance_analysis(self, stock1, stock2, normalized_data):
        """
        This method calls the FeatureImportance class to find the top features that have the most
        predictive capabilities for the given pair using the normalized data made.
        """
        feature_importance_model = FeatureImportance(stock1, stock2, normalized_data=normalized_data)
        feature_importance_model.get_feature_importance_data()
        feature_importance_model.train_model()
        top_features = feature_importance_model.plot_simplified_analysis()
        return top_features

    def run_base_model(self, batch_size=32, epochs=10):
        """
        This method calls to run the base model (LSTM) for training and evaluation using the Purged K-CV approach.
        """
        logistic_model = LogisticRegressionClassifier(normalized_df=self.normalized_df)
        base_model_result = logistic_model.run()
        print(base_model_result)
        accuracy = logistic_model.evaluate(base_model_result)
        return base_model_result
        
    def Run_NN(self, stock1, stock2, returns, normalized_df):
        '''
        This method is called to run the Neural Network model, which is the final model for our strategy.
        '''
        NN = Neural_Network(stock1, stock2, returns, normalized_df)
        Final_df = NN.run()
        accuracy = NN.evaluate(Final_df, normalized_df)
        NN.plot_calibration_curve(Final_df)
        return Final_df, accuracy
    
    # Add method to evaluate combined performance of the models
    def Combining_Models(self, base_model_result, final_df, normalized_df):
        Combine = Combiner()
        combined_df = Combine.Combine_models(base_model_result, final_df)
        combined_eval = Combine.Evaluate(combined_df)
        combined_result = Combine.Label(combined_df, normalized_df)
        return combined_result
    
    def Run_Backtest(self, stock1, stock2, data, final_df, normalized_df, combined_final, returns):
        '''
        This method is called to run the backtest for the final model, using the Backtest class.
        '''
        backtester = Backtest(stock1, stock2, data, final_df, normalized_df, combined_final, returns)
        backtester.run_backtest()

        return backtester


def run_multiple_pairs(stat_arb, pairs_to_process=None, max_pairs=None):
    """
    Run backtest on multiple pairs and combine results
    
    Args:
        stat_arb: StatArb instance
        pairs_to_process: List of pair indices to process (if None, process all)
        max_pairs: Maximum number of pairs to process
    
    Returns:
        portfolio: PortfolioBacktest instance with combined results
    """
    # Initialize portfolio
    portfolio = PortfolioBacktest()
    
    # Determine which pairs to process
    if pairs_to_process is None:
        # Process all pairs up to max_pairs
        if max_pairs is None:
            max_pairs = len(stat_arb.pairs)
        pairs_to_process = list(range(min(max_pairs, len(stat_arb.pairs))))
    
    # Process each pair
    for i in pairs_to_process:
        if i >= len(stat_arb.pairs):
            print(f"Warning: Pair index {i} out of range. Skipping.")
            continue
            
        pair = stat_arb.pairs[i]
        stock1 = pair[0].replace(' ', '_')
        stock2 = pair[1].replace(' ', '_')
        pair_name = f'{stock1}_{stock2}'
        
        print(f"Processing {pair_name}... ({i+1}/{len(pairs_to_process)})")
        
        # Get pair dataframe
        pair_df_name = f'{stock1}_{stock2}_total_df'
        pair_df = globals()[pair_df_name]
        
        # Process the pair
        stat_arb.process_pair(pair_df)
        
        # Normalize features
        normalized_data = stat_arb.normalize_features([pair])
        clipped_data = normalized_data.clip(-3, 3)  # Ensures all values are between -3 and 3 for stability
        
        # Feature importance analysis
        top_features = stat_arb.feature_importance_analysis(stock1, stock2, clipped_data)
        top_features.append('z_score')
        stat_arb.normalized_df = clipped_data[top_features]
        
        # Run base model (LSTM) with Purged-K CV
        print('📌 Running Base Model')
        base_model_result = stat_arb.run_base_model()
        
        # Align indices and fill missing values
        normalized_df = pd.concat([stat_arb.normalized_df, base_model_result['prediction']], 
                                 axis=1, join='outer').fillna(0)
        
        # Run NN Model
        print('📌 Running NN Model')
        stock1_orig = pair[0]  # Original names without space replacement
        stock2_orig = pair[1]
        returns = stat_arb.returns
        final_df, accuracy = stat_arb.Run_NN(stock1_orig, stock2_orig, returns, normalized_df)
        
        # Filter to relevant period
        # final_df = final_df.loc[final_df.index >= '2021-03-01'] # This is due to no test period on the first training data set
        
        # Combine model results
        combined_final = stat_arb.Combining_Models(base_model_result, final_df, normalized_df)
        
        # Run backtest
        data = stat_arb.pricing
        backtest = stat_arb.Run_Backtest(stock1_orig, stock2_orig, data, final_df, 
                                        normalized_df, combined_final, returns)
        
        # Add this pair's results to the portfolio
        portfolio.add_pair_result(pair_name, backtest)
        
        print(f"Completed processing pair {pair_name}")
        
    # Calculate portfolio performance
    portfolio_df = portfolio.calculate_portfolio_performance()
    
    # Visualize portfolio results
    portfolio.visualize_portfolio(portfolio_df)
    
    # Export results
    portfolio.export_results(f"portfolio_results_{len(pairs_to_process)}_pairs.csv")
    
    return portfolio

# Running the Method
if __name__ == "__main__":
    '''
    First we load the data, then we create dataframes for each pair, then we perform EDA on the pairs data.
    '''
    stat_arb = StatArb('SX7P_cleaned.csv')
    stat_arb.load_data()
    stat_arb.pairs_dataframes()
    stat_arb.EDA()

    run_multiple_pairs(stat_arb)
    
    
