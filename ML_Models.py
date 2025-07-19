import math
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import keras
import plotly.express as px
import plotly.graph_objects as go
from keras import layers, models
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras_tuner as kt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, classification_report
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import seaborn as sns
from ML_Model_Trainer import Purged_K_CV
import scipy.stats as stats

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set random seed for reproducibility
np.random.seed(42)
random.seed()

class LogisticRegressionClassifier:
    """
    A logistic regression-based trading signal classifier for pairs trading strategies.
    
    This class implements a machine learning pipeline for generating trading signals based on 
    z-score normalization and Sharpe ratio-based labeling. It uses Purged K-fold Cross-Validation 
    to prevent data leakage and optimize for recall performance with a custom threshold.
    
    Attributes:
        df (pd.DataFrame): Normalized dataframe containing trading features and z-scores
        sequence_length (int): Length of sequences for time series analysis (default: 50)
        lookback (int): Number of periods to look back for predictions (default: 1)
        threshold (float): Classification threshold for binary predictions (default: 0.3)
        model (LogisticRegression): Scikit-learn logistic regression model
        scaler (MinMaxScaler): Feature scaling transformer
        purged_k_cv (Purged_K_CV): Custom cross-validation handler for time series
        dates (array): Preserved date indices for temporal analysis
        period (int): Prediction horizon in days (default: 3)
        SR (float): Target Sharpe ratio threshold for labeling (default: 0.5)
        RF (float): Risk-free rate for Sharpe calculation (default: 0)
    
    Methods:
        prepare_data(): Prepares features and labels using Sharpe ratio-based strategy
        build_model(): Constructs a balanced logistic regression model
        calculate_linear_weights(num_samples): Generates time-weighted sample weights
        evaluate(result): Evaluates model performance with focus on recall metrics
        run(): Executes the complete training and evaluation pipeline
    """

    def __init__(self, normalized_df, sequence_length=50, lookback=1, threshold=0.3):
        self.df = normalized_df
        self.sequence_length = sequence_length
        self.lookback = lookback
        self.threshold = threshold  # Setting threshold to 0.3 as requested
        self.model = self.build_model()  # Build model
        self.scaler = MinMaxScaler()
        self.purged_k_cv = Purged_K_CV()  
        self.dates = None
        self.period = 3 # N - Prediction Horizon
        self.SR = 0.5 # SF - Sharpe Ratio
        self.RF = 0 # 0.02 / 252 # RF - Risk Free Rate (daily, assumed 2% annual)

    def prepare_data(self):
        """
        Prepare the data for training with Purged K-fold CV, preserving original date indices.
        """
    
        # Assigning variables for Prediction Horizon, Sharpe Ratio and Risk Free Rate
        N = self.period
        SR_target = self.SR
        Rf = self.RF

        # Calculate Log returns
        log_returns = np.log(self.df['z_score'] / self.df['z_score'].shift(1)).fillna(0)

        # Calculate future return: sum of next N returns
        future_return = log_returns.shift(-N).rolling(window=N).sum()

        # Calculate future volatility: standard deviation over the next N periods
        future_vol = log_returns.shift(-N).rolling(window=N).std()

        # Avoid division by zero
        safe_future_vol = future_vol.replace(0, np.nan)

        # Calculate expected Sharpe ratio for long and short trades
        expected_sharpe_long = future_return / safe_future_vol
        expected_sharpe_short = -future_return / safe_future_vol  # Short trades reverse the sign

        # Defining labeling strategy similar to original but using Sharpe comparison
        self.df['label'] = np.where(expected_sharpe_long > SR_target, 1,  # Trade: long when expected Sharpe > target
                        np.where(expected_sharpe_short > SR_target, 1,  # Trade: short when expected Sharpe > target
                                    0))  # No trade

        print('Printing the number of trades:')
        print(self.df['label'].value_counts())

        features = self.df.drop(columns=['label', 'z_score']).shift(1).iloc[1:].copy()

        # Set the target as the label column
        target = self.df['label'][1:]

        # Get dates from the original dataframe (assuming it has a datetime index)
        original_dates = self.df.index[1:]  # Skip the first row as we shifted features
      
        # Normalize the features
        #features_scaled = pd.DataFrame(self.scaler.fit_transform(features), columns=features.columns, index=features.index)
        
        # For logistic regression, we don't need sequences, just the latest data point
        X = features
        y = target
        preserved_dates = original_dates
        
        # Calculate time-based sample weights (linear)
        weights = self.calculate_linear_weights(len(y))

        # Adding back the z_score returns for fold fft calculations
        z_score = self.df['z_score']
        self.df['z_score_returns'] = np.log(z_score) - np.log(z_score.shift(1))
        self.df['z_score_returns'] = self.df['z_score_returns'].replace([np.inf, -np.inf], np.nan).fillna(0)

        df = self.df

        return X, y, weights, preserved_dates, df

    def build_model(self):
        """
        Build a Logistic Regression model optimized for recall
        """
        # Use class_weight='balanced' to handle class imbalance
        # and use 'liblinear' solver which works well with small datasets
        model = LogisticRegression(
            class_weight='balanced',  # Help with class imbalance
            solver='lbfgs',       # Better for smaller datasets
            penalty='l2',             # Ridge regularization
            C=0.1,                    # Regularization strength (smaller values = stronger regularization)
            random_state=42,
            max_iter=1000,            # Increase max iterations for convergence
        )
        
        return model
        
    def calculate_linear_weights(self, num_samples):
        """
        Calculate linearly increasing weights for time-based weighting.
        The most recent data points receive higher weights.
        """
        weights = np.linspace(1, num_samples, num_samples)
        weights = weights / weights.sum()  # Normalize the weights so that they sum to 1
        return weights

    def evaluate(self, result):
        """
        Evaluate the model performance with focus on recall at threshold 0.3.
        """
        y_true = result['true_value'][20:].copy()
        y_pred_proba = result['prediction'][20:].copy()

        # Apply 0.3 threshold for predictions, converting probs to binary
        y_pred = np.where(y_pred_proba >= self.threshold, 1, 0)

        print('Printing y_true from Evaluation')
        print(y_true)
        print('Printing y_pred from Evaluation')
        print(y_pred)
        print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

        # Calculate and print recall
        recall_before = recall_score(y_true, y_pred, zero_division=0)
        accuracy_before = accuracy_score(y_true, y_pred)
        print(f"Recall (threshold={self.threshold}): {recall_before:.4f}")
        print(f"Accuracy : {accuracy_before:.4f}")

        # Print classification report for initial predictions
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=[0, 1]))

        # Compute AUC-ROC and Average Precision Score
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        print("\n--- Advanced Model Performance ---")
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        print(f"Average Precision Score: {avg_precision:.4f}")
        print(f"Prediction Threshold: {self.threshold}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        return {
            'recall_before': recall_before,
            'accuracy_before': accuracy_before,
            'auc_roc': auc_roc,
            'avg_precision': avg_precision
        }

    def run(self):
        """
        Run the entire model workflow using Purged K-fold CV with preserved date indices.
        """
        # Prepare the data
        X, y, weights, preserved_dates, df = self.prepare_data()
        
        # Modify the ML_Model_Trainer.Purged_K_CV class to handle logistic regression
        # we'll rely on the existing implementation but adapt it for our simpler model
        
        # Train the model using purged k-fold CV
        results = self.purged_k_cv.train_walk_forward_Logistic(
            self.model, X, y, df, 
            original_indices=preserved_dates,
            sample_weights=weights
        )

        return results


class Neural_Network:

    """
    An LSTM-based neural network classifier for advanced trading signal prediction.
    
    This class implements a deep learning pipeline combining LSTM architecture
    with comprehensive feature engineering, regime detection, and probabilistic calibration
    for pairs trading strategies. It uses Purged K-fold Cross-Validation and advanced
    regularization techniques to prevent overfitting.
    
    Attributes:
        stock1, stock2 (str): Trading pair identifiers
        returns (array): Return series for the trading pair
        df (pd.DataFrame): Normalized dataframe with trading features
        sequence_length (int): LSTM sequence length (default: 50)
        lookback (int): Prediction lookback period (default: 1)
        batch_size (int): Training batch size (default: 32)
        epochs (int): Maximum training epochs (default: 100)
        learning_rate (float): Model learning rate (default: 0.001)
        dropout_rate (float): Dropout regularization rate (default: 0.2)
        scaler (RobustScaler): Feature scaling transformer
        purged_k_cv (Purged_K_CV): Time series cross-validation handler
        dates (array): Preserved temporal indices
        threshold (float): Binary classification threshold (default: 0.5)
        period (int): Prediction horizon in days (default: 3)
        SR (float): Target Sharpe ratio for labeling (default: 0.5)
        RF (float): Risk-free rate (default: 0)
        platt_scaler (LogisticRegression): Probability calibration model
        model (keras.Model): Compiled LSTM neural network
    
    Methods:
        prepare_data(): Comprehensive data preparation with advanced feature engineering
        create_sequences(features, target, dates): Creates LSTM-compatible sequences
        Additional_Features(z_score): Generates 100+ technical and statistical features
        normalize_features(features_df): Applies rolling window normalization
        build_model(): Constructs regularized LSTM with temperature scaling
        calculate_exponential_weights(num_samples): Generates exponential time weights
        evaluate(data, normalized_df): Comprehensive model evaluation with AUC metrics
        plot_calibration_curve(final_df, n_bins, save_path): Plots reliability curves
        run(): Executes complete training pipeline with advanced callbacks
    
    Feature Engineering:
        - Multi-window statistical measures (mean, std, skew, kurtosis)
        - Velocity and acceleration indicators
        - Momentum and trend calculations
        - Distance from moving averages (normalized)
        - CUSUM regime change detection
        - Extreme value indicators
        - Exponential moving averages
        - Rolling normalization with 252-day windows
    
    Model Architecture:
        - Gaussian noise input layer for regularization
        - Dual LSTM layers (128, 64 units) with tanh activation
        - Batch normalization and dropout (0.4) between layers
        - Dense layers with LeakyReLU activation
        - Temperature scaling for aggressive probability predictions
        - Binary cross-entropy loss with class balancing
    
    Advanced Features:
        - Purged K-fold CV prevents temporal data leakage
        - Exponential sample weighting emphasizes recent observations
        - Temperature scaling pushes predictions toward extremes (0 or 1)
        - Early stopping and learning rate reduction callbacks
        - Comprehensive evaluation metrics (AUC-ROC, Average Precision)
        - Calibration curve plotting for probability assessment

    """
     
    def __init__(self, stock1, stock2, returns, normalized_df, sequence_length=50, lookback=1,  
                 batch_size=32, epochs=100, learning_rate=0.001, dropout_rate=0.2):
        self.stock1 = stock1
        self.stock2 = stock2
        self.returns = returns
        self.df = normalized_df
        self.sequence_length = sequence_length
        self.lookback = lookback
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.scaler = RobustScaler()
        self.purged_k_cv = Purged_K_CV()  
        self.dates = None
        self.threshold = 0.5  # For precision classification scoring of second model
        self.period = 3 # N - Prediction Horizon
        self.SR = 0.5 # SF - Sharpe Ratio
        self.RF = 0 # 0.02 / 252 # RF - Risk Free Rate (daily, assumed 2% annual)
        self.platt_scaler = LogisticRegression()  # Logistic Regression for Platt Scaling
        self.model = self.build_model()  # Build model

    def prepare_data(self):
        """
        Prepare the data for training with Purged K-fold CV, preserving original date indices.
        """
        
        # Assigning variables for Prediction Horizon, Sharpe Ratio and Risk Free Rate
        N = self.period
        SR_target = self.SR
        Rf = self.RF

        # Calculate Log returns
        log_returns = np.log(self.df['z_score'] / self.df['z_score'].shift(1)).fillna(0)

        # Calculate future return: sum of next N returns
        future_return = log_returns.shift(-N).rolling(window=N).sum()

        # Calculate future volatility: standard deviation over the next N periods
        future_vol = log_returns.shift(-N).rolling(window=N).std()

        # Avoid division by zero
        safe_future_vol = future_vol.replace(0, np.nan)

        # Calculate expected Sharpe ratio for long and short trades
        expected_sharpe_long = future_return / safe_future_vol
        expected_sharpe_short = -future_return / safe_future_vol  # Short trades reverse the sign

        # Defining labeling strategy similar to original but using Sharpe comparison
        self.df['label'] = np.where(expected_sharpe_long > SR_target, 1,  # Trade: long when expected Sharpe > target
                        np.where(expected_sharpe_short > SR_target, 1,  # Trade: short when expected Sharpe > target
                                    0))  # No trade
        
        # Extract z_score as a Series
        z_score = self.df['z_score']

        # Making sure to add the z_score returns to the dataframe
        z_score = self.df['z_score']
        self.df['z_score_returns'] = np.log(z_score) - np.log(z_score.shift(1))
        self.df['z_score_returns'] = self.df['z_score_returns'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Create additional features
        additional_features = self.Additional_Features(z_score)
        print("Shape of additional features before normalization:", additional_features.shape)
        
        # Normalize the features
        additional_features_scaled = self.normalize_features(additional_features)
        print("Shape of additional features after normalization:", additional_features_scaled.shape)
        
        # Append new features to the dataframe
        print("Shape before concat:", self.df.shape)
        self.df = pd.concat([self.df, additional_features_scaled], axis=1)
        print("Shape after concat:", self.df.shape)

        # Extract features and target
        features = self.df.drop(columns=['label', 'z_score', 'z_score_returns']).shift(1).iloc[1:].copy()  # Shift and drop first row
        # Set the target as the label column
        target = self.df['label'].values[1:]
        
        # Get dates from the original dataframe (assuming it has a datetime index)
        original_dates = self.df.index[1:]  # Skip the first row as we shifted features
        
        # Normalize the features using the scaler
        # features_scaled = pd.DataFrame(self.scaler.fit_transform(features), columns=features.columns, index=features.index)

        # Checking if we have fft columns present
        fft_cols = [col for col in features.columns if col.startswith('z_score_fft_recon_')]
        fft_num = []
        # Getting the component number of the fft columns
        for fft_col in fft_cols:
            component_num = int(fft_col.split('_')[-1])
            fft_num.append(component_num) # Appending the integar value

        # Now dropping the fft columns
        features = features.drop(columns=[col for col in features.columns if col.startswith('z_score_fft_recon_')])
        
        # Checking if we have arima model present
        arima_model = False

        if 'arima_predicted' in features.columns:
            features = features.drop(columns='arima_predicted', axis=1)
            arima_model = True
        
        # Create sequences and preserve the exact date mappings
        X, y, preserved_dates = self.create_sequences(features, target, original_dates)
        print("Shape of X (LSTM input) after sequencing:", X.shape)
        
        # Calculate time-based sample weights (Exponential this time)
        weights = self.calculate_exponential_weights(len(y))
        
        self.dates = preserved_dates

        df = self.df

        return X, y, weights, preserved_dates, df, fft_num, arima_model
    
    def create_sequences(self, features, target, dates):
        """
        Create sequences from the time series data for LSTM input while preserving original dates.
        """
        X, y = [], []
        sequence_dates = []

        # Ensure dates are in an appropriate format
        dates = np.array(dates)  # Convert DatetimeIndex to NumPy array

        for i in range(len(features) - self.sequence_length - self.lookback + 1):
            target_idx = i + self.sequence_length + self.lookback - 1

            X.append(features[i:i+self.sequence_length])
            y.append(target[target_idx])

            # Preserve the corresponding date
            sequence_dates.append(dates[target_idx])

        X_array = np.array(X)
        y_array = np.array(y)
        sequence_dates_array = np.array(sequence_dates, dtype='datetime64')

        # Target Distribution Check
        unique, counts = np.unique(y_array, return_counts=True)
        print("\nTarget Distribution:")
        for val, count in zip(unique, counts):
            print(f"Class {val}: {count} samples ({count/len(y_array)*100:.2f}%)")
        
        # Feature Characteristics
        X_features = X_array.reshape(-1, X_array.shape[-1])

        return X_array, y_array, sequence_dates_array

    def Additional_Features(self, z_score):
        """
        Creating comprehensive and scalable additional features for the dataset,
        to help with advanced statistics and regime detection for final model.
        
        Dynamically generates features across multiple window sizes.
        """
        # Define window sizes to use
        window_sizes = [3, 5, 7, 10, 20, 30, 50, 100, 252]
        
        # Create a DataFrame to store all features
        features_df = pd.DataFrame(index=z_score.index)
        features_df['z_score'] = z_score
        
        # Comprehensive feature generation function
        def generate_features_for_windows(series, windows):
            """
            Generate a comprehensive set of features for multiple window sizes
            """
            features = {}
            
            for window in windows:
                # 1. Rolling Statistical Measures
                features[f'z_score_mean_{window}d'] = series.rolling(window=window).mean()
                features[f'z_score_std_{window}d'] = series.rolling(window=window).std()
                features[f'z_score_skewness_{window}d'] = series.rolling(window=window).skew()
                features[f'z_score_kurtosis_{window}d'] = series.rolling(window=window).kurt()
                
                # 2. Velocity and Acceleration
                rolling_mean = series.rolling(window=window).mean()
                features[f'z_score_velocity_{window}d'] = rolling_mean.diff()
                features[f'z_score_acceleration_{window}d'] = features[f'z_score_velocity_{window}d'].diff()
                
                # 3. Momentum Features
                features[f'z_score_momentum_{window}d'] = series.pct_change(periods=window)
                
                # 4. Distance from Moving Averages
                features[f'z_score_dist_from_ma_{window}d'] = (series - features[f'z_score_mean_{window}d']) / features[f'z_score_std_{window}d']
                
                # 5. Trend Calculation
                features[f'z_score_trend_{window}d'] = (features[f'z_score_mean_{window}d'] - features[f'z_score_mean_{window}d'].shift(window//2)) / (window//2)
            
            return features
        
        # Generate comprehensive features
        window_features = generate_features_for_windows(z_score, window_sizes)
        
        # Add generated features to the DataFrame
        for feature_name, feature_series in window_features.items():
            features_df[feature_name] = feature_series
        
        # 6. Regime Change Indicators
        def calculate_cusum(series, threshold=0.5):
            cusum_up = np.zeros_like(series)
            cusum_down = np.zeros_like(series)
            
            for i in range(1, len(series)):
                daily_change = series[i] - series[i-1]
                
                cusum_up[i] = max(0, cusum_up[i-1] + daily_change - threshold)
                cusum_down[i] = max(0, cusum_down[i-1] - daily_change - threshold)
            
            return cusum_up, cusum_down
        
        features_df['z_score_cusum_up'], features_df['z_score_cusum_down'] = calculate_cusum(z_score)
        
        # 7. Extreme Value Indicators
        extreme_thresholds = [1, 2, 3]
        for threshold in extreme_thresholds:
            features_df[f'z_score_above_{threshold}'] = (z_score > threshold).astype(int)
            features_df[f'z_score_below_neg_{threshold}'] = (z_score < -threshold).astype(int)
        
        # 8. Exponential Moving Averages
        for window in [3, 5, 7, 10, 20, 50, 100]:
            features_df[f'z_score_ema_{window}d'] = z_score.ewm(span=window, adjust=False).mean()
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def normalize_features(self, features_df):
        """
        Normalize the features using a rolling window approach.
        """
        # Create a copy of the features DataFrame
        normalized_df = pd.DataFrame(index=features_df.index)
        
        # Apply rolling normalization with a 252 window size to each numeric column
        rolling_window = 252
        
        rolling_mean = features_df.rolling(window=rolling_window, min_periods=1).mean()
        rolling_std = features_df.rolling(window=rolling_window, min_periods=1).std()
        
        # Handle potential zero standard deviation
        rolling_std = rolling_std.replace(0, 1)
        
        # Normalize the data by subtracting the rolling mean and dividing by the rolling std
        normalized_df = (features_df - rolling_mean) / rolling_std
        
        # Handling missing values after normalization
        normalized_df = normalized_df.fillna(0)
        
        return normalized_df
    
    def build_model(self):
        """
        Build a binary classification LSTM with techniques to ensure full-range predictions.
        """
        input_shape = (self.sequence_length, self.df.shape[1] - 4)  # Input shape for LSTM, -4 accounts for dropping of columns (duplicated z_score column, z_score_returns & label)
        
        # Define input layer
        inputs = keras.layers.Input(shape=input_shape)
        
        # Add noise for regularization
        x = keras.layers.GaussianNoise(0.1)(inputs)
        
        # First LSTM layer
        x = keras.layers.LSTM(128, activation='tanh', return_sequences=True)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)  # Increased dropout
        
        # Second LSTM layer
        x = keras.layers.LSTM(64, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.4)(x)  # Increased dropout
        
        # Dense layers with stronger activations
        x = keras.layers.Dense(32, kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)  # LeakyReLU instead of ReLU
        
        # Additional layer with strong activation
        x = keras.layers.Dense(16, kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        
        # Logits layer
        logits = keras.layers.Dense(1, kernel_initializer='glorot_normal', name='logits')(x)
        
        # Very aggressive temperature scaling (values < 0.5 push predictions to extremes)
        scaled_logits = keras.layers.Lambda(lambda x: x / 0.5, name='aggressive_scaling')(logits)
        
        # Output layer
        outputs = keras.layers.Activation('sigmoid', dtype='float32')(scaled_logits)
        
        # Build model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Use a lower learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.1, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model

    def calculate_exponential_weights(self, num_samples):
        """
        Calculate exponentially increasing weights for time-based weighting.
        This gives more importance to recent observations than linear weighting.
        """
        # Exponential weights with base 1.01
        weights = np.power(1.01, np.arange(num_samples))
        weights = weights / weights.sum()  # Normalize
        return weights

    def evaluate(self, data, normalized_df):
        """
        Evaluate the model performance.
        """
        print('Printing data from Evaluation')
        print(data)
        y_true = data['true_value'][20:].copy()
        print('Printing y_test from Evaluation')
        print(y_true)
        y_pred = data['prediction'][20:].copy()
        
        # Convert y_pred to binary classification: Long (1), Short (0)
        y_pred = np.where(y_pred >= 0.5, 1, 0)

        print('Printing before concatenating y_true and y_pred')
        print('Printing y_true from Evaluation')
        print(y_true)
        print('Printing y_pred from Evaluation')
        print(y_pred)
        print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        print("y_true counts:\n", pd.Series(y_true).value_counts())
        print("y_pred counts:\n", pd.Series(y_pred).value_counts())

        # Calculate and print accuracy before meta-labeling
        accuracy_before = accuracy_score(y_true, y_pred)
        print(f"Accuracy after Meta-Labelling: {accuracy_before:.4f}")

        # Compute AUC-ROC
        auc_roc = roc_auc_score(y_true, y_pred)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)

        # Convert y_test to binary classification: Long (1), Short (0)
        y_true = np.where(y_true >= 0.0, 1, 0)

        # Average Precision Score
        avg_precision = average_precision_score(y_true, y_pred)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        
        # Print comprehensive metrics
        print("\n--- Advanced Model Performance ---")
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        print(f"Average Precision Score: {avg_precision:.4f}")
        print(f"Prediction Threshold: {0.5}")
        
        # Detailed breakdown
        print("\nDetailed Classification Report:")
        print(f"Total Samples: {len(y_true)}")
        print(f"Positive Samples: {np.sum(y_true)}")
        print(f"Predicted Positives: {np.sum(y_pred)}")
        
        return {
            'auc_roc': auc_roc,
            'avg_precision': avg_precision
        }
    
    def plot_calibration_curve(self, final_df, n_bins=30, save_path="plots/Calibration/calibration_plot.png"):
        """
        Plot a calibration (reliability) curve using Plotly with a DodgerBlue theme and save as an image.
        """

        # Bin predictions into equal probability ranges
        final_df['bin'] = pd.cut(final_df['prediction_score'], bins=np.linspace(0, 1, n_bins + 1))

        # Compute mean predicted probability and actual outcome rate per bin
        calibration_data = final_df.groupby('bin').agg(
            mean_predicted=('prediction_score', 'mean'),
            actual_rate=('true_value', 'mean')
        ).reset_index()

        # Create calibration plot
        fig = go.Figure()

        # Add calibration curve
        fig.add_trace(go.Scatter(
            x=calibration_data["mean_predicted"], 
            y=calibration_data["actual_rate"],
            mode="lines+markers",
            line=dict(color="dodgerblue", width=3),
            marker=dict(size=7, color="dodgerblue", line=dict(width=1, color="black")),
            name="Calibration Curve"
        ))

        # Add diagonal reference line (perfect calibration)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="darkorange", width=2),
            name="Perfect Calibration"
        ))

        # Update layout for aesthetics
        fig.update_layout(
            title="Calibration Plot (Reliability Curve)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed Positive Rate",
            template="plotly_white",
            font=dict(family="Arial", size=14, color="black"),
            plot_bgcolor="white",
            width=700,
            height=500
        )

        # Save figure as an image
        fig.write_image(save_path)

        print(f"Plot saved to {save_path}")

    def run(self):
        """
        Run the entire model workflow with the correct setup for imbalanced binary classification.
        """
        # Prepare the data
        X, y, weights, preserved_dates, df, fft_num, arima_model = self.prepare_data()
        
        # Build the model
        self.model = self.build_model()
        print("Binary classification LSTM model built for imbalanced data.")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_roc',  # Keep ROC in check
                mode='max',  # Maximize ROC
                patience=20, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # Reduce LR if validation loss stagnates
                factor=0.5, 
                patience=10
            )
        ]
        
        # Train using Purged K-fold CV
        results = self.purged_k_cv.train_walk_forward(
            self.model,
            X, y, df, 
            original_indices=preserved_dates,
            sample_weights=weights,  # Ensure weights are passed
            fft_num=fft_num, # Passing the fft numbers list
            threshold=self.threshold,  # Binary classification threshold
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            callbacks=callbacks,
            arima_model=arima_model
        )
        
        return results


