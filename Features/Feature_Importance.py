'''
This class contains the FeatureImportance class which is used to identify the most important features in the dataset
that has the most predictive power for the z_score returns of the pair. The class uses the XGBoost regressor model.

Classes:
    FeatureImportance: A comprehensive feature analysis tool for pairs trading that identifies the most predictive 
                      features for z-score returns using XGBoost regression with multicollinearity filtering.
'''

import os
import pandas as pd
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.tsa.arima.model import ARIMA

class FeatureImportance:
    """
    A feature importance analysis class for pairs trading using XGBoost regression.
    
    This class analyzes normalized pairs trading data to identify the most important features
    that predict z-score returns. It includes multicollinearity filtering, FFT and ARIMA
    feature engineering, and comprehensive visualization capabilities.
    
    Attributes:
        data (pd.DataFrame): Normalized input data containing features and z_score column
        X (pd.DataFrame): Feature matrix (independent variables)
        y (pd.Series): Target variable (binary classification based on Sharpe ratio)
        X_train (pd.DataFrame): Training feature set
        X_test (pd.DataFrame): Testing feature set  
        y_train (pd.Series): Training target values
        y_test (pd.Series): Testing target values
        model (xgb.XGBRegressor): Trained XGBoost model
        eval_result (dict): Model evaluation results during training
        stock1 (str): First stock symbol in the pair
        stock2 (str): Second stock symbol in the pair
        filtered_features (list): List of features after multicollinearity filtering
        vif_stats (pd.DataFrame): Variance Inflation Factor statistics for features
        dropped_features (list): Features removed during multicollinearity filtering
        period (int): Prediction horizon (N) in days, default 3
        SR (float): Sharpe ratio target threshold, default 0.5
        RF (float): Risk-free rate (daily), default 0
        
    """
    
    def __init__(self, stock1, stock2, normalized_data):
        """
        Initialize the FeatureImportance class with normalized pairs trading data.
        
        Args:
            stock1 (str): Symbol for the first stock in the pair
            stock2 (str): Symbol for the second stock in the pair  
            normalized_data (pd.DataFrame): Normalized dataset containing features and z_score column.
                                          Expected to have a 'z_score' column and various technical indicators.
                                          
        Note:
            The normalized_data should contain:
            - z_score: The pairs trading z-score values
            - Various technical indicators and features
            - Optional FFT reconstruction columns (z_score_fft_recon_*)
            - Optional ARIMA prediction column (arima_prediction)
        """
        self.data = normalized_data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.eval_result = None
        self.stock1 = stock1
        self.stock2 = stock2
        self.filtered_features = None
        self.vif_stats = None
        self.dropped_features = []
        self.period = 3 # N - Prediction Horizon
        self.SR = 0.5 # SF - Sharpe Ratio
        self.RF = 0 # 0.02 / 252 # RF - Risk Free Rate (daily, assumed 2% annual)
    
    def get_feature_importance_data(self):
        """
        Prepare data for machine learning by creating features and target variables.
        
        This method performs several critical data preparation steps:
        1. Calculates log returns from z_score values
        2. Computes future returns and volatility over the prediction horizon
        3. Creates binary target variable based on expected Sharpe ratio thresholds
        4. Splits data into training and testing sets (70/30)
        5. Performs FFT feature engineering for frequency domain analysis
        6. Applies ARIMA modeling for time series forecasting
        7. Normalizes FFT and ARIMA features using rolling statistics
        
        The target variable (y) is binary:
        - 1: Expected Sharpe ratio (long or short) exceeds the threshold
        - 0: Expected Sharpe ratio is below the threshold
        
        FFT Features:
        - Reconstructs z_score using top frequency components
        - Forecasts test period using pattern repetition from training cycles
        
        ARIMA Features:
        - Fits ARIMA(1,0,1) model to training z_score data
        - Generates forecasts for the test period
        
        Rolling Normalization:
        - Uses 252-day rolling window (approximately 1 trading year)
        - Standardizes features to have mean=0, std=1 within each window
        
        """
        data = self.data.copy()

        # Assigning variables for Prediction Horizon, Sharpe Ratio and Risk Free Rate
        N = self.period
        SR_target = self.SR

        # Calculate Log returns
        log_returns = np.log(data['z_score'] / data['z_score'].shift(1)).fillna(0)

        # Calculate future return: sum of next N returns
        future_return = log_returns.shift(-N).rolling(window=N).sum()

        # Calculate future volatility: standard deviation over the next N periods
        future_vol = log_returns.shift(-N).rolling(window=N).std()

        # Avoid division by zero
        safe_future_vol = future_vol.replace(0, np.nan)

        # Calculate expected Sharpe ratio for long and short trades
        expected_sharpe_long = future_return / safe_future_vol
        expected_sharpe_short = -future_return / safe_future_vol  # Short trades reverse the sign

        # Define y as a Pandas Series instead of NumPy array
        self.y = pd.Series(np.where(expected_sharpe_long > SR_target, 1, 
                                    np.where(expected_sharpe_short > SR_target, 1, 0)),
                        index=data.index)  # Align index with data after shift

        self.y = self.y.iloc[1:]  # Drop the first entry to match X's index

        # Ensure X and y have the same index to avoid misalignment
        X_temp = data.drop(columns=['z_score'])

        # Align X with y's index after pct_change
        self.X = X_temp.iloc[1:]  # Shift X to match y after pct_change

        self.X.replace([np.inf, -np.inf], np.nan).fillna(0)

        #print(f"Length of X: {len(self.X)}")
        #print(f"Length of y: {len(self.y)}")
        #print(f"X index range: {self.X.index[0]} to {self.X.index[-1]}")
        #print(f"y index range: {self.y.index[0]} to {self.y.index[-1]}")


        # Extract FFT and ARIMA columns
        fft_cols = [col for col in self.X.columns if col.startswith('z_score_fft_recon_')]
        arima_cols = ['arima_prediction'] if 'arima_prediction' in self.X.columns else []

        # Ensure X and y have the same length
        assert len(self.X) == len(self.y), "X and y must have the same length"

        # Split into train/test sets
        train_samples = int(self.X.shape[0] * 0.7)

        self.X_train, self.X_test = self.X.iloc[:train_samples], self.X.iloc[train_samples:]
        self.y_train, self.y_test = self.y.iloc[:train_samples], self.y.iloc[train_samples:]

        ### FFT Feature Engineering ###
        if fft_cols:
            z_scores_train = self.y_train.values  # Use the training z-score pct change series

            for fft_col in fft_cols:
                component_num = int(fft_col.split('_')[-1])  # Extract component number

                # Perform FFT on training data
                fft_result = fft.fft(z_scores_train)
                frequencies = fft.fftfreq(len(z_scores_train))

                # Select top components
                magnitude = np.abs(fft_result)
                idx = np.argsort(magnitude)[::-1][:component_num]

                # Reconstruct time series
                filtered_fft = np.zeros_like(fft_result, dtype=complex)
                filtered_fft[idx] = fft_result[idx]

                reconstruction_train = fft.ifft(filtered_fft).real

                # Forecast test period using last known cycle
                last_cycle = reconstruction_train[-component_num:]
                test_length = len(self.y_test)
                repetitions_needed = int(np.ceil(test_length / len(last_cycle)))
                forecast_pattern = np.tile(last_cycle, repetitions_needed)[:test_length]

                self.X_train[fft_col] = reconstruction_train
                self.X_train[fft_col].fillna(0, inplace=True)  # Replace NaNs with 0
                self.X_test[fft_col] = forecast_pattern
                self.X_test[fft_col].fillna(0, inplace=True)  # Replace NaNs with 0

        ### ARIMA Feature Engineering ###
        if arima_cols:
            z_scores_train = self.y_train.values  # Use training target

            try:
                arima_model = ARIMA(z_scores_train, order=(1, 0, 1))
                arima_fit = arima_model.fit()

                forecast_steps = len(self.y_test)
                forecast = arima_fit.forecast(steps=forecast_steps)

                self.X_train['arima_prediction'] = arima_fit.fittedvalues
                self.X_test['arima_prediction'] = forecast

            except Exception as e:
                #print(f"ARIMA model fitting failed: {e}")
                last_value = z_scores_train[-1]
                self.X_test['arima_prediction'] = np.full(len(self.y_test), last_value)

        ### Rolling Normalization for FFT and ARIMA ###
        rolling_window = 252
        columns_to_normalize = fft_cols + arima_cols

        for col in columns_to_normalize:
            rolling_mean = self.X_train[col].rolling(window=rolling_window, min_periods=1).mean()
            rolling_std = self.X_train[col].rolling(window=rolling_window, min_periods=1).std()

            last_mean = rolling_mean.iloc[-1]
            last_std = rolling_std.iloc[-1] if rolling_std.iloc[-1] > 0 else 1.0  # Avoid division by zero

            # Normalize training data
            self.X_train[col] = ((self.X_train[col] - rolling_mean) / rolling_std.replace(0, 1.0)).fillna(0)

            # Normalize test data using last train stats
            self.X_test[col] = ((self.X_test[col] - last_mean) / last_std)
    
    def filter_multicollinearity(self, correlation_threshold=0.80, vif_threshold=7.0):
        """
        Filter features to reduce multicollinearity using correlation and VIF analysis.
        
        This method implements a two-step filtering process:
        1. Correlation-based filtering: Removes features with pairwise correlation above threshold
        2. VIF-based filtering: Removes features with Variance Inflation Factor above threshold
        
        The process helps improve model stability and interpretability by removing redundant features
        that provide similar information.
        
        Args:
            correlation_threshold (float, optional): Maximum allowed correlation between features.
                                                   Features with correlation above this threshold
                                                   will be removed. Defaults to 0.80.
            vif_threshold (float, optional): Maximum allowed Variance Inflation Factor.
                                           Features with VIF above this threshold will be removed.
                                           Defaults to 7.0.
                                           
        Returns:
            pd.DataFrame: Filtered feature matrix with multicollinear features removed.
            
        Algorithm Details:
            Step 1 - Correlation Filtering:
            - Computes absolute correlation matrix for all features
            - Uses upper triangular matrix to avoid duplicate pairs
            - Removes one feature from each highly correlated pair
            
            Step 2 - VIF Filtering:
            - Calculates Variance Inflation Factor for each remaining feature
            - VIF measures how much the variance of a coefficient increases due to collinearity
            - Features with VIF > threshold indicate problematic multicollinearity
            
        Data Cleaning:
            - Replaces infinite values with NaN, then fills with column means
            - Handles constant columns (zero variance) by setting VIF to NaN
            - Caps extremely high VIF values at 1000 for numerical stability
            
        """
        import pandas as pd
        import numpy as np
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        #print(f"Starting multicollinearity filtering with {self.X.shape[1]} features")
        #print(f"Correlation threshold: {correlation_threshold}, VIF threshold: {vif_threshold}")
        
        # Make a copy of X and clean it
        X_clean = self.X.copy()
        
        # Replace inf, -inf with NaN, then fill NaNs with column means
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # For columns with all NaNs, fill with zeros instead of mean
        for col in X_clean.columns:
            if X_clean[col].isna().all():
                X_clean[col] = 0
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
        
        # Step 1: Filter based on correlation
        corr_matrix = X_clean.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_tri.columns 
                if any(upper_tri[column] > correlation_threshold)]
        
        #print(f"Found {len(to_drop)} features with correlation > {correlation_threshold}")
        
        # Step 2: Calculate VIF for remaining features
        X_filtered = X_clean.drop(columns=to_drop)
        
        # Initialize VIF dataframe
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_filtered.columns
        
        # Calculate VIF directly using variance_inflation_factor
        vif_values = []
        for i in range(len(X_filtered.columns)):
            try:
                # Check for constant columns (which cause VIF issues)
                if X_filtered.iloc[:, i].std() == 0:
                    #print(f"Warning: Column {X_filtered.columns[i]} has zero variance. Setting VIF to np.nan.")
                    vif_values.append(np.nan)
                    continue
                    
                # Calculate VIF directly
                vif = variance_inflation_factor(X_filtered.values, i)
                
                # Cap extremely high values for numerical stability
                if vif > 1000 or np.isinf(vif):
                    #print(f"Warning: Very high VIF for {X_filtered.columns[i]}. Capping at 1000.")
                    vif = 1000
                    
                vif_values.append(vif)
                
            except Exception as e:
                #print(f"Error calculating VIF for {X_filtered.columns[i]}: {e}")
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        
        # Sort by VIF value
        vif_data = vif_data.sort_values("VIF", ascending=False)
        
        # Drop features with NaN VIF
        nan_vif_features = vif_data[vif_data["VIF"].isna()]["Feature"].tolist()
        #print(f"Found {len(nan_vif_features)} features with NaN VIF")
        
        # Identify features with VIF > threshold
        high_vif_features = vif_data[(vif_data["VIF"] > vif_threshold) & (~vif_data["VIF"].isna())]["Feature"].tolist()
        #print(f"Found {len(high_vif_features)} features with VIF > {vif_threshold}")
        
        # Drop high VIF features and NaN VIF features
        to_drop.extend(high_vif_features)
        to_drop.extend(nan_vif_features)
        to_drop = list(set(to_drop))  # Remove duplicates
        
        # Final filtered feature set
        X_final = self.X.drop(columns=to_drop)
        
        #print(f"Final feature count after filtering: {X_final.shape[1]}")
        #print(f"Dropped {len(to_drop)} features due to multicollinearity or invalid VIF")
        
        # Store the filtered features and VIF stats
        self.filtered_features = X_final.columns.tolist()
        self.vif_stats = vif_data
        self.dropped_features = to_drop
        
        # Update training and test sets
        train_samples = int(X_final.shape[0] * 0.7)
        self.X_train = X_final.iloc[:train_samples]
        self.X_test = X_final.iloc[train_samples:]
        
        return X_final
    
    def train_model(self, use_filtered_features=True):
        """
        Train an XGBoost regressor model with optimized hyperparameters and regularization.
        
        This method trains a sophisticated XGBoost model designed for financial time series prediction.
        The model includes balanced regularization, early stopping, and optional feature filtering
        to prevent overfitting while maintaining predictive power.
        
        Args:
            use_filtered_features (bool, optional): Whether to use features filtered for multicollinearity.
                                                  If True, uses features from filter_multicollinearity().
                                                  If False, uses all available features.
                                                  Defaults to True.
            
        Model Configuration:
            - subsample=0.7: Uses 70% of samples per tree to prevent overfitting
            - reg_lambda=1.0: L2 regularization strength
            - reg_alpha=1.0: L1 regularization strength  
            - n_estimators=200: Maximum number of boosting rounds
            - max_depth=8: Maximum tree depth to control complexity
            - learning_rate=0.001: Conservative learning rate for stable training
            - gamma=0.1: Minimum loss reduction required for further partitioning
            - colsample_bytree=0.9: Fraction of features used per tree
            - min_child_weight=3: Minimum sum of instance weight in child node
            - early_stopping_rounds=100: Stop if no improvement for 100 rounds
            
        Sample Weighting:
            - Applies exponential weighting to give more importance to recent observations
            - Uses exponential growth pattern: exp(linspace(0.5, 1.0)) - 1
            - Normalizes weights to maintain original sample size importance
            - Only applied if training set has more than 100 samples
            
        Evaluation:
            - Monitors both training and validation RMSE during training
            - Uses early stopping to prevent overfitting
            - Tracks evaluation results for later analysis
            
        Data Preprocessing:
            - Replaces infinite values with NaN, then fills with 0
            - Ensures numerical stability for XGBoost training
            
        """
        import xgboost as xgb

        self.filtered_features = self.filter_multicollinearity().columns.tolist()
        
        # If filtered features are available and we want to use them
        if use_filtered_features and self.filtered_features is not None:
            X_train = self.X_train[self.filtered_features]
            X_test = self.X_test[self.filtered_features]
            #print(f"Training with {len(self.filtered_features)} filtered features")
        else:
            X_train = self.X_train
            X_test = self.X_test
            #print(f"Training with all {X_train.shape[1]} features")

        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # XGBoost regressor model
        regressor = xgb.XGBRegressor(
            subsample=0.7,              
            reg_lambda=1.0,            
            reg_alpha=1.0,               
            n_estimators=200,           
            max_depth=8,                
            learning_rate=0.001,          
            gamma=0.1,                   
            colsample_bytree=0.9,        
            min_child_weight=3,          
            early_stopping_rounds=100     
        )
        
        # Add sample weights to focus more on recent data if time series
        sample_weights = None
        if len(self.X_train) > 100:
            # Create exponential weights that give more importance to recent data
            sample_weights = np.linspace(0.5, 1.0, len(self.X_train))
            # Apply exponential growth to weights
            sample_weights = np.exp(sample_weights) - 1
            # Normalize weights
            sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

        self.model = regressor.fit(
            X_train, self.y_train, 
            sample_weight=sample_weights,
            eval_set=[(X_train, self.y_train), (X_test, self.y_test)], 
            verbose=False
        )
        
        self.eval_result = self.model.evals_result()
        
        # print minimum validation error and corresponding iteration
        val_errors = self.eval_result['validation_1']['rmse']
        min_val_error = min(val_errors)
        min_val_iter = val_errors.index(min_val_error)
        #print(f"Minimum validation error: {min_val_error:.6f} at iteration {min_val_iter}")
    
    def plot_simplified_analysis(self, use_filtered_features=True):
        """
        Generate comprehensive visualization analysis of model performance and feature relationships.
        
        This method creates a multi-panel visualization dashboard that provides insights into:
        1. Model training progression and overfitting detection
        2. Feature importance rankings and values
        3. Feature correlation structure and multicollinearity patterns  
        4. Pairwise relationships between top predictive features
        
        The visualizations are saved as high-resolution PNG files for reporting and analysis.
        
        Args:
            use_filtered_features (bool, optional): Whether to analyze filtered features or all features.
                                                  If True, uses features from multicollinearity filtering.
                                                  If False, uses all original features.
                                                  Defaults to True.
                                                  
        Returns:
            list: List of top feature names ranked by importance (up to top 10 features).
            
        Visualizations Created:
            
            1. Training vs Validation RMSE (Top Left Panel):
               - Line plot showing RMSE progression over training rounds
               - Helps identify overfitting (divergence between curves)
               - Shows early stopping effectiveness
               
            2. Feature Importance Bar Chart (Top Right Panel):
               - Horizontal bar chart of top 10 most important features
               - Annotated with exact importance values  
               - Features ranked by XGBoost importance scores
               
            3. Feature Correlation Heatmap (Bottom Panel - Full Width):
               - Correlation matrix heatmap for top 10 features
               - Color-coded from -1 (negative) to +1 (positive correlation)
               - Annotated with correlation coefficients
               - Helps identify multicollinear feature relationships
               
            4. Feature Pairplot (Separate Figure):
               - Pairwise scatter plots for top 5 features
               - Includes target variable (z_score_returns) relationships
               - Diagonal shows kernel density estimation plots
               - Corner plot layout to avoid redundancy
        
        File Outputs:
            - '{stock1}_{stock2}_feature_importance.png': Main dashboard with 3 panels
            - '{stock1}_{stock2}_pairplot.png': Separate pairplot visualization
            - Saved to 'plots/Feature_Importance/' directory (created if doesn't exist)
            - High resolution (300 DPI) for publication quality
            
        Console Output:
            - File save locations and confirmation messages
            - Ranked list of top N most important features with scores
            - Feature importance values printed to 4 decimal places
            
        Technical Details:
            - Uses matplotlib GridSpec for custom layout proportions
            - Bottom correlation panel is taller (1.5x) than top panels
            - Pairplot limited to top 5 features to maintain readability
            - Correlation heatmap uses blue color scheme for consistency
            - All plots use consistent styling and color schemes
            
        Directory Structure:
            Creates: plots/Feature_Importance/{stock1}_{stock2}_*.png
 
        Example Output Files:
            - plots/Feature_Importance/AAPL_MSFT_feature_importance.png
            - plots/Feature_Importance/AAPL_MSFT_pairplot.png
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Ensure the plots directory exists
        directory_path = 'plots/Feature_Importance'
        os.makedirs(directory_path, exist_ok=True)

        stock1, stock2 = self.stock1, self.stock2
        
        # Choose which feature set to use
        feature_importance = self.model.feature_importances_
        if use_filtered_features and self.filtered_features is not None:
            feature_names = self.filtered_features
        else:
            feature_names = self.X.columns.tolist()
        
        # Sort features by importance
        importance_dict = {feature_names[i]: feature_importance[i] for i in range(len(feature_names))}
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_n = min(10, len(sorted_importance))
        
        # Create a 2x2 grid but with different proportions
        fig = plt.figure(figsize=(14, 12))  # Reduced vertical height
        
        # Define grid specs for custom layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, height_ratios=[1, 1.5])  # Make bottom row taller for correlation plot
        
        # Training vs Validation Error - Top left
        ax1 = fig.add_subplot(gs[0, 0])
        training_rounds = range(len(self.eval_result['validation_0']['rmse']))
        ax1.plot(training_rounds, self.eval_result['validation_0']['rmse'], label='Training', color='royalblue')
        ax1.plot(training_rounds, self.eval_result['validation_1']['rmse'], label='Validation', color='orange')
        ax1.set_title('Training vs Validation RMSE')
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('RMSE')
        ax1.legend()
        
        # Feature Importance - Top right
        sorted_idx = np.argsort(feature_importance)[-10:]
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.barh([feature_names[i] for i in sorted_idx], 
                    [feature_importance[i] for i in sorted_idx], 
                    color='royalblue', edgecolor='black')
        
        # Annotate bars with values
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.001, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', 
                    ha='left', va='center')
        
        ax2.set_title('Feature Importance (Top 10)')
        ax2.set_xlabel('Importance')
        ax2.set_ylabel('Features')
        
        # Feature Correlation Heatmap - Full width on bottom
        ax3 = fig.add_subplot(gs[1, :])  # Span both columns
        top_features = [feature_names[i] for i in sorted_idx]
        corr_matrix = self.X[top_features].corr()
        sns.heatmap(corr_matrix, cmap='Blues', vmax=1, vmin=-1, center=0,
                    annot=True, fmt='.2f', square=True, linewidths=0.5, ax=ax3)
        ax3.set_title('Feature Correlation')
        
        plt.tight_layout()
        plt.savefig(f'{directory_path}/{stock1}_{stock2}_feature_importance.png', dpi=300, bbox_inches='tight')
        
        # Pairplot for top features (separate figure)
        if len(top_features) >= 2:
            features_for_pairplot = top_features[:min(5, len(top_features))]
            
            pairplot_df = self.X[features_for_pairplot].copy()
            pairplot_df['z_score_returns'] = self.y.values
            
            sns.pairplot(pairplot_df, corner=True, diag_kind='kde',
                        plot_kws={'alpha': 0.6, 'edgecolor': 'black', 's': 40, 'color': 'royalblue'},
                        diag_kws={'color': 'royalblue', 'shade': True, 'linewidth': 2})
            
            plt.suptitle(f'{stock1}-{stock2} Feature Relationships (Top {len(features_for_pairplot)} Features)',
                        fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(f'{directory_path}/{stock1}_{stock2}_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        #print information to console
        #print(f"Feature importance plots saved to {directory_path}/feature_importance_{stock1}_{stock2}.png")
        #print(f"Feature pairplot saved to {directory_path}/feature_pairplot_{stock1}_{stock2}.png")
        #print(f"Top {top_n} Important Features:")
        # for i, (feature, importance) in enumerate(sorted_importance[:top_n]):
        #     print(f"{i+1}. {feature}: {importance:.4f}")
        
        return top_features