import pandas as pd
import numpy as np

class FeatureLabelling:
    """
    A class for preprocessing and normalizing trading pair features for machine learning models.
    
    This class handles the normalization of financial time series data by dropping price-related
    columns and applying rolling normalization to prepare features for model training.
    """
    
    def __init__(self):
        """
        Initialize the FeatureLabelling class.
        
        Currently no initialization parameters are required.
        """
        pass

    def normalize_features(self, pairs, globals_dict):
        """
        Drop price, returns, and log returns columns, and feature scale the remaining features 
        using rolling normalization with a 252 window size.
        
        This method processes trading pairs by:
        1. Removing price-based columns (prices, returns, log returns, spread data)
        2. Converting all data to numeric format
        3. Applying rolling normalization with a 252-day window
        4. Reattaching non-numeric columns and z_score
        
        Args:
            pairs (list): List of tuples containing stock pair names (e.g., [('AAPL', 'MSFT')])
            globals_dict (dict): Dictionary containing DataFrames with naming pattern 
                                '{stock1}_{stock2}_total_df'
        
        Returns:
            pd.DataFrame: The final normalized DataFrame for the last processed pair
            
        Note:
            - Modifies the DataFrames in globals_dict in-place
            - Uses 252-day rolling window (typical trading days in a year)
            - Fills NaN values with 0 after normalization
            - Prints processing status and preview for each pair
        """
        for pair in pairs:
            stock1 = pair[0].replace(' ', '_')
            stock2 = pair[1].replace(' ', '_')
            pair_name = f'{stock1}_{stock2}_total_df'
            pair_df = globals_dict[pair_name]

            pair_df = pair_df.fillna(0)

            # Drop price, returns, and log returns columns
            pair_df = pair_df.drop(columns=[stock1, stock2, f'{stock1}_returns', f'{stock2}_returns', f'{stock1}_log_returns', f'{stock2}_log_returns', 'spread', 'spread_mean', 'spread_std'])

            # Ensure everything is numeric (and coerce any errors to NaN)
            pair_df = pair_df.apply(pd.to_numeric, errors='coerce')

            # Drop any rows with NaN values (after coercion)
            pair_df = pair_df.dropna()

            # Store the z_score column
            z_score_column = pair_df['z_score']
            pair_df = pair_df.drop(columns=['z_score'])  # Remove z_score column for normalization

            pair_df = pair_df.astype(float)

            # Identify only numeric columns (exclude int64 columns and z_score)
            numeric_columns = pair_df.select_dtypes(include=[float]).columns

            '''
            This section is the standard rolling scaling of the data to help with appropriate normalization.
            '''

            # Apply rolling normalization with a 252 window size to the numeric columns
            rolling_window = 252
            rolling_mean = pair_df[numeric_columns].rolling(window=rolling_window, min_periods=1).mean()
            rolling_std = pair_df[numeric_columns].rolling(window=rolling_window, min_periods=1).std()

            # Normalize the data by subtracting the rolling mean and dividing by the rolling std
            normalized_df = (pair_df[numeric_columns] - rolling_mean) / rolling_std
            
            # Handling missing values after normalization
            normalized_df = normalized_df.fillna(0)  
            normalized_df = normalized_df.dropna()

            # Reattach the non-numeric columns (int64 and z_score) to the normalized dataframe
            non_numeric_columns = pair_df.select_dtypes(exclude=[float]).columns
            final_df = pd.concat([normalized_df, pair_df[non_numeric_columns]], axis=1)

            # Reattach the z_score column 
            final_df['z_score'] = z_score_column

            # Update the global variable with the new DataFrame
            globals_dict[pair_name] = final_df

            print(f'{stock1}_{stock2}_total_df normalized successfully.')
            print(final_df.head())
            return final_df

class Combiner:
    """
    A class for combining predictions from multiple trading models and generating trading labels.
    
    This class combines predictions from base and neural network models, evaluates performance,
    and generates final trading labels (Long/Short/Neutral) based on future returns.
    
    Attributes:
        period (int): Number of days to look ahead for return calculation (default: 3)
        SR (float): Target Sharpe Ratio threshold (default: 0.5)
        Rf (float): Risk-free rate for calculations (default: 0)
    """
    
    def __init__(self):
        """
        Initialize the Combiner class with default trading parameters.
        
        Sets up the evaluation period, Sharpe ratio threshold, and risk-free rate
        for model combination and labeling.
        """
        self.period = 3
        self.SR = 0.5 # Sharpe Ratio
        self.Rf = 0 # 0.02 / 252 # Daily risk free rate (assuming 2% annual rate)
    
    def Combine_models(self, base_result: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine the results of the base and neural network models.
        
        This method creates a consensus prediction by only keeping signals where both
        models agree, setting disagreements to neutral (0).
        
        Args:
            base_result (pd.DataFrame): DataFrame containing predictions from the base model
                                       Must have a 'prediction' column
            final_df (pd.DataFrame): DataFrame containing predictions from the neural network model
                                    Must have a 'prediction' column
        
        Returns:
            pd.DataFrame: DataFrame with a single 'combined' column containing:
                         - Original prediction value if both models agree
                         - 0 (neutral) if models disagree

        """
        # Concatenate the predictions from the two models (base_result and final_df)
        combined_result = pd.concat([base_result['prediction'], final_df['prediction']], axis=1)

        # Use np.where to keep predictions where both models agree and set 0 where they disagree
        combined_result['combined'] = np.where(combined_result['prediction'].iloc[:, 0] == combined_result['prediction'].iloc[:, 1],
                                                combined_result['prediction'].iloc[:, 0], 0)

        # Return only the combined column
        return combined_result[['combined']]
    
    def Evaluate(self, combined_result: pd.DataFrame) -> float:
        pass
    
    def Label(self, combined_result: pd.DataFrame, normalized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate final trading labels based on combined model predictions and future returns.
        
        This method creates trading labels by:
        1. Calculating future returns using z_score log returns
        2. Looking ahead N periods to determine if the trade would be profitable
        3. Assigning labels: 1 (Long), -1 (Short), or 0 (Neutral)
        
        Args:
            combined_result (pd.DataFrame): DataFrame with combined model predictions
                                          Must have a 'combined' column
            normalized_df (pd.DataFrame): Normalized DataFrame containing z_score column
        
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional columns:
                         - 'future_return': Sum of log returns over next N periods
                         - 'final_label': Trading label (1=Long, -1=Short, 0=Neutral)
        
        Label Logic:
            - If combined prediction = 1 (Long signal):
                * Label = 1 if future_return > 0 (profitable long)
                * Label = -1 if future_return <= 0 (unprofitable long)
            - If combined prediction = 0 (Neutral): Label = 0
            - If combined prediction = -1 (Short signal):
                * Label = 1 if future_return > 0 (unprofitable short)
                * Label = -1 if future_return <= 0 (profitable short)
        
        """
        print('Printing final_df:')
        print(normalized_df.head())
        print(normalized_df.columns)

        N = self.period

        z_score = normalized_df['z_score']
        log_return = np.log(z_score / z_score.shift(1)).fillna(0)

        # Generate future return for label computation (sum of returns over N days)
        combined_result['future_return'] = log_return.shift(-N).rolling(window=N).sum()

        # Label the combined model
        combined_result['final_label'] = np.where(combined_result['combined'] == 1, 
                              np.where(combined_result['future_return'] > 0, 1, -1), 
                              0) # Turns values into 'Long' [1], 'Short' [-1], or 'Neutral' [0]
        
        print('Printing combined_result:')
        print(combined_result.head())

        return combined_result