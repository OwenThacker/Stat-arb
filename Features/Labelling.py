import pandas as pd
import numpy as np

class FeatureLabelling:
    def __init__(self):
        pass

    def normalize_features(self, pairs, globals_dict):
        """
        Drop price, returns, and log returns columns, and feature scale the remaining features using rolling normalization with a 252 window size.
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
    def __init__(self):
        self.period = 3
        self.SR = 1.0 # Sharpe Ratio
        self.Rf = 0.02 / 252 # Daily risk free rate (assuming 2% annual rate)
    
    def Combine_models(self, base_result: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
        '''Combining the results of the base and neural network models'''
        # Concatenate the predictions from the two models (base_result and final_df)
        combined_result = pd.concat([base_result['prediction'], final_df['prediction']], axis=1)

        # Use np.where to keep predictions where both models agree and set 0 where they disagree
        combined_result['combined'] = np.where(combined_result['prediction'].iloc[:, 0] == combined_result['prediction'].iloc[:, 1],
                                                combined_result['prediction'].iloc[:, 0], 0)

        # Return only the combined column
        return combined_result[['combined']]
    
    def Evaluate(self, combined_result: pd.DataFrame) -> float:
        '''Evaluating the combined model''' # Finish this later
        pass
    
    def Label(self, combined_result: pd.DataFrame, normalized_df: pd.DataFrame) -> pd.DataFrame:
        '''Labelling the combined model''' 
        print('Printing final_df:')
        print(normalized_df.head())
        print(normalized_df.columns)

        N = self.period
        SR_target = self.SR
        Rf = self.Rf = 0.02 / 252

        z_score = normalized_df['z_score']
        log_return = np.log(z_score / z_score.shift(1)).fillna(0)

        rolling_vol = log_return.rolling(window=N).std()

        threshold_long = SR_target * rolling_vol + Rf
        threshold_short = -SR_target * rolling_vol - Rf

        # Generate future return for label computation (sum of returns over N days)
        combined_result['future_return'] = log_return.rolling(window=N).sum().shift(-N)

        # Label the combined model
        combined_result['final_label'] = np.where(combined_result['combined'] == 1, 
                              np.where(combined_result['future_return'] > 0, 1, -1), 
                              0) # Turns values into 'Long' [1], 'Short' [-1], or 'Neutral' [0]
        
        print('Printing combined_result:')
        print(combined_result.head())

        return combined_result
        


