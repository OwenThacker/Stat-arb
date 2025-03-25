import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy.fft as fft

class Purged_K_CV:

    def __init__(self, n_splits=3, purge_window=10, patience=5):
        """
        Initialize the Purged_K_CV class.
        """
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.patience = patience
        self.scaler_logistic = StandardScaler()
        self.scaler_nn = RobustScaler()
        

    def train_walk_forward_Logistic(self, model, X, y, df, original_indices, sample_weights, batch_size=32, epochs=10):
        """
        Perform walk-forward (expanding window) cross-validation with purging and early stopping based on balanced recall.
        Handles cases without data leakage for FFT and ARIMA features for each fold.
        """
        
        from sklearn.metrics import recall_score, precision_recall_curve, confusion_matrix
        import numpy as np
        
        n_samples = len(X)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Identify FFT and ARIMA columns
        fft_cols = [col for col in X.columns if col.startswith('z_score_fft_recon_')]
        arima_cols = [col for col in X.columns if col == 'arima_prediction']

        X_original = X.copy()
        z_score_col = 'z_score_returns'  # Update this with actual column name
        print('Logistic - Printing columns')
        print(X_original.columns)

        pred_proba = np.zeros(n_samples)  # Store raw probabilities
        pred_labels = np.zeros(n_samples)  # Store predicted labels
        
        # Store the optimal threshold for each fold
        fold_thresholds = []

        # DataFrame to store results
        results_data = {
            'true_value': y,
            'prediction': np.zeros(n_samples),
            'probability': np.zeros(n_samples),  # Store raw probabilities
            'fold': np.zeros(n_samples),
            'prediction_source': np.array(['not_predicted'] * n_samples, dtype=object)
        }

        fold_size = n_samples // self.n_splits
        min_train_size = fold_size

        # Early stopping setup
        best_macro_recall = -1
        best_model = None
        best_threshold = 0.5  # Initial default threshold

        for fold in range(self.n_splits - 1):  
            print(f"Training fold {fold + 1}/{self.n_splits}")

            # Expanding training window for each fold
            train_end = min_train_size if fold == 0 else (fold + 1) * fold_size
            test_start = train_end
            test_end = min(n_samples, test_start + fold_size)

            # Apply purging around the boundary between train and test
            boundary = train_end - 1
            purge_start = max(0, boundary - self.purge_window + 1)
            purge_end = min(n_samples, test_start + self.purge_window)

            # Create train and test indices with purging
            train_idx = np.arange(0, purge_start)
            test_idx = np.arange(purge_end, test_end)

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue  

            X_fold = X_original.copy()

            # Recalculate FFT features
            if fft_cols:
                print(f'Performing FFT for fold: {fold}')
                z_scores_train = df.iloc[:train_end][z_score_col].values

                for fft_col in fft_cols:
                    component_num = int(fft_col.split('_')[-1])
                    
                    fft_result = fft.fft(z_scores_train)
                    frequencies = fft.fftfreq(len(z_scores_train))

                    magnitude = np.abs(fft_result)
                    idx = np.argsort(magnitude)[::-1][:component_num]

                    filtered_fft = np.zeros_like(fft_result, dtype=complex)
                    filtered_fft[idx] = fft_result[idx]

                    reconstruction_train = fft.ifft(filtered_fft).real

                    test_length = len(test_idx)
                    last_cycle = reconstruction_train[-component_num:]

                    repetitions_needed = int(np.ceil(test_length / len(last_cycle)))
                    forecast_pattern = np.tile(last_cycle, repetitions_needed)[:test_length]

                    X_fold.loc[X_fold.index[:train_end], fft_col] = reconstruction_train 
                    X_fold.loc[X_fold.index[test_start:test_start + len(forecast_pattern)], fft_col] = forecast_pattern 

            # Recalculate ARIMA features
            if arima_cols:
                z_scores_train = df.iloc[:train_end][z_score_col].values
                
                try:
                    arima_model = ARIMA(z_scores_train, order=(1, 0, 1))
                    arima_fit = arima_model.fit()

                    forecast_steps = len(test_idx)
                    forecast = arima_fit.forecast(steps=forecast_steps)

                    X_fold.loc[X_fold.index[:train_end], 'arima_prediction'] = arima_fit.fittedvalues
                    X_fold.loc[X_fold.index[test_start:test_end], 'arima_prediction'] = forecast

                except Exception as e:
                    print(f"ARIMA model fitting failed: {e}")
                    last_value = z_scores_train[-1]
                    X_fold.loc[X_fold.index[test_start:test_end], 'arima_prediction'] = last_value

            # Apply rolling normalization ONLY to FFT and ARIMA columns
            rolling_window = 252
            columns_to_normalize = fft_cols + arima_cols
            
            for col in columns_to_normalize:
                # Calculate rolling stats up to train_end only to prevent leakage
                rolling_mean = X_fold[col][:train_end].rolling(window=rolling_window, min_periods=1).mean()
                rolling_std = X_fold[col][:train_end].rolling(window=rolling_window, min_periods=1).std()
                
                # Use the last calculated mean and std for the test data to avoid leakage
                last_mean = rolling_mean.iloc[-1]
                last_std = rolling_std.iloc[-1] if rolling_std.iloc[-1] > 0 else 1.0  # Avoid division by zero
                
                # Normalize training data
                X_fold.loc[X_fold.index[:train_end], col] = (
                    (X_fold.loc[X_fold.index[:train_end], col] - rolling_mean) / 
                    rolling_std.replace(0, 1.0)  # Replace zeros with 1.0 to avoid division by zero
                ).fillna(0)
                
                # Normalize test data using last train stats
                X_fold.loc[X_fold.index[test_start:test_end], col] = (
                    (X_fold.loc[X_fold.index[test_start:test_end], col] - last_mean) / 
                    last_std
                )

            # Get the updated training and testing data for this fold
            X_train = X_fold.iloc[train_idx].values
            y_train = y[train_idx]
            X_test = X_fold.iloc[test_idx].values
            y_test = y[test_idx]

            # Add this debugging code before model training
            print("Feature statistics before scaling:")
            for col in X_fold.columns:
                print(f"{col}: mean={X_fold[col].mean():.4f}, std={X_fold[col].std():.4f}, min={X_fold[col].min():.4f}, max={X_fold[col].max():.4f}")

            # Apply scaling for Logistic Regression - on top of normalization
            X_train_scaled = self.scaler_logistic.fit_transform(X_train)
            X_test_scaled = self.scaler_logistic.transform(X_test)

            print(f"X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")

            fold_weights = None
            if sample_weights is not None:
                fold_weights = sample_weights[train_idx]

            # Use scaled data for model fitting
            model.fit(X_train_scaled, y_train, sample_weight=fold_weights)

            # Use scaled data for predictions
            fold_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Check class distribution in test set
            class_0_count = np.sum(y_test == 0)
            class_1_count = np.sum(y_test == 1)
            class_ratio = class_1_count / max(1, len(y_test))  # Avoid division by zero
            print(f"Class distribution in test set - Class 0: {class_0_count} ({(1-class_ratio)*100:.2f}%), " +
                f"Class 1: {class_1_count} ({class_ratio*100:.2f}%)")
            
            # Print probability distribution statistics
            print(f"Probability distribution - Min: {np.min(fold_pred_proba):.4f}, Mean: {np.mean(fold_pred_proba):.4f}, " +
                f"Max: {np.max(fold_pred_proba):.4f}")
            print(f"Quantiles - 25%: {np.quantile(fold_pred_proba, 0.25):.4f}, 50%: {np.quantile(fold_pred_proba, 0.5):.4f}, " +
                f"75%: {np.quantile(fold_pred_proba, 0.75):.4f}")
            
            # Simpler approach to find the best threshold that balances recall
            # Try a range of thresholds and pick the one that gives the most similar recalls for both classes
            thresholds = np.linspace(0.1, 0.9, 100)
            best_diff = float('inf')
            best_threshold_for_fold = 0.5
            best_recalls = (0, 0)
            best_macro_recall_for_fold = 0
            
            # For debugging: store all threshold results
            all_threshold_results = []
            
            for threshold in thresholds:
                temp_preds = (fold_pred_proba >= threshold).astype(int)
                
                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_test, temp_preds).ravel()
                
                # Calculate recalls
                # Recall for class 0 = TN / (TN + FP)
                recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
                # Recall for class 1 = TP / (TP + FN)
                recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Calculate absolute difference between recalls
                diff = abs(recall_0 - recall_1)
                macro_recall = (recall_0 + recall_1) / 2
                
                all_threshold_results.append((threshold, recall_0, recall_1, diff, macro_recall))
                
                # We want the smallest difference, but both recalls should be above a minimum value
                min_acceptable_recall = 0.1  # Both classes should have at least 10% recall
                
                if recall_0 >= min_acceptable_recall and recall_1 >= min_acceptable_recall:
                    # Prefer thresholds that give higher macro recall when differences are similar
                    if diff < best_diff or (abs(diff - best_diff) < 0.05 and macro_recall > best_macro_recall_for_fold):
                        best_diff = diff
                        best_threshold_for_fold = threshold
                        best_recalls = (recall_0, recall_1)
                        best_macro_recall_for_fold = macro_recall
            
            # If we couldn't find a good threshold with acceptable recalls for both classes,
            # fall back to maximizing macro recall
            if best_macro_recall_for_fold == 0:
                print("Couldn't find threshold with acceptable recalls for both classes. Falling back to maximizing macro recall.")
                for threshold, recall_0, recall_1, diff, macro_recall in all_threshold_results:
                    if macro_recall > best_macro_recall_for_fold:
                        best_macro_recall_for_fold = macro_recall
                        best_threshold_for_fold = threshold
                        best_recalls = (recall_0, recall_1)
            
            print(f"Optimal threshold for fold {fold + 1}: {best_threshold_for_fold:.4f}")
            print(f"Individual recalls at optimal threshold - Class 0: {best_recalls[0]:.4f}, Class 1: {best_recalls[1]:.4f}")
            print(f"Macro recall at optimal threshold: {best_macro_recall_for_fold:.4f}")
            fold_thresholds.append(best_threshold_for_fold)
            
            # Apply the optimal threshold
            fold_pred_labels = (fold_pred_proba >= best_threshold_for_fold).astype(int)
            
            # Log the confusion matrix
            cm = confusion_matrix(y_test, fold_pred_labels)
            print("Confusion Matrix:")
            print(cm)
            
            # Calculate recall for both classes
            recall_0 = best_recalls[0]
            recall_1 = best_recalls[1]
            fold_macro_recall = (recall_0 + recall_1) / 2
            
            print(f"Fold {fold + 1} - Recall (class 0): {recall_0:.4f}, Recall (class 1): {recall_1:.4f}, Macro-Recall: {fold_macro_recall:.4f}")
            print(f"Predictions distribution - Class 0: {np.sum(fold_pred_labels == 0)}, Class 1: {np.sum(fold_pred_labels == 1)}")

            pred_proba[test_idx] = fold_pred_proba
            pred_labels[test_idx] = fold_pred_labels

            # Save the best model based on macro-averaged recall
            if fold_macro_recall > best_macro_recall:
                best_macro_recall = fold_macro_recall
                best_model = model
                best_threshold = best_threshold_for_fold
                print(f"New best macro-averaged recall: {best_macro_recall:.4f} with threshold: {best_threshold:.4f}")

            # Add after model training
            coefficients = pd.DataFrame({
                'Feature': X_fold.columns,
                'Coefficient': model.coef_[0]
            })
            print("Feature importance:")
            print(coefficients.sort_values(by='Coefficient', ascending=False))

        if best_model is None:
            print("Warning: No model improved macro-averaged recall")
            best_model = model  # Use the original model
            best_threshold = 0.5  # Default threshold
        else:
            print(f"Best model has macro-averaged recall of {best_macro_recall:.4f} with threshold: {best_threshold:.4f}")
        
        # Calculate average threshold across folds as an alternative option
        # if fold_thresholds:
        #     avg_threshold = sum(fold_thresholds) / len(fold_thresholds)
        #     print(f"Average optimal threshold across folds: {avg_threshold:.4f}")
            
        #     # Use a balanced approach: consider both the best single threshold and the average
        #     final_threshold = (best_threshold + avg_threshold) / 2
        #     print(f"Using balanced threshold (average of best and avg): {final_threshold:.4f}")
        # else:
        #     final_threshold = best_threshold

        final_threshold = best_threshold
        
        # Final predictions for test set
        # Get the updated full dataset with all recalculated features
        X_final = X_fold.copy()
        
        # Make predictions on the test set with best model using scaled data
        test_idx = np.arange(min_train_size, n_samples)  # Test set for final evaluation
        if len(test_idx) > 0:
            X_test_final = X_final.iloc[test_idx].values
            X_test_scaled = self.scaler_logistic.transform(X_test_final)
            
            if hasattr(best_model, 'predict_proba'):
                test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            else:
                test_proba = best_model.predict(X_test_scaled)
            
            test_preds = np.where(test_proba >= final_threshold, 1, 0)
            
            for i, idx in enumerate(test_idx):
                results_data['probability'][idx] = test_proba[i]
                results_data['prediction'][idx] = test_preds[i]
                results_data['fold'][idx] = self.n_splits
                results_data['prediction_source'][idx] = 'test_pred'
        
        # Make predictions on the training data with best model using scaled data
        train_idx = np.arange(0, min_train_size)  # Training set for final evaluation
        X_train_final = X_final.iloc[train_idx].values
        X_train_scaled = self.scaler_logistic.transform(X_train_final)
        
        if hasattr(best_model, 'predict_proba'):
            train_proba = best_model.predict_proba(X_train_scaled)[:, 1]
        else:
            train_proba = best_model.predict(X_train_scaled)

        print("Training probabilities percentiles:")
        print(np.percentile(train_proba, [0, 25, 50, 75, 100]))
            
        train_preds = np.where(train_proba >= final_threshold, 1, 0)
        
        for i, idx in enumerate(train_idx):
            results_data['probability'][idx] = train_proba[i]
            results_data['prediction'][idx] = train_preds[i]
            results_data['fold'][idx] = 0
            results_data['prediction_source'][idx] = 'train_pred'

        # Probability Distribution Analysis
        plt.figure(figsize=(10, 6))
        
        # Split probabilities by true class for better analysis
        probs_class_0 = [pred_proba[i] for i in range(len(pred_proba)) if y[i] == 0]
        probs_class_1 = [pred_proba[i] for i in range(len(pred_proba)) if y[i] == 1]
        
        plt.hist(probs_class_0, bins=30, alpha=0.6, color='blue', edgecolor='black', label='Class 0')
        plt.hist(probs_class_1, bins=30, alpha=0.6, color='red', edgecolor='black', label='Class 1')
        plt.axvline(final_threshold, color='green', linestyle='dashed', linewidth=2, label=f'Threshold: {final_threshold:.4f}')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(label="Distribution of Predicted Probabilities by Class", fontsize=16)
        plt.legend()
        plt.savefig("plots\\Logistic\\probability_distribution_by_class.png")
        
        # Overall probability distribution
        plt.figure(figsize=(8, 5))
        plt.hist(pred_proba, bins=30, alpha=0.75, color='dodgerblue', edgecolor='white')
        plt.axvline(final_threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold: {final_threshold:.4f}')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(label="Distribution of Predicted Probabilities", fontsize=18, color='dodgerblue')
        plt.legend()
        plt.savefig("plots\\Logistic\\probability_distribution.png")
        
        # Class distribution analysis for debugging
        class_dist = pd.DataFrame({
            'Actual': results_data['true_value'],
            'Predicted': results_data['prediction']
        })
        print("Class distribution summary:")
        print(class_dist.groupby(['Actual', 'Predicted']).size())
        
        # Calculate final recalls
        final_recall_0 = recall_score(results_data['true_value'], results_data['prediction'], pos_label=0)
        final_recall_1 = recall_score(results_data['true_value'], results_data['prediction'], pos_label=1)
        final_macro_recall = (final_recall_0 + final_recall_1) / 2
        print(f"Final Recall (class 0): {final_recall_0:.4f}, Recall (class 1): {final_recall_1:.4f}, Macro-Recall: {final_macro_recall:.4f}")

        # Return results as DataFrame
        results_df = pd.DataFrame(results_data, index=original_indices)
        return results_df
    
    def train_walk_forward(self, model, X, y, df, original_indices, sample_weights, fft_num, threshold=0.5, batch_size=32, epochs=10, callbacks=None, arima_model=False, arima_order=(1,0,1)):
        """
        Perform walk-forward (expanding window) cross-validation with purging. Handles cases without data leakage for FFT and ARIMA features.
        """
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import precision_recall_curve, f1_score
        from statsmodels.tsa.arima.model import ARIMA
        import numpy.fft as fft
        import matplotlib.pyplot as plt
        
        n_samples = len(X)
        
        # Print overall dimensions and class distribution
        print(f"Overall X shape: {X.shape}")
        print(f"Overall y shape: {y.shape}")
        print(f"Class distribution in y: {np.bincount(y.astype(int))}")
        
        # Store original X
        original_feature_count = X.shape[2]
        print(f"Original feature count: {original_feature_count}")
        
        # Calculate how many features we'll add (one for each FFT component number)
        total_fft_features = len(fft_num)
        
        # Add one feature for ARIMA if enabled
        total_arima_features = 1 if arima_model else 0
        
        # Create an expanded tensor to hold original features plus FFT and ARIMA features
        # Shape: (samples, sequence_length, original_features + fft_features + arima_features)
        X_expanded = np.zeros((X.shape[0], X.shape[1], X.shape[2] + total_fft_features + total_arima_features))
        
        # Copy original features to the expanded tensor
        X_expanded[:, :, :original_feature_count] = X
        
        # Find z_score_returns column in dataframe
        z_score_col = 'z_score_returns'
        if z_score_col in df.columns:
            print(f"Found {z_score_col} in dataframe")
        else:
            print(f"Warning: {z_score_col} not found in dataframe. Using first numeric column as fallback.")
            z_score_col = df.select_dtypes(include=[np.number]).columns[0]
            print(f"Using {z_score_col} as fallback")
        
        # Create a DataFrame to store all results
        results_data = {
            'true_value': y,
            'prediction': np.zeros(n_samples),
            'fold': np.zeros(n_samples),
            'prediction_source': np.array(['not_predicted'] * n_samples, dtype=object),
            'prediction_score': np.zeros(n_samples)
        }
        
        # Calculate fold size
        fold_size = n_samples // self.n_splits
        min_train_size = fold_size

        # Initialize EarlyStopping callback with more patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_precision', 
            patience=10,
            mode='max',
            verbose=1, 
            restore_best_weights=True
        )
        
        # Add learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_precision',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Store predictions for global calibration
        all_train_preds = []
        all_train_labels = []
        fold_test_preds = []
        fold_test_indices = []
        
        # Store initial model weights
        try:
            initial_weights = 'Model_weights/initial_model_weights.weights.h5'
            model.save_weights(initial_weights)
        except Exception as e:
            print(f"Could not save initial weights: {e}")
            initial_weights = None
        
        # Loop through folds for walk-forward validation
        for fold in range(self.n_splits - 1):  # We need at least one fold for testing
            print(f"\n{'='*50}\nTraining fold {fold + 1}/{self.n_splits}\n{'='*50}")
            
            if fold == 0:
                train_end = min_train_size
            else:
                train_end = (fold + 1) * fold_size
            
            test_start = train_end
            test_end = min(n_samples, test_start + fold_size)
            
            if test_start >= test_end or test_start >= n_samples:
                print(f"Skipping fold {fold + 1} - no test data available")
                continue
            
            boundary = train_end - 1
            purge_start = max(0, boundary - self.purge_window + 1)
            purge_end = min(n_samples, test_start + self.purge_window)
            
            train_idx = np.arange(0, purge_start)
            test_idx = np.arange(purge_end, test_end)
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"Skipping fold {fold + 1} - empty train or test set after purging")
                continue
            
            # Start with a clean expanded tensor for this fold
            X_fold = X_expanded.copy()
            
            # Get the dates corresponding to our sequences
            train_dates = original_indices[train_idx]
            test_dates = original_indices[test_idx]
            
            # Get z-score returns for the training period
            z_scores_train = df.loc[train_dates[0]:train_dates[-1], z_score_col].values
            
            # Process each FFT component number
            for i, component_num in enumerate(fft_num):
                # Calculate the feature index for this FFT component in the expanded tensor
                fft_feature_idx = original_feature_count + i
                print(f"Adding FFT feature with {component_num} components at index {fft_feature_idx}")
                
                # Perform FFT on training data
                fft_result = fft.fft(z_scores_train)
                frequencies = fft.fftfreq(len(z_scores_train))
                
                # Sort frequencies by magnitude and keep only top 'component_num' components
                magnitude = np.abs(fft_result)
                idx = np.argsort(magnitude)[::-1][:component_num]
                
                # Create a filtered FFT result (zero out all other frequencies)
                filtered_fft = np.zeros_like(fft_result, dtype=complex)
                filtered_fft[idx] = fft_result[idx]
                
                # Compute inverse FFT for training data
                reconstruction_train = fft.ifft(filtered_fft).real
                
                # Now forecast for test data based on the training FFT pattern
                test_length = len(test_idx)
                
                # Create forecast by extending the pattern
                last_cycle = reconstruction_train[-component_num:]
                repetitions_needed = int(np.ceil(test_length / len(last_cycle)))
                forecast_pattern = np.tile(last_cycle, repetitions_needed)[:test_length]
                
                # Update the 3D tensor for all sequence steps
                # For each sequence in train set
                for j, train_i in enumerate(train_idx):
                    # Get the corresponding index in the reconstruction
                    recon_idx = min(j, len(reconstruction_train)-1)
                    # For each time step in the sequence
                    for t in range(X_fold.shape[1]):  # sequence_length dimension
                        X_fold[train_i, t, fft_feature_idx] = reconstruction_train[recon_idx]
                
                # For each sequence in test set
                for j, test_i in enumerate(test_idx):
                    # Get the corresponding index in the forecast
                    forecast_idx = min(j, len(forecast_pattern)-1)
                    # For each time step in the sequence
                    for t in range(X_fold.shape[1]):  # sequence_length dimension
                        X_fold[test_i, t, fft_feature_idx] = forecast_pattern[forecast_idx]
            
            # Add ARIMA model forecasting if enabled
            if arima_model:
                # Define ARIMA feature index (after all FFT features)
                arima_feature_idx = original_feature_count + total_fft_features
                print(f"Adding ARIMA feature at index {arima_feature_idx}")
                
                # Try to fit ARIMA model to training data
                try:
                    # Create and fit ARIMA model
                    arima = ARIMA(z_scores_train, order=arima_order)
                    arima_fit = arima.fit()
                    print(f"ARIMA model fit: {arima_order}")
                    
                    # Generate in-sample predictions for training data
                    arima_train_pred = arima_fit.fittedvalues
                    
                    # If there are any NaN values at the beginning due to differencing,
                    # fill them with the first non-NaN value
                    if np.isnan(arima_train_pred[0]):
                        first_valid = np.where(~np.isnan(arima_train_pred))[0][0]
                        arima_train_pred[:first_valid] = arima_train_pred[first_valid]
                    
                    # Generate forecasts for test data
                    test_length = len(test_idx)
                    arima_test_forecast = arima_fit.forecast(steps=test_length)
                    
                    # Add the ARIMA predictions to the tensor
                    # For each sequence in train set
                    for j, train_i in enumerate(train_idx):
                        # Get the corresponding index in the ARIMA predictions
                        pred_idx = min(j, len(arima_train_pred)-1)
                        # For each time step in the sequence
                        for t in range(X_fold.shape[1]):  # sequence_length dimension
                            X_fold[train_i, t, arima_feature_idx] = arima_train_pred[pred_idx]
                    
                    # For each sequence in test set
                    for j, test_i in enumerate(test_idx):
                        # Get the corresponding index in the forecast
                        forecast_idx = min(j, len(arima_test_forecast)-1)
                        # For each time step in the sequence
                        for t in range(X_fold.shape[1]):  # sequence_length dimension
                            X_fold[test_i, t, arima_feature_idx] = arima_test_forecast[forecast_idx]
                    
                except Exception as e:
                    print(f"Error fitting ARIMA model: {e}")
                    # In case of error, fill with zeros
                    for i in range(X_fold.shape[0]):
                        for t in range(X_fold.shape[1]):
                            X_fold[i, t, arima_feature_idx] = 0.0
            
            # Get the updated training and testing data for this fold
            X_train, y_train = X_fold[train_idx], y[train_idx]
            X_test, y_test = X_fold[test_idx], y[test_idx]

            # Replace NaN with 0
            X_train = np.nan_to_num(X_train, nan=0)
            X_test = np.nan_to_num(X_test, nan=0)

            # Replace Inf and -Inf with 0
            X_train = np.where(np.isinf(X_train), 0, X_train)
            X_test = np.where(np.isinf(X_test), 0, X_test)

            # Apply rolling normalization to FFT and ARIMA columns
            rolling_window = 252
            
            # Determine which columns need normalization
            norm_cols = list(range(original_feature_count, original_feature_count + total_fft_features + total_arima_features))
            
            # NumPy array implementation for 3D tensor
            for col in norm_cols:
                # For each time step in the sequence
                for t in range(X_fold.shape[1]):  # sequence_length dimension
                    # Extract values for this column at this time step for training data
                    train_values = X_fold[:train_end, t, col]
                    
                    # Calculate rolling means and stds using numpy's implementation
                    rolling_means = np.zeros_like(train_values)
                    rolling_stds = np.zeros_like(train_values)
                    
                    for i in range(len(train_values)):
                        start_idx = max(0, i - rolling_window + 1)
                        window_values = train_values[start_idx:i+1]
                        rolling_means[i] = np.mean(window_values)
                        rolling_stds[i] = np.std(window_values) if len(window_values) > 1 else 1.0
                    
                    # Use the last calculated mean and std for the test data
                    last_mean = rolling_means[-1] if len(rolling_means) > 0 else 0
                    last_std = rolling_stds[-1] if len(rolling_stds) > 0 and rolling_stds[-1] > 0 else 1.0
                    
                    # Normalize training data
                    for i in range(train_end):
                        X_fold[i, t, col] = (X_fold[i, t, col] - rolling_means[i]) / (rolling_stds[i] if rolling_stds[i] > 0 else 1.0)
                    
                    # Normalize test data using last train stats
                    for i in range(test_start, test_end):
                        X_fold[i, t, col] = (X_fold[i, t, col] - last_mean) / last_std

            # Apply scaling for model training
            # Get original dimensions
            n_samples_train, n_timesteps, n_features = X_train.shape
            n_samples_test = X_test.shape[0]
            
            # Reshape to 2D (combine samples and timesteps)
            X_train_reshaped = X_train.reshape(n_samples_train * n_timesteps, n_features)
            X_test_reshaped = X_test.reshape(n_samples_test * n_timesteps, n_features)
            
            # Apply scaling
            X_train_scaled_2d = self.scaler_nn.fit_transform(X_train_reshaped)
            X_test_scaled_2d = self.scaler_nn.transform(X_test_reshaped)
            
            # Reshape back to 3D
            X_train_scaled = X_train_scaled_2d.reshape(n_samples_train, n_timesteps, n_features)
            X_test_scaled = X_test_scaled_2d.reshape(n_samples_test, n_timesteps, n_features)
            
            # Print fold data dimensions and class distribution
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"Train class distribution: {np.bincount(y_train.astype(int))}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_test shape: {y_test.shape}")
            print(f"Test class distribution: {np.bincount(y_test.astype(int))}")
            
            fold_weights = None
            if sample_weights is not None:
                fold_weights = sample_weights[train_idx]

            # Reshape y_train to (n_samples, 1) if needed
            y_train_reshaped = y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train
            print(f"y_train shape after reshape: {y_train_reshaped.shape}")

            # Try to reset the model's weights to avoid fold-specific learning
            if initial_weights:
                try:
                    model.load_weights(initial_weights)
                    print("Successfully reset model weights")
                except Exception as e:
                    print(f"Could not reset weights: {e}")
                    
            # Train the model with sample weights and class weights
            history = model.fit(
                X_train_scaled, 
                y_train_reshaped.ravel(),
                sample_weight=fold_weights,
                validation_data=(X_test_scaled, y_test.ravel()),
                callbacks=[early_stopping, reduce_lr] + (callbacks or []),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1
            )   

            # Get raw predictions (probabilities)
            test_preds = model.predict(X_test_scaled, batch_size=batch_size, verbose=0)
            train_preds = model.predict(X_train_scaled, batch_size=batch_size, verbose=0)
            
            # Print prediction stats
            print(f"Raw test_preds shape: {test_preds.shape}")
            print(f"Raw test_preds mean: {np.mean(test_preds)}, min: {np.min(test_preds)}, max: {np.max(test_preds)}")
            print(f"Raw train_preds shape: {train_preds.shape}")
            print(f"Raw train_preds mean: {np.mean(train_preds)}, min: {np.min(train_preds)}, max: {np.max(train_preds)}")

            # Ensure predictions are the right shape
            if len(test_preds.shape) > 1 and test_preds.shape[1] > 1:
                test_preds = test_preds[:, 1] if test_preds.shape[1] >= 2 else test_preds[:, 0]
                train_preds = train_preds[:, 1] if train_preds.shape[1] >= 2 else train_preds[:, 0]

            # Reshape to 2D array for sklearn
            test_preds = test_preds.reshape(-1, 1)
            train_preds = train_preds.reshape(-1, 1)
            
            # Store predictions and indices for global calibration
            all_train_preds.append(train_preds)
            all_train_labels.append(y_train)
            fold_test_preds.append(test_preds)
            fold_test_indices.append(test_idx)
            
            # Store fold information
            for i, idx in enumerate(test_idx):
                if idx < len(results_data['fold']):
                    results_data['fold'][idx] = fold + 1
                    results_data['prediction_source'][idx] = 'test_pred'
            
            for i, idx in enumerate(train_idx):
                if idx < len(results_data['fold']):
                    results_data['fold'][idx] = fold + 1
                    results_data['prediction_source'][idx] = 'train_pred'
        
        # Apply global calibration after all folds
        if all_train_preds:
            print("\nApplying global calibration to all predictions...")
            # Combine all train predictions and labels
            all_train_preds_combined = np.vstack(all_train_preds)
            all_train_labels_combined = np.concatenate(all_train_labels)
            
            print(f"Combined train predictions shape: {all_train_preds_combined.shape}")
            print(f"Combined train labels shape: {all_train_labels_combined.shape}")
            
            # Train a single calibration model on all training data
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.impute import SimpleImputer
            
            # Try isotonic regression for calibration
            platt_model = CalibratedClassifierCV(
                estimator=LogisticRegression(C=0.3, solver='lbfgs'),
                method='isotonic',
                cv=5
            )
            imputer = SimpleImputer(strategy='mean')
            all_train_preds_combined_imputed = imputer.fit_transform(all_train_preds_combined)
            platt_model.fit(all_train_preds_combined_imputed, all_train_labels_combined)
            
            # Apply calibration to all folds
            for fold_idx, (test_preds, test_idx) in enumerate(zip(fold_test_preds, fold_test_indices)):
                # Get calibrated probabilities
                test_preds_calibrated = platt_model.predict_proba(test_preds)[:, 1]
                
                print(f"Fold {fold_idx+1} calibrated test predictions stats:")
                print(f"  Mean: {np.mean(test_preds_calibrated):.4f}")
                print(f"  Min: {np.min(test_preds_calibrated):.4f}")
                print(f"  Max: {np.max(test_preds_calibrated):.4f}")
                
                # Determine optimal threshold using validation data
                if fold_idx == 0:
                    precisions, recalls, thresholds = precision_recall_curve(
                        all_train_labels_combined, 
                        platt_model.predict_proba(all_train_preds_combined)[:, 1]
                    )
                    # Find threshold that maximizes F1 score
                    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
                    optimal_idx = np.argmax(f1_scores)
                    optimal_threshold = thresholds[optimal_idx]
                    print(f"Determined optimal threshold: {optimal_threshold:.4f} (F1: {f1_scores[optimal_idx]:.4f})")
                    # Only update threshold if it's significantly different and seems reasonable
                    if 0.3 <= optimal_threshold <= 0.7:
                        threshold = optimal_threshold
                    
                print(f"Using threshold: {threshold}")
                
                # Now convert calibrated probabilities to binary using the threshold
                test_preds_binary = (test_preds_calibrated > threshold).astype(int)
                
                # Store test predictions
                for i, idx in enumerate(test_idx):
                    if idx < len(results_data['prediction']):
                        results_data['prediction'][idx] = test_preds_binary[i]
                        results_data['prediction_score'][idx] = test_preds_calibrated[i]
            
            # Also apply calibration to training data
            for fold_idx, (train_preds, train_idx) in enumerate(zip(all_train_preds, all_train_labels)):
                train_preds_calibrated = platt_model.predict_proba(train_preds)[:, 1]
                train_preds_binary = (train_preds_calibrated > threshold).astype(int)
                
                # Store calibrated train predictions
                for i, idx in enumerate(train_idx):
                    if idx < len(results_data['prediction']):
                        results_data['prediction_score'][idx] = train_preds_calibrated[i]

        # Probability Distribution Analysis
        plt.figure(figsize=(8, 5))
        plt.hist(results_data['prediction_score'], bins=30, alpha=0.75, color='dodgerblue', edgecolor='white')
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold: {threshold}')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title("Distribution of Predicted Probabilities", fontsize=18, color='dodgerblue')
        plt.legend()
        plt.savefig("plots\\NeuralNetwork\\probability_distribution.png")

        # Create and return a DataFrame from the results_data
        results_df = pd.DataFrame(results_data, index=original_indices)
        print("\nTraining complete.")
        print(results_df.head())
        
        # Print summary statistics
        print("\nFinal Results Summary:")
        print(f"Overall prediction distribution: {np.bincount(results_data['prediction'].astype(int))}")
        print(f"True values distribution: {np.bincount(y.astype(int))}")
        
        # Calculate and print metrics
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(y, results_data['prediction']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, results_data['prediction']))
        
        print(f"\nPrediction scores summary:")
        print(f"Mean: {np.mean(results_data['prediction_score']):.4f}")
        print(f"Min: {np.min(results_data['prediction_score']):.4f}")
        print(f"Max: {np.max(results_data['prediction_score']):.4f}")
        
        # Add model feature information
        features_used = {
            'original_features': original_feature_count,
            'fft_features': total_fft_features,
            'arima_features': total_arima_features,
            'total_features': original_feature_count + total_fft_features + total_arima_features
        }
        
        print("\nFeatures used in model:")
        for k, v in features_used.items():
            print(f"  {k}: {v}")
        
        return results_df
