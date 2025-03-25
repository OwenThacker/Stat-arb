import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

import io
from datetime import datetime

class Backtest:
    def __init__(self, stock1, stock2, data, final_df, normalized_df, combined_final, returns):
        self.stock1 = stock1
        self.stock2 = stock2
        self.data = data
        self.combined_final = combined_final  # Using combined_df directly now
        self.final_df = final_df
        self.z_score = normalized_df['z_score']
        self.returns = returns
        self.cum_returns = None
        self.daily_returns = None
        self.strategy_returns = None
        self.position_sizes = None
        self.trades = None
        self.metrics = {}
        self.completed_trades = []
        
        # Parameters for stop loss and take profit
        self.base_stop_loss_scale = 0.15 # \\ Alternative Optimal 8.0 risk-reward, 5 day window, 3 day vol base stop - 0.15, max pos 20, vol factor 0, pos scaling 6
        self.risk_reward_ratio = 8.0
        
        # Position sizing parameters
        self.min_position_size = 0.1
        self.max_position_size = 2.5
        self.max_concurrent_positions = 20
        self.first_position_size = 0.50
        self.position_scaling_factor = 1.0 # \\ Used as a risk factor control for position sizing
        
        # Transaction costs
        self.transaction_cost_fixed = 0.0
        self.transaction_cost_percentage = 0.001 # (0.1% transaction cost per trade)

        # Volatillity
        self.z_volatility_window = 12
        self.volatility_factor = 0 # (1/10 starting reference value) # Used to dynamically adjust stop loss size.(squared)
        
        # Direction tracking
        self.direction_success = {1: 0, -1: 0}
        self.direction_positions = {1: 0, -1: 0}
        self.active_positions = []

        # Trailing stop loss parameters
        self.trailing_stop_loss = False # Enable/disable trailing stop loss
        self.trailing_threshold = 0.35  # 35% drawdown from peak profit triggers exit

    def run_backtest(self):
        print('📌 Running backtest with Kelly position sizing and simplified signal logic...')
        
        # Get z-score for mean reversion signals
        z_score = self.z_score
        z_score_returns = z_score.pct_change().fillna(0).to_frame(name='Returns')
        
        # Calculate z-score velocity (rate of change)
        z_score_velocity = z_score.diff(periods=3)  # 3-day rate of change
        
        # Create a dataframe for backtest results
        combined_final = self.combined_final.copy()
        backtest_df = pd.DataFrame(index=combined_final.index)
        
        # Use the signal from combined_df directly
        backtest_df['prediction'] = combined_final['final_label'].copy()

        # Add z-velocity to the dataframe
        backtest_df['z_velocity'] = z_score_velocity
        
        # Apply mean reversion filters:
        # 1. For long positions (signal=1): z-score must be below -1.25 (oversold)
        # 2. For short positions (signal=-1): z-score must be above 1.25 (overbought)
        
        # Initialize filtered prediction
        backtest_df['filtered_prediction'] = 0
        
        # Apply mean reversion logic for longs: signal=1 and z-score < -1.25
        long_condition = (backtest_df['prediction'] == 1) & (z_score < -1.25)
        backtest_df.loc[long_condition, 'filtered_prediction'] = 1
        
        # Apply mean reversion logic for shorts: signal=-1 and z-score > 1.25
        short_condition = (backtest_df['prediction'] == -1) & (z_score > 1.25)
        backtest_df.loc[short_condition, 'filtered_prediction'] = -1
        
        # Store trades for analysis
        self.trades = backtest_df['filtered_prediction'].copy()
        
        # Get stock returns for both stocks in the pair
        stock1_returns = self.returns[self.stock1]
        stock2_returns = self.returns[self.stock2]

        # Ensure datetime index
        stock1_returns.index = pd.to_datetime(stock1_returns.index)
        stock2_returns.index = pd.to_datetime(stock2_returns.index)
        
        # Create a DataFrame to track position sizes
        position_size_df = pd.DataFrame(index=backtest_df.index)
        position_size_df['trade_active'] = backtest_df['filtered_prediction'] != 0
        position_size_df['position_size'] = 0.0
        
        # Track current equity
        initial_equity = 1000.00
        current_equity = initial_equity
        max_equity = initial_equity
        equity_curve = pd.Series(index=backtest_df.index, data=initial_equity)
        
        # Initialize the strategy returns
        long_short_returns = pd.Series(index=backtest_df.index, data=0.0)
        
        # Calculate z-score volatility for dynamic position sizing
        z_volatility = pd.Series(index=backtest_df.index)
        volatility_window = self.z_volatility_window  # Window for volatility calculation
        
        # Calculate rolling volatility
        for i in range(len(backtest_df)):
            if i >= volatility_window:
                z_volatility.iloc[i] = z_score_returns.iloc[i-volatility_window:i]['Returns'].std()
            else:
                z_volatility.iloc[i] = z_score_returns.iloc[:max(1, i)]['Returns'].std()
        
        # Track entry/exit points for visualization
        entry_points = []
        exit_points = []
        stop_loss_levels = []
        take_profit_levels = []
        
        # Run through the backtest day by day
        for i in range(len(position_size_df)):
            current_date = position_size_df.index[i]
            daily_total_return = 0.0
            positions_to_remove = []
            
            # Get current z-score volatility
            current_z_volatility = z_volatility.iloc[i]
            
            # Calculate daily returns for existing positions and check for exits
            for idx, position in enumerate(self.active_positions):
                # Update days held
                position['days_held'] += 1
                
                # Calculate the current z-score change since entry
                zscore_change = z_score.loc[current_date] - position['entry_zscore']
                
                # Check standard exit conditions
                take_profit_hit = (position['direction'] == 1 and zscore_change >= position['take_profit_level']) or \
                                (position['direction'] == -1 and zscore_change <= -position['take_profit_level'])

                stop_loss_hit = (position['direction'] == 1 and zscore_change <= position['stop_loss_level']) or \
                            (position['direction'] == -1 and zscore_change >= -position['stop_loss_level'])
                
                # New trailing stop loss logic
                trailing_stop_exit = False
                if self.trailing_stop_loss:
                    # Track peak cumulative return for this position
                    position['peak_cumulative_return'] = max(
                        position.get('peak_cumulative_return', position['cumulative_return']), 
                        position['cumulative_return']
                    )
                    
                    # Check if position has dropped more than trailing threshold from peak
                    trailing_stop_exit = (
                        position['cumulative_return'] <= 
                        position['peak_cumulative_return'] * (1 - self.trailing_threshold)
                    )
                
                # Determine if we should exit the position
                exit_position = take_profit_hit or stop_loss_hit or trailing_stop_exit
                
                # Calculate daily return for this position
                if i > 0 and current_date in stock1_returns.index and current_date in stock2_returns.index:
                    # For a pairs trade:
                    # - Direction 1: Long stock1, Short stock2
                    # - Direction -1: Short stock1, Long stock2
                    daily_pos_return = ((stock1_returns.loc[current_date] * position['direction']) - 
                        (stock2_returns.loc[current_date] * position['direction'])) * position['position_size']
                    
                    # Update position performance
                    position['cumulative_return'] = position.get('cumulative_return', 0) + daily_pos_return
                    
                    # Add to daily total return
                    daily_total_return += daily_pos_return
                    
                    # Update equity
                    current_equity *= (1 + daily_pos_return)
                    max_equity = max(max_equity, current_equity)
                
                # If exiting, mark position for removal and record trade details
                if exit_position:
                    positions_to_remove.append(idx)
                    
                    # Update direction success tracking
                    if take_profit_hit:
                        self.direction_success[position['direction']] += 1
                    
                    # Decrement position count for this direction
                    self.direction_positions[position['direction']] = max(0, self.direction_positions[position['direction']] - 1)
                    
                    # Add exit point for visualization
                    exit_points.append({
                        'date': current_date,
                        'zscore': z_score.loc[current_date],
                        'direction': position['direction'],
                        'reason': "Take Profit" if take_profit_hit else "Trailing Stop" if trailing_stop_exit else "Stop Loss"
                    })
                    
                    # Record completed trade details
                    if position['entry_date'] is not None:
                        try:
                            entry_index = backtest_df.index.get_loc(position['entry_date'])
                            exit_index = i
                            
                            if entry_index < exit_index:
                                holding_dates = backtest_df.index[entry_index:exit_index+1]
                                valid_dates = [d for d in holding_dates if d in stock1_returns.index and d in stock2_returns.index]
                                
                                if valid_dates:
                                    # Calculate returns for both stocks
                                    stock1_daily_returns = stock1_returns.loc[valid_dates]
                                    stock2_daily_returns = stock2_returns.loc[valid_dates]
                                    
                                    # Calculate cumulative returns
                                    stock1_cum_return = (1 + stock1_daily_returns).prod() - 1
                                    stock2_cum_return = (1 + stock2_daily_returns).prod() - 1
                                    
                                    # Raw trade return
                                    raw_trade_return = (stock1_cum_return * position['direction']) - (stock2_cum_return * position['direction'])
                                    
                                    # Apply transaction costs
                                    transaction_cost = (self.transaction_cost_fixed + 
                                                    position['position_size'] * self.transaction_cost_percentage)
                                    adjusted_trade_return = raw_trade_return - transaction_cost
                                    
                                    # Calculate PnL
                                    pnl = raw_trade_return * position['position_size']
                                    
                                    # Store trade details
                                    trade_details = {
                                        'entry_date': position['entry_date'],
                                        'exit_date': current_date,
                                        'holding_period': position['days_held'],
                                        'direction': "Long " + self.stock1 + "/Short " + self.stock2 if position['direction'] == 1 else 
                                                "Short " + self.stock1 + "/Long " + self.stock2,
                                        'entry_zscore': position['entry_zscore'],
                                        'exit_zscore': z_score.loc[current_date],
                                        'zscore_change': z_score.loc[current_date] - position['entry_zscore'],
                                        'position_size': position['position_size'],
                                        
                                        # New fields for stock cumulative returns
                                        'stock1_cum_return': stock1_cum_return,
                                        'stock2_cum_return': stock2_cum_return,
                                        
                                        'return': raw_trade_return,
                                        'pnl': pnl,
                                        'adjusted_return': adjusted_trade_return,
                                        'exit_reason': "Take Profit" if take_profit_hit else "Trailing Stop" if trailing_stop_exit else "Stop Loss",
                                        'stop_loss_level': position['stop_loss_level'],
                                        'take_profit_level': position['take_profit_level'],
                                        'n_days_traded': len(valid_dates) - 1,  # Excluding entry day
                                    }
                                    trade_details['peak_return'] = position.get('peak_cumulative_return', 0)

                                    self.completed_trades.append(trade_details)
                        except Exception as e:
                            print(f"Error processing trade exit: {e}")
            
            # Remove closed positions (in reverse order to avoid index issues)
            for idx in sorted(positions_to_remove, reverse=True):
                self.active_positions.pop(idx)
                
            # If positions were removed, recalculate position sizes with Kelly
            if positions_to_remove and self.active_positions:
                self.kelly_position_sizing(current_date)
            
            # Check if we can enter a new position
            can_add_position = len(self.active_positions) < self.max_concurrent_positions
            
            # Enter new position based on the filtered prediction signal
            if can_add_position and backtest_df.loc[current_date, 'filtered_prediction'] != 0:
                current_direction = backtest_df.loc[current_date, 'filtered_prediction']
                
                # Increment position count for this direction
                self.direction_positions[current_direction] = self.direction_positions.get(current_direction, 0) + 1
                direction_count = self.direction_positions[current_direction]
                
                # Get current prediction score for position sizing
                try:
                    current_probability = self.combined_final.loc[current_date, 'prediction_score']
                    # Ensure probability is between 0 and 1
                    if isinstance(current_probability, (int, float)):
                        current_probability = max(0.05, min(current_probability, 0.95))  # Avoid extreme values
                    else:
                        current_probability = 0.6
                except (KeyError, AttributeError):
                    current_probability = 0.6
                
                # Calculate dynamic stop loss INDEPENDENTLY of position sizing
                volatility_factor = max(current_z_volatility * (self.volatility_factor), 0.01) 
                
                # Stop loss calculation is now completely separate from position sizing
                dynamic_stop_loss = -self.base_stop_loss_scale * (1 + volatility_factor)
                
                # Take profit is risk-reward multiple of stop loss
                dynamic_take_profit = abs(dynamic_stop_loss) * self.risk_reward_ratio
                
                # Create new position with placeholder size to be determined by Kelly
                new_position = {
                    'direction': current_direction,
                    'entry_date': current_date,
                    'entry_zscore': z_score.loc[current_date],
                    'position_size': 0.0,  # Will be set by Kelly criterion
                    'days_held': 0,
                    'stop_loss_level': dynamic_stop_loss,
                    'take_profit_level': dynamic_take_profit,
                    'cumulative_return': 0.0,
                    'position_count': direction_count,
                    'win_prob': current_probability
                }
                
                # Add to active positions
                self.active_positions.append(new_position)
                
                # Apply Kelly position sizing to all positions
                self.kelly_position_sizing(current_date)
                
                # Get the updated position size after Kelly calculation
                position_size = new_position['position_size']
                posiion_size = position_size / dynamic_stop_loss # this makes risk consistent 
                
                # print(f"New position: Date={current_date.date()}, Dir={current_direction}, Size={position_size:.4f}, " 
                 #     f"Stop={dynamic_stop_loss:.4f}, TP={dynamic_take_profit:.4f}, Prob={current_probability:.4f}")
                
                # Add entry point for visualization
                entry_points.append({
                    'date': current_date,
                    'zscore': z_score.loc[current_date],
                    'direction': new_position['direction'],
                    'position_size': position_size
                })
                
                # Add stop loss and take profit levels for visualization
                stop_loss_levels.append({
                    'date': current_date,
                    'entry_zscore': z_score.loc[current_date],
                    'level': dynamic_stop_loss,
                    'direction': new_position['direction']
                })
                
                take_profit_levels.append({
                    'date': current_date,
                    'entry_zscore': z_score.loc[current_date],
                    'level': dynamic_take_profit,
                    'direction': new_position['direction']
                })
            
            # Record daily total return
            long_short_returns.loc[current_date] = daily_total_return
            
            # Calculate total position size (sum of all active positions)
            total_position_size = sum(pos['position_size'] for pos in self.active_positions)
            
            # Store total active position size (risk exposure)
            position_size_df.loc[current_date, 'position_size'] = total_position_size

            # Update equity curve
            equity_curve[current_date] = current_equity
        
        # Store the calculated position sizes (risk exposure over time)
        self.position_sizes = position_size_df['position_size']
        
        # Store strategy returns
        self.strategy_returns = long_short_returns
        
        # Calculate daily returns
        self.daily_returns = self.strategy_returns.copy()
        
        # Calculate cumulative returns
        self.cum_returns = (1 + self.daily_returns).cumprod() - 1
        
        # Calculate performance metrics
        self.calculate_metrics()
        
        # Store trade visualization data
        self.entry_points = entry_points
        self.exit_points = exit_points
        self.stop_loss_levels = stop_loss_levels
        self.take_profit_levels = take_profit_levels
        self.z_volatility = z_volatility
        self.equity_curve = equity_curve
        
        # Print trade history
        self.print_trade_history()

        self.create_tearsheet(self)
        
        # Return performance metrics
        return self.metrics
        
    def kelly_position_sizing(self, current_date):
        """
        Calculate position sizes using the Adapted Kelly Criterion for investment environments.
        """
        if not self.active_positions:
            return

        try:
            if current_date in self.final_df.index:
                current_probability = self.final_df.loc[current_date, 'prediction_score']
                current_probability = max(0.05, min(current_probability, 0.95))  # Keep within 5% - 95%
            else:
                current_probability = 0.6
        except (KeyError, AttributeError):
            current_probability = 0.6

        positions_by_direction = {1: [], -1: []}
        for pos in self.active_positions:
            positions_by_direction[pos['direction']].append(pos)

        for direction, positions in positions_by_direction.items():
            if not positions:
                continue

            sorted_positions = sorted(positions, key=lambda x: x.get('entry_date', pd.Timestamp.now()))

            for i, pos in enumerate(sorted_positions):
                pos['position_count'] = i + 1
                p = pos.get('win_prob', current_probability)
                q = 1 - p

                loss_pct = 1.0  
                win_pct = self.risk_reward_ratio  

                # Kelly Formula: f* = (p / loss_pct) - (q / win_pct)
                kelly_fraction = (p / loss_pct) - (q / win_pct)
                half_kelly = max(0, kelly_fraction * 0.5)  # Ensure non-negative Kelly fraction

                z_score_volatility = self.z_score.rolling(window=12).std()
                curr_z_volatility = z_score_volatility.loc[current_date]
                curr_z_score = self.z_score.loc[current_date]

                # Ensure volatility is nonzero to avoid division errors
                if curr_z_volatility == 0 or np.isnan(curr_z_volatility):
                    curr_z_volatility = 1e-6  # Small value to prevent division by zero

                # Calculate final position size
                final_position_size = half_kelly * (1 / np.sqrt(curr_z_volatility)) * self.position_scaling_factor
                
                # Ensure position size is between 0.1 and 1
                pos['position_size'] = min(1, max(0.1, final_position_size))

        self.direction_positions[direction] = len(positions)

       
    def calculate_rolling_metrics(self):
        """Calculate rolling performance metrics."""
        # Calculating rolling returns
        rolling_returns = pd.DataFrame(index=self.daily_returns.index)
        rolling_returns['1D'] = self.daily_returns
        rolling_returns['5D'] = self.daily_returns.rolling(window=5).sum()
        rolling_returns['10D'] = self.daily_returns.rolling(window=10).sum()
        rolling_returns['20D'] = self.daily_returns.rolling(window=20).sum()
        rolling_returns['60D'] = self.daily_returns.rolling(window=60).sum()
        
        # Calculating rolling volatility
        rolling_volatility = pd.DataFrame(index=self.daily_returns.index)
        rolling_volatility['10D'] = self.daily_returns.rolling(window=10).std() * np.sqrt(252)
        rolling_volatility['20D'] = self.daily_returns.rolling(window=20).std() * np.sqrt(252)
        rolling_volatility['60D'] = self.daily_returns.rolling(window=60).std() * np.sqrt(252)
        
        # Calculating rolling Sharpe ratio (assuming 0% risk-free rate)
        rolling_sharpe = pd.DataFrame(index=self.daily_returns.index)
        rolling_sharpe['10D'] = (self.daily_returns.rolling(window=10).mean() * 252) / (self.daily_returns.rolling(window=10).std() * np.sqrt(252))
        rolling_sharpe['20D'] = (self.daily_returns.rolling(window=20).mean() * 252) / (self.daily_returns.rolling(window=20).std() * np.sqrt(252))
        rolling_sharpe['60D'] = (self.daily_returns.rolling(window=60).mean() * 252) / (self.daily_returns.rolling(window=60).std() * np.sqrt(252))
        
        # Calculating drawdowns
        rolling_drawdown = pd.DataFrame(index=self.daily_returns.index)
        cum_returns = (1 + self.daily_returns).cumprod()
        rolling_max = cum_returns.rolling(window=252, min_periods=1).max()
        rolling_drawdown['252D'] = (cum_returns / rolling_max) - 1
        
        # Store the calculated metrics
        self.rolling_metrics = {
            'returns': rolling_returns,
            'volatility': rolling_volatility,
            'sharpe': rolling_sharpe,
            'drawdown': rolling_drawdown
        }
    
    def calculate_metrics(self):
        """Calculate and store performance metrics."""
        # Calculate total return
        if len(self.cum_returns) > 0:
            total_return = self.cum_returns.iloc[-1]
        else:
            total_return = 0.0
        
        # Calculate annualized return
        if len(self.daily_returns) > 0:
            days = len(self.daily_returns)
            annualized_return = ((1 + total_return) ** (252 / days)) - 1
        else:
            annualized_return = 0.0
        
        # Calculate annualized volatility
        if len(self.daily_returns) > 0:
            annualized_volatility = self.daily_returns.std() * np.sqrt(252)
        else:
            annualized_volatility = 0.0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if annualized_volatility > 0:
            sharpe_ratio = annualized_return / annualized_volatility
        else:
            sharpe_ratio = 0.0
        
        # Calculate maximum drawdown
        if len(self.daily_returns) > 0:
            cum_returns = (1 + self.daily_returns).cumprod()
            max_returns = cum_returns.cummax()
            drawdowns = (cum_returns / max_returns) - 1
            max_drawdown = drawdowns.min()
        else:
            max_drawdown = 0.0
        
        # Calculate win rate from completed trades
        if self.completed_trades:
            profitable_trades = [trade for trade in self.completed_trades if trade['pnl'] > 0]
            win_rate = len(profitable_trades) / len(self.completed_trades)
            
            # Calculate average profit and loss
            if profitable_trades:
                avg_profit = sum(trade['pnl'] for trade in profitable_trades) / len(profitable_trades)
            else:
                avg_profit = 0.0
                
            losing_trades = [trade for trade in self.completed_trades if trade['pnl'] <= 0]
            if losing_trades:
                avg_loss = sum(trade['pnl'] for trade in losing_trades) / len(losing_trades)
            else:
                avg_loss = 0.0
                
            # Calculate profit factor
            total_profit = sum(trade['pnl'] for trade in profitable_trades)
            total_loss = sum(abs(trade['pnl']) for trade in losing_trades)
            if total_loss > 0:
                profit_factor = total_profit / total_loss
            else:
                profit_factor = float('inf') if total_profit > 0 else 0.0
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        # Store the metrics
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.completed_trades)
        }
        
        # Print the metrics
        print('📊 Performance Metrics:')
        print(f'Total Return: {total_return:.2%}')
        print(f'Annualized Return: {annualized_return:.2%}')
        print(f'Annualized Volatility: {annualized_volatility:.2%}')
        print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
        print(f'Maximum Drawdown: {max_drawdown:.2%}')
        print(f'Win Rate: {win_rate:.2%}')
        print(f'Average Profit: {avg_profit:.2%}')
        print(f'Average Loss: {avg_loss:.2%}')
        print(f'Profit Factor: {profit_factor:.2f}')
        print(f'Total Trades: {len(self.completed_trades)}')
    
    def print_trade_history(self):
        """Print a history of all completed trades."""
        if not self.completed_trades:
            print('No completed trades.')
            return
            
        print('\n📜 Trade History:')
        print(f"{'Entry Date':<12} {'Exit Date':<12} {'Direction':<25} {'P/L':<16} {'Return':<8} {'Exit Reason':<12}")
        print('-' * 80)
        
        for trade in self.completed_trades:
            entry_date = trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date'])
            exit_date = trade['exit_date'].strftime('%Y-%m-%d') if hasattr(trade['exit_date'], 'strftime') else str(trade['exit_date'])
            
            print(f"{entry_date:<12} {exit_date:<12} {trade['direction']:<25} {trade['pnl']:>+7.2%} {trade['return']/100:>+7.2%} {trade['exit_reason']:<12}")
            
    def create_tearsheet(self, backtest_instance):
        """Create a tearsheet."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
        import numpy as np
        from matplotlib.ticker import FuncFormatter
        import pandas as pd
        import matplotlib.font_manager as fm
        
        # Try to set Aptos font - fallback gracefully if not available
        try:
            fm.findfont('Aptos')
            plt.rcParams['font.family'] = 'Aptos'
        except:
            # If Aptos is not available, try some common sans-serif fonts
            for font in ['Arial', 'Helvetica', 'DejaVu Sans']:
                try:
                    fm.findfont(font)
                    plt.rcParams['font.family'] = font
                    break
                except:
                    continue
        
        # Set up the figure
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 22), dpi=100)  # Increased height for new plots
        # Set the background color of the entire figure to light blue
        gs = GridSpec(5, 2, figure=fig, height_ratios=[0.5, 1, 1, 1, 1])  # Modified grid layout
        
        # Define an updated color palette
        royal_blue = '#0047AB'       # Royal blue for titles
        pale_blue = '#B3C7E6'        # Pale blue for fills and accents
        light_blue_gray = '#C7D3E3'  # Light blue-gray for z-score line
        bright_green = '#2ECC71'     # Brighter green for profitable trades
        bright_red = '#E74C3C'       # Brighter red for unprofitable trades
        orange = '#FF8C00'           # Orange for Sharpe ratio
        pale_purple = '#D1C2E0'      # Pale purple for position sizing
        background_color = '#F5F8FC' # Light blue background
        
        # Set figure background color
        fig.patch.set_facecolor(background_color)
        
        # Create title
        fig.suptitle(f"Pair Trading Analysis: {self.stock1} - {self.stock2}", 
                    fontsize=20, color=royal_blue, fontweight='bold', y=0.98)
        
        # 1. METRICS TABLE
        ax_table = fig.add_subplot(gs[0, :])
        ax_table.axis('off')
        ax_table.set_facecolor(light_blue_gray)
        
        # Create metrics list
        metrics = [
            ('Total Return', f"{backtest_instance.metrics['total_return']:.2%}"),
            ('Annualized Return', f"{backtest_instance.metrics['annualized_return']:.2%}"),
            ('Ann. Volatility', f"{backtest_instance.metrics['annualized_volatility']:.2%}"),
            ('Sharpe Ratio', f"{backtest_instance.metrics['sharpe_ratio']:.2f}"),
            ('Max Drawdown', f"{backtest_instance.metrics['max_drawdown']:.2%}"),
            ('Win Rate', f"{backtest_instance.metrics['win_rate']:.2%}"),
            ('Profit Factor', f"{backtest_instance.metrics['profit_factor']:.2f}"),
            ('Total Trades', f"{backtest_instance.metrics['total_trades']}")
        ]
        
        # Use a table approach for better layout control
        table_data = [[metric[0] for metric in metrics[:4]],
                    [metric[1] for metric in metrics[:4]],
                    [metric[0] for metric in metrics[4:]],
                    [metric[1] for metric in metrics[4:]]]
        
        # Create table with proper styling
        table = ax_table.table(cellText=table_data,
                            loc='center',
                            cellLoc='center',
                            bbox=[0.05, 0.2, 0.9, 0.6])
        
        # Style the table
        table.auto_set_font_size(False)
        
        # Style header cells
        for i in range(4):
            table[(0, i)].set_text_props(weight='bold', color=royal_blue, fontsize=12)
            table[(0, i)].set_facecolor('#EBF5FF')
            table[(2, i)].set_text_props(weight='bold', color=royal_blue, fontsize=12)
            table[(2, i)].set_facecolor('#EBF5FF')
            
            # Style data cells
            table[(1, i)].set_text_props(fontsize=11)
            table[(1, i)].set_facecolor('white')
            table[(3, i)].set_text_props(fontsize=11)
            table[(3, i)].set_facecolor('white')
        
        # Set table borders to black
        for cell in table._cells:
            table._cells[cell].set_edgecolor('black')
            table._cells[cell].set_linewidth(0.5)
        
        # Add table title
        ax_table.text(0.5, 0.85, 'Performance Metrics', fontsize=14, fontweight='bold', 
                    ha='center', color=royal_blue)
        
        # 2. EQUITY CURVE
        ax2 = fig.add_subplot(gs[1, :])
        ax2.set_facecolor(background_color)
        
        # Convert percentage returns to dollar amounts
        starting_equity = 1000
        equity_values = starting_equity * (1 + backtest_instance.cum_returns)
        
        # Plot equity curve
        ax2.plot(backtest_instance.cum_returns.index, equity_values, color=royal_blue, linewidth=2)
        ax2.fill_between(backtest_instance.cum_returns.index, starting_equity, equity_values, 
                        color=pale_blue, alpha=0.5)
        
        # Add horizontal line for initial equity
        ax2.axhline(y=starting_equity, color='gray', linestyle='--', alpha=0.3)
        
        # Format y-axis without $ unit
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Set title and labels - black non-bold labels as requested
        ax2.set_title('Equity Curve', color=royal_blue, fontsize=14, fontweight='bold')
        ax2.set_ylabel('Equity', color='black', fontsize=12)
        ax2.grid(False)
        
        # 3. Z-SCORE AND TRADES
        ax1 = fig.add_subplot(gs[2, :])
        ax1.set_facecolor(background_color)
        
        # Plot z-score with light blue-gray color
        ax1.plot(backtest_instance.z_score.index, backtest_instance.z_score, color=light_blue_gray, alpha=1.0, linewidth=1.2)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax1.axhline(y=1.25, color=royal_blue, linestyle='--', alpha=0.5)
        ax1.axhline(y=-1.25, color=royal_blue, linestyle='--', alpha=0.5)
        
        # Plot entry points with white border
        for entry in backtest_instance.entry_points:
            marker = '^' if entry['direction'] == 1 else 'v'
            color = bright_green if entry['direction'] == 1 else bright_red
            ax1.scatter(entry['date'], entry['zscore'], marker=marker, s=25, color=color, 
                       edgecolor='white', linewidth=0.8, alpha=0.8)
        
        # Plot exit points with white border
        for exit in backtest_instance.exit_points:
            if exit['reason'] == 'Take Profit':
                ax1.scatter(exit['date'], exit['zscore'], marker='o', s=25, color=bright_green, 
                        edgecolor='white', linewidth=0.8, alpha=0.8)
            else:  # Stop Loss
                ax1.scatter(exit['date'], exit['zscore'], marker='x', s=25, color=bright_red, 
                            edgecolor='white', linewidth=0.8, alpha=0.8)
        
        # Set title and labels - black non-bold labels as requested
        ax1.set_title('Z-Score and Trades', color=royal_blue, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Z-Score', color='black', fontsize=12)
        ax1.grid(False)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. SHARPE RATIO OVER TIME and WIN RATE EVOLUTION (Side by side)
        ax_sharpe = fig.add_subplot(gs[3, 0])
        ax_sharpe.set_facecolor(background_color)
        
        # Calculate rolling Sharpe ratio (assuming we need to calculate it)
        if not hasattr(backtest_instance, 'rolling_sharpe'):
            # Calculate rolling returns
            returns = backtest_instance.cum_returns.pct_change().dropna()
            # Calculate 30-day rolling Sharpe ratio (annualized)
            rolling_window = min(30, len(returns) - 1)  # Use shorter window if data is limited
            if rolling_window > 0:
                rolling_returns = returns.rolling(window=rolling_window)
                rolling_mean = rolling_returns.mean() * 252  # Annualize
                rolling_std = rolling_returns.std() * np.sqrt(252)  # Annualize
                backtest_instance.rolling_sharpe = rolling_mean / rolling_std
                backtest_instance.rolling_sharpe = backtest_instance.rolling_sharpe.fillna(0)
            else:
                # Create empty series if not enough data
                backtest_instance.rolling_sharpe = pd.Series(0, index=returns.index)
        
        # Plot rolling Sharpe ratio
        ax_sharpe.plot(backtest_instance.rolling_sharpe.index, backtest_instance.rolling_sharpe, 
                    color=orange, linewidth=2)
        
        # Add horizontal line at 1.0 (common threshold for good Sharpe ratio)
        ax_sharpe.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Set title and labels
        ax_sharpe.set_title('Rolling Sharpe Ratio (30-day)', color=royal_blue, fontsize=14, fontweight='bold')
        ax_sharpe.set_ylabel('Sharpe Ratio', color='black', fontsize=12)
        ax_sharpe.grid(False)
        
        # Format x-axis
        ax_sharpe.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_sharpe.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_sharpe.xaxis.get_majorticklabels(), rotation=45)
        
        # WIN RATE EVOLUTION as subplot next to SHARPE RATIO
        ax_winrate = fig.add_subplot(gs[3, 1])
        ax_winrate.set_facecolor(background_color)
        
        # Calculate cumulative win rate evolution
        if self.completed_trades and len(self.completed_trades) > 0:
            trade_dates = [trade['exit_date'] for trade in self.completed_trades]
            trade_results = [1 if trade['pnl'] > 0 else 0 for trade in self.completed_trades]
            
            # Create DataFrame with trade results
            win_rate_df = pd.DataFrame({'result': trade_results}, index=trade_dates)
            win_rate_df = win_rate_df.sort_index()
            
            # Calculate cumulative win rate
            win_rate_df['cumulative_wins'] = win_rate_df['result'].cumsum()
            win_rate_df['cumulative_trades'] = pd.Series(range(1, len(win_rate_df) + 1), index=win_rate_df.index)
            win_rate_df['win_rate'] = win_rate_df['cumulative_wins'] / win_rate_df['cumulative_trades']
            
            # Plot win rate evolution with the same pale_blue color as equity curve
            ax_winrate.plot(win_rate_df.index, win_rate_df['win_rate'], color=royal_blue, linewidth=2)
            ax_winrate.fill_between(win_rate_df.index, 0, win_rate_df['win_rate'], color=pale_blue, alpha=0.5)
            
            # Add horizontal line at 0.5 (break-even win rate)
            ax_winrate.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Set title and labels
            ax_winrate.set_title('Win Rate Evolution', color=royal_blue, fontsize=14, fontweight='bold')
            ax_winrate.set_ylabel('Win Rate', color='black', fontsize=12)
            ax_winrate.set_ylim(0, 1)  # Limit y-axis from 0 to 1
            ax_winrate.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0%}'))  # Format as percentage
            ax_winrate.grid(False)
            
            # Format x-axis
            ax_winrate.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_winrate.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax_winrate.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax_winrate.text(0.5, 0.5, 'No completed trades', ha='center', va='center', color=royal_blue)
            ax_winrate.set_title('Win Rate Evolution', color=royal_blue, fontsize=14, fontweight='bold')
        
        # 5. TRADE RETURNS DISTRIBUTION and POSITION SIZES side by side
        ax3 = fig.add_subplot(gs[4, 0])
        ax3.set_facecolor(background_color)
        
        if self.completed_trades and len(self.completed_trades) > 0:
            # Extract trade returns
            trade_returns = [trade['pnl'] for trade in self.completed_trades]
            
            # Split into profitable and unprofitable trades
            profitable = [r for r in trade_returns if r > 0]
            unprofitable = [r for r in trade_returns if r <= 0]
            
            # Plot histogram with brighter colors
            bins = np.linspace(min(trade_returns), max(trade_returns), 15)
            ax3.hist(profitable, bins=bins, alpha=0.7, color=bright_green, label='Profitable', edgecolor='black', linewidth=0.5)
            ax3.hist(unprofitable, bins=bins, alpha=0.7, color=bright_red, label='Unprofitable', edgecolor='black', linewidth=0.5)
            
            # Add vertical line at zero
            ax3.axvline(0, color='black', linestyle='--', alpha=0.3)
            
            # Set title and labels - black non-bold labels as requested
            ax3.set_title('Distribution of Trade Returns', color=royal_blue, fontsize=14, fontweight='bold')
            ax3.set_xlabel('Return', color='black', fontsize=12)
            ax3.set_ylabel('Frequency', color='black', fontsize=12)
            ax3.grid(False)
            ax3.legend(frameon=True, facecolor='white', edgecolor=royal_blue, framealpha=0.7, loc='best')
        else:
            ax3.text(0.5, 0.5, 'No completed trades', ha='center', va='center', color=royal_blue)
            ax3.set_title('Distribution of Trade Returns', color=royal_blue, fontsize=14, fontweight='bold')
        
        # POSITION SIZES with increased alpha for better visibility
        ax4 = fig.add_subplot(gs[4, 1])
        ax4.set_facecolor(background_color)
        
        # Plot position sizes with pale purple and increased alpha
        ax4.plot(backtest_instance.position_sizes.index, backtest_instance.position_sizes, 
                color=pale_purple, linewidth=2)
        
        # Add shaded area with increased alpha
        ax4.fill_between(backtest_instance.position_sizes.index, 0, 
                        backtest_instance.position_sizes, color=pale_purple, alpha=0.8)
        
        # Set title and labels - black non-bold labels as requested
        ax4.set_title('Position Sizes', color=royal_blue, fontsize=14, fontweight='bold')
        ax4.set_ylabel('Position Size', color='black', fontsize=12)
        ax4.grid(False)
        
        # Format x-axis
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend for trade markers - moved to bottom of the figure
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=bright_green, markersize=10, 
                  markeredgecolor='white', label='Long Entry'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=bright_red, markersize=10, 
                  markeredgecolor='white', label='Short Entry'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=bright_green, markersize=10, 
                  markeredgecolor='white', label='Take Profit Exit'),
            Line2D([0], [0], marker='x', color='white', markersize=10, 
                  markeredgewidth=2, markeredgecolor=bright_red, label='Stop Loss Exit')
        ]
        
        # Add legend at the bottom of the figure
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                frameon=True, facecolor='white', edgecolor=royal_blue, framealpha=0.7,
                ncol=4)  # All items in one row
        
        # Add a subtle watermark
        fig.text(0.5, 0.5, f"{self.stock1}-{self.stock2}", fontsize=60, 
                color='#EBF5FF', ha='center', va='center', alpha=0.05, rotation=30)
        
        # Adjust layout with space for the legend at the bottom
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the tearsheet
        filename = f"Results\\{self.stock1}_{self.stock2}_tearsheet.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=background_color)
        
        plt.close()
        return filename
    
    def get_performance_data(self):
        """
        Export the performance data needed for portfolio-level analysis.
        
        Returns:
            dict: A dictionary containing performance metrics and time series data
        """
        return {
            'pair_name': f"{self.stock1}_{self.stock2}",
            'metrics': self.metrics,
            'daily_returns': self.daily_returns,
            'cum_returns': self.cum_returns,
            'position_sizes': self.position_sizes,
            'equity_curve': self.equity_curve,
            'trades': self.completed_trades,
            'entry_points': self.entry_points,
            'exit_points': self.exit_points
        }
    


class PortfolioBacktest:
    def __init__(self):
        self.pair_results = {}
        self.portfolio_daily_returns = None
        self.portfolio_cum_returns = None
        self.portfolio_equity_curve = None
        self.portfolio_metrics = {}
        self.all_trades = []
        
    def add_pair_result(self, pair_name, backtest_obj):
        """
        Add a pair's backtest results to the portfolio
        """
        # Get performance data from the backtest
        perf_data = backtest_obj.get_performance_data()
        self.pair_results[pair_name] = perf_data
        
        # Add trades to the overall list
        for trade in perf_data['trades']:
            trade['pair'] = pair_name  # Add pair identifier to each trade
            self.all_trades.append(trade)
            
    def calculate_portfolio_performance(self, equal_weight=True):
        """
        Calculate portfolio-level performance by combining all pairs. (Need to add portfolio optimisation)
        """
        if not self.pair_results:
            print("No pair results to analyze")
            return
            
        # Get all unique dates across all pairs
        all_dates = set()
        for pair_data in self.pair_results.values():
            if 'daily_returns' in pair_data and not pair_data['daily_returns'].empty:
                all_dates.update(pair_data['daily_returns'].index)
        
        all_dates = sorted(all_dates)
        
        # Create a dataframe with all dates
        portfolio_df = pd.DataFrame(index=all_dates)
        
        # Add returns for each pair
        for pair_name, pair_data in self.pair_results.items():
            if 'daily_returns' in pair_data and not pair_data['daily_returns'].empty:
                portfolio_df[pair_name] = pd.Series(pair_data['daily_returns']).reindex(all_dates).fillna(0)
            else:
                # Handle pairs with no trades by setting returns to 0
                portfolio_df[pair_name] = 0
            
        # Calculate portfolio returns (equal weighted for now)
        if equal_weight:
            # Count valid pairs for weighting
            valid_pairs = [pair for pair in self.pair_results.keys()]
            if not valid_pairs:
                print("Warning: No valid pairs with returns data")
                return portfolio_df
                
            weights = {pair: 1.0/len(valid_pairs) for pair in self.pair_results.keys()}
        else:
            # TODO: Implement custom weighting logic if needed
            weights = {pair: 1.0/len(self.pair_results) for pair in self.pair_results.keys()}
            
        # Apply weights to get portfolio returns
        for pair in self.pair_results.keys():
            #if 'BSPO' in pair.columns:
            #    portfolio_df[f"{pair}_weighted"] = portfolio_df[pair] * weights[pair] * 1/4 # (BSPO attatched to 4 pairs, so scale weighting)

            portfolio_df[f"{pair}_weighted"] = portfolio_df[pair] * weights[pair]
            
        # Calculate overall portfolio return
        portfolio_df['portfolio_return'] = portfolio_df[[f"{pair}_weighted" for pair in self.pair_results.keys()]].sum(axis=1)
        
        # Calculate cumulative returns
        portfolio_df['portfolio_cum_return'] = (1 + portfolio_df['portfolio_return']).cumprod() - 1
        
        # Calculate portfolio equity curve (starting with 1000)
        portfolio_df['portfolio_equity'] = 1000 * (1 + portfolio_df['portfolio_cum_return'])
        
        # Store results
        self.portfolio_daily_returns = portfolio_df['portfolio_return']
        self.portfolio_cum_returns = portfolio_df['portfolio_cum_return']
        self.portfolio_equity_curve = portfolio_df['portfolio_equity']
        
        # Calculate portfolio metrics
        self.calculate_portfolio_metrics()
        
        return portfolio_df
    
    def calculate_portfolio_metrics(self):
        """Calculate performance metrics for the overall portfolio"""
        if self.portfolio_daily_returns is None:
            print("No portfolio returns calculated yet")
            return
        
        returns = self.portfolio_daily_returns
        
        # Annual risk-free rate (e.g., 2%)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        
        # Basic metrics
        self.portfolio_metrics['total_return'] = self.portfolio_cum_returns.iloc[-1]
        self.portfolio_metrics['annualized_return'] = self.portfolio_cum_returns.iloc[-1] / (len(returns) / 252)
        self.portfolio_metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio - calculate directly from returns
        excess_returns = returns - risk_free_rate
        self.portfolio_metrics['sharpe_ratio'] = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252))

        # Drawdown analysis 
        # Create a wealth index from the daily returns
        wealth_index = (1 + self.portfolio_daily_returns).cumprod()
        
        # Calculate the running maximum
        running_max = wealth_index.cummax()
        
        # Calculate the drawdown
        drawdown = (wealth_index / running_max) - 1
        
        # This ensures drawdowns are properly calculated and always negative
        self.portfolio_metrics['max_drawdown'] = drawdown.min()
        
        # Calmar ratio: Annualized return / Max drawdown
        # Avoid division by zero in case max_drawdown is 0
        if abs(self.portfolio_metrics['max_drawdown']) > 0:
            self.portfolio_metrics['calmar_ratio'] = self.portfolio_metrics['annualized_return'] / abs(self.portfolio_metrics['max_drawdown'])
        else:
            self.portfolio_metrics['calmar_ratio'] = np.nan
        
        # Win rate for all trades
        if self.all_trades:
            profitable_trades = sum(1 for trade in self.all_trades if trade['pnl'] > 0)
            self.portfolio_metrics['win_rate'] = profitable_trades / len(self.all_trades)
            self.portfolio_metrics['total_trades'] = len(self.all_trades)
            
            # Average trade metrics
            self.portfolio_metrics['avg_trade_return'] = np.mean([trade['return'] for trade in self.all_trades])
            self.portfolio_metrics['avg_trade_pnl'] = np.mean([trade['pnl'] for trade in self.all_trades])
            self.portfolio_metrics['avg_holding_period'] = np.mean([trade['holding_period'] for trade in self.all_trades])
        else:
            # Set default values if no trades
            self.portfolio_metrics['win_rate'] = 0
            self.portfolio_metrics['total_trades'] = 0
            self.portfolio_metrics['avg_trade_return'] = 0
            self.portfolio_metrics['avg_trade_pnl'] = 0
            self.portfolio_metrics['avg_holding_period'] = 0
            
        return self.portfolio_metrics
    
    def visualize_portfolio(self, figsize=(20, 14)):
        """
        Creates a professional and visually appealing portfolio performance visualization
        """
        if self.portfolio_equity_curve is None:
            print("No portfolio data to visualize")
            return
        
        # Refined Professional Color Palette
        color_palette = {
            'background': '#FFFFFF',  # Clean white background
            'primary': '#2C3E50',     # Deep navy blue
            'secondary': '#34495E',   # Slightly lighter navy
            'accent': '#3498DB',      # Bright azure blue
            'text': '#2F4F4F',        # Dark slate gray
            'grid': '#ECF0F1',        # Light gray grid
            'highlight': '#E74C3C',   # Vibrant red for drawdowns
            'equity': '#2980B9',      # Professional blue for equity curve
            'chart_background': '#F7F9FA'  # Very light gray-blue for chart backgrounds
        }
        
        # Set up professional matplotlib style
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'Arial',
            'axes.labelcolor': color_palette['text'],
            'xtick.color': color_palette['text'],
            'ytick.color': color_palette['text']
        })
        
        fig = plt.figure(figsize=figsize, facecolor=color_palette['background'], dpi=300)
        fig.suptitle('Portfolio Performance Dashboard', 
                    fontsize=26, 
                    fontweight='bold', 
                    color=color_palette['primary'], 
                    y=0.98)
        
        # More sophisticated grid layout
        gs = GridSpec(3, 2, figure=fig, wspace=0.15, hspace=0.2)
        
        # Filter data from a specific start date
        start_date = "2021-11-01"
        self.portfolio_equity_curve = self.portfolio_equity_curve.loc[self.portfolio_equity_curve.index >= start_date]
        self.portfolio_cum_returns = self.portfolio_cum_returns.loc[self.portfolio_cum_returns.index >= start_date]
        self.portfolio_daily_returns = self.portfolio_daily_returns.loc[self.portfolio_daily_returns.index >= start_date]
        
        # Sophisticated Equity Curve with Drawdown
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.portfolio_equity_curve.index, self.portfolio_equity_curve, 
                color=color_palette['equity'], linewidth=3, label='Equity Curve')
        
        # Refined Drawdown Overlay
        cum_returns = self.portfolio_cum_returns
        drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
        drawdown.fillna(0, inplace=True)
        
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(drawdown.index, 0, drawdown, 
                            color=color_palette['highlight'], alpha=0.3, label='Drawdown')
        
        ax1.set_title('Portfolio Equity Trajectory', 
                    fontsize=16, 
                    color=color_palette['primary'], 
                    fontweight='bold')
        ax1.grid(True, linestyle='--', linewidth=0.5, color=color_palette['grid'])
        
        # Performance Metrics with Enhanced Styling
        ax_metrics = fig.add_subplot(gs[1, 0])
        metrics_data = [
            ["Performance Metrics", "Value"],
            ["Total Return", f"{self.portfolio_metrics.get('total_return', 0):.2%}"],
            ["Annualized Return", f"{self.portfolio_metrics.get('annualized_return', 0):.2%}"],
            ["Sharpe Ratio", f"{self.portfolio_metrics.get('sharpe_ratio', 0):.2f}"],
            ["Max Drawdown", f"{self.portfolio_metrics.get('max_drawdown', 0):.2%}"],
            ["Win Rate", f"{self.portfolio_metrics.get('win_rate', 0):.2%}"],
            ["Average Trade PnL", f"${self.portfolio_metrics.get('avg_trade_pnl', 0):.2f}"]
        ]
        
        table = ax_metrics.table(cellText=metrics_data, 
                                loc='center', 
                                cellLoc='center', 
                                colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        # Modern Table Styling
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor(color_palette['grid'])
            if i == 0:  # Header row
                cell.set_facecolor(color_palette['primary'])
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('white' if i % 2 == 0 else color_palette['chart_background'])
        
        ax_metrics.axis('off')
        
        # Rolling Sharpe Ratio with Professional Design
        ax_rolling = fig.add_subplot(gs[1, 1])
        rolling_sharpe = (self.portfolio_daily_returns.rolling(window=30).mean() /
                        self.portfolio_daily_returns.rolling(window=30).std())
        ax_rolling.plot(rolling_sharpe.index, rolling_sharpe, 
                        color=color_palette['accent'], linewidth=2)
        ax_rolling.set_title('Rolling 30-Day Sharpe Ratio', 
                            fontsize=14, 
                            color=color_palette['primary'], 
                            fontweight='bold')
        ax_rolling.axhline(y=0, color=color_palette['highlight'], linestyle='--')
        ax_rolling.grid(True, linestyle='--', linewidth=0.5, color=color_palette['grid'])
        
        # Pairwise Performance Charts
        ax_equity = fig.add_subplot(gs[2, 0])
        ax_box = fig.add_subplot(gs[2, 1])
        
        box_positions, box_data = [], []
        
        for i, (pair, data) in enumerate(self.pair_results.items()):
            if 'equity_curve' in data:
                data['equity_curve'] = data['equity_curve'].loc[data['equity_curve'].index >= start_date]
                ax_equity.plot(data['equity_curve'].index, data['equity_curve'], 
                            label=pair, linewidth=2, alpha=0.7)
            
            if 'daily_returns' in data:
                returns = data['daily_returns'].dropna()
                returns = returns.loc[returns.index >= start_date]
                box_data.append(returns)
                box_positions.append(i + 1)
        
        ax_equity.set_title('Pairwise Equity Curves', 
                            fontsize=14, 
                            color=color_palette['primary'], 
                            fontweight='bold')
        ax_equity.legend(loc='best')
        ax_equity.grid(True, linestyle='--', linewidth=0.5, color=color_palette['grid'])
        
        # Enhanced Box Plot
        box_props = dict(linestyle='-', linewidth=1.5, color=color_palette['secondary'])
        ax_box.boxplot(box_data, positions=box_positions, widths=0.5, 
                    patch_artist=True, boxprops=box_props)
        ax_box.set_xticks(box_positions)
        ax_box.set_xticklabels([pair for pair in self.pair_results.keys()], 
                                rotation=45, ha='right')
        ax_box.set_title('Pairwise Returns Distribution', 
                        fontsize=14, 
                        color=color_palette['primary'], 
                        fontweight='bold')
        ax_box.grid(True, linestyle='--', linewidth=0.5, color=color_palette['grid'])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('Portfolio/Portfolio_Performance_Visualization.png', 
                    dpi=300, 
                    bbox_inches='tight', 
                    facecolor=color_palette['background'])
        
        return fig

        
    def export_results(self, filename="Portfolio/portfolio_results.csv"):
        """Export portfolio results to CSV"""
        if self.portfolio_daily_returns is None:
            print("No results to export")
            return
            
        # Create a DataFrame with portfolio performance
        results_df = pd.DataFrame({
            'Date': self.portfolio_daily_returns.index,
            'Daily_Return': self.portfolio_daily_returns.values,
            'Cumulative_Return': self.portfolio_cum_returns.values,
            'Equity': self.portfolio_equity_curve.values
        })
        
        results_df.set_index('Date', inplace=True)
        
        # Add individual pair returns
        for pair_name, pair_data in self.pair_results.items():
            if 'daily_returns' in pair_data and not pair_data['daily_returns'].empty:
                # Reindex to match the portfolio dates
                pair_returns = pair_data['daily_returns'].reindex(results_df.index).fillna(0)
                results_df[f'{pair_name}_return'] = pair_returns
            else:
                # Handle pairs with no trades
                results_df[f'{pair_name}_return'] = 0
            
        # Save to CSV
        results_df.to_csv(filename)
        print(f"Results exported to {filename}")
        
        return results_df