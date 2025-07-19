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

    """
    Pairs trading backtest engine with Kelly criterion position sizing.
    
    This class implements a mean-reversion pairs trading strategy with dynamic position sizing,
    risk management through stop-loss and take-profit levels, and detailed performance analytics.
    The strategy uses z-score based signals filtered through mean reversion conditions and 
    applies the Kelly criterion for optimal position sizing.
    
    Key Features:
    - Kelly criterion position sizing with risk-adjusted scaling
    - Dynamic stop-loss and take-profit levels based on volatility
    - Maximum holding period constraints
    - Concurrent position management with portfolio-level risk controls
    - Comprehensive performance metrics and visualization
    - Transaction cost modeling
    
    Attributes:
        stock1 (str): Symbol for the first stock in the pair
        stock2 (str): Symbol for the second stock in the pair  
        data (pd.DataFrame): Historical price data for both stocks
        final_df (pd.DataFrame): Processed dataframe with prediction scores
        z_score (pd.Series): Z-score time series for mean reversion signals
        returns (pd.DataFrame): Daily returns for both stocks
        cum_returns (pd.Series): Cumulative strategy returns
        daily_returns (pd.Series): Daily strategy returns
        strategy_returns (pd.Series): Raw strategy returns before compounding
        position_sizes (pd.Series): Time series of total position sizes
        trades (pd.Series): Time series of trade signals
        metrics (dict): Performance metrics dictionary
        completed_trades (list): List of completed trade records
        
    Risk Management Parameters:
        base_stop_loss_scale (float): Base stop loss as fraction of entry (default: 0.001)
        risk_reward_ratio (float): Take profit to stop loss ratio (default: 1.3)
        max_holding (int): Maximum holding period in days (default: 7)
        
    Position Sizing Parameters:
        min_position_size (float): Minimum position size (default: 0.1)
        max_position_size (float): Maximum position size (default: 2.5)  
        max_concurrent_positions (int): Maximum number of concurrent positions (default: 50)
        first_position_size (float): Size for first position (default: 0.50)
        position_scaling_factor (float): Risk scaling factor (default: 1.0)
        
    Transaction Costs:
        transaction_cost_fixed (float): Fixed cost per trade (default: 0.0)
        transaction_cost_percentage (float): Percentage cost per trade (default: 0.001)
        
    Volatility Parameters:
        z_volatility_window (int): Window for z-score volatility calculation (default: 2)
        volatility_factor (float): Volatility adjustment factor for stop loss (default: 0)
        
    Trailing Stop Parameters:
        trailing_stop_loss (bool): Enable trailing stop loss (default: False)
        trailing_threshold (float): Trailing stop threshold (default: 0.20)
    """

    def __init__(self, stock1, stock2, data, final_df, normalized_df, combined_final, returns):
        self.stock1 = stock1
        self.stock2 = stock2
        self.data = data
        self.combined_final = combined_final  
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
        
        # Parameters for stop loss, take profit and max holding period
        self.base_stop_loss_scale = 0.001 # \\ 0.001, 1.3, 7, window [5,3], volatiltiy factor 0
        self.risk_reward_ratio =  1.3 # 
        self.max_holding = 7
        
        # Position sizing parameters
        self.min_position_size = 0.1
        self.max_position_size = 2.5
        self.max_concurrent_positions = 50
        self.first_position_size = 0.50
        self.position_scaling_factor = 1.0 # \\ Used as a risk factor control for position sizing
        
        # Transaction costs
        self.transaction_cost_fixed = 0.0
        self.transaction_cost_percentage = 0.001 # (0.1% transaction cost per trade)

        # Volatillity
        self.z_volatility_window = 2
        self.volatility_factor = 0 # Used to dynamically adjust stop loss size.
        
        # Direction tracking
        self.direction_success = {1: 0, -1: 0}
        self.direction_positions = {1: 0, -1: 0}
        self.active_positions = []

        # Trailing stop loss parameters
        self.trailing_stop_loss = False # Enable/disable trailing stop loss
        self.trailing_threshold = 0.20  # 20% drawdown from peak profit triggers exit

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
        # 1. For long positions (signal=1): z-score must be below -1.00 (oversold)
        # 2. For short positions (signal=-1): z-score must be above 1.00 (overbought)
        
        # Initialize filtered prediction
        backtest_df['filtered_prediction'] = 0
        
        # Apply mean reversion logic for longs: signal=1 and z-score < -1.00
        long_condition = (backtest_df['prediction'] == 1) & (z_score < -1.0)
        backtest_df.loc[long_condition, 'filtered_prediction'] = 1
        
        # Apply mean reversion logic for shorts: signal=-1 and z-score > 1.00
        short_condition = (backtest_df['prediction'] == -1) & (z_score > 1.0)
        backtest_df.loc[short_condition, 'filtered_prediction'] = -1
        
        # Store trades for analysis
        self.trades = backtest_df['filtered_prediction'].copy()
        
        # Get stock returns for both stocks in the pair
        stock1_returns = self.returns[self.stock1]
        stock2_returns = self.returns[self.stock2]

        # Ensure datetime index
        stock1_returns.index = pd.to_datetime(stock1_returns.index).tz_localize(None)
        stock2_returns.index = pd.to_datetime(stock2_returns.index).tz_localize(None)

        backtest_df.index = pd.to_datetime(backtest_df.index).tz_localize(None)
        
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
        
        for i in range(len(position_size_df)):
            current_date = position_size_df.index[i].replace(tzinfo=None)
            daily_total_return = 0.0
            positions_to_remove = []
            
            # Calculate daily returns for existing positions and check for exits
            for idx, position in enumerate(self.active_positions):
                # Update days held
                position['days_held'] += 1
                position['entry_date'] = position['entry_date'].replace(tzinfo=None)
                
                # Calculate actual cumulative returns for the specific position
                if i > 0 and current_date in stock1_returns.index and current_date in stock2_returns.index:
                    # Pairs trade return calculation for this specific position
                    daily_pos_return = ((stock1_returns.loc[current_date] * position['direction']) - 
                        (stock2_returns.loc[current_date] * position['direction'])) * position['position_size']
                    
                    # Update cumulative return for this specific position
                    position['cumulative_return'] += daily_pos_return
                    daily_total_return += daily_pos_return
                    
                    # Simplified Stop Loss Condition

                    # Z-Score Exit Conditions
                    current_z_score = z_score.loc[current_date]

                    # ------------ z_score based exit ----------
                    
                    # Exit long position conditions
                    # if position['direction'] == 1:
                    #     # Exit long if z-score falls below stop loss (mean reversion)
                    #     stop_loss_exit = current_z_score >= position['entry_zscore'] + dynamic_stop_loss
                    
                    # # Exit short position conditions  
                    # elif position['direction'] == -1:
                    #     # Exit short if z-score rises above stop loss (mean reversion)
                    #     stop_loss_exit = current_z_score >= position['entry_zscore'] - dynamic_stop_loss
                    
                    # # Exit long position conditions - take profit
                    # if position['direction'] == 1:
                    #     # Exit long if z-score rises above take profit (mean reversion)
                    #     take_profit_exit = current_z_score >= position['entry_zscore'] + dynamic_take_profit
                    
                    # # Exit short position conditions  
                    # elif position['direction'] == -1:
                    #     # Exit short if z-score falls bellow take profit (mean reversion)
                    #     take_profit_exit = current_z_score >= position['entry_zscore'] - dynamic_take_profit

                    # ------- Cumulative return based exit ----------

                    # Exit if absolute cumulative return exceeds stop loss level
                    stop_loss_exit = (position['cumulative_return']) <= (position['stop_loss_level'])
                    
                    # Take Profit Condition
                    take_profit_exit = (position['cumulative_return']) >= (position['take_profit_level'])

                    # Maximum Holding Period
                    max_holding_period_exit = position['days_held'] >= self.max_holding
                    
                    # Determine if we should exit this specific position
                    exit_position = stop_loss_exit or take_profit_exit or max_holding_period_exit
                    
                    if exit_position:
                        positions_to_remove.append(idx)
                        
                        # Determine exit reason
                        exit_reason = (
                            "Stop Loss" if stop_loss_exit else
                            "Take Profit" if take_profit_exit else
                            "Max Holding Period"
                        )
                        
                        # Record completed trade details
                        trade_details = {
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'holding_period': position['days_held'],
                            'direction': "Long/Short" if position['direction'] == 1 else "Short/Long",
                            'return': position['cumulative_return'],
                            'pnl': position['cumulative_return'] * position['position_size'],
                            'position_size': position['position_size'],
                            'exit_reason': exit_reason
                        }
                        self.completed_trades.append(trade_details)
            
            # Remove closed positions
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
                
                current_z_volatility = z_score.rolling(window=5).std()[i]
                
                # Calculate dynamic stop loss INDEPENDENTLY of position sizing
                volatility_factor = 0 if self.volatility_factor == 0 else max(current_z_volatility * (self.volatility_factor), 0.01)
                
                # Stop loss calculation is now completely separate from position sizing
                dynamic_stop_loss = -self.base_stop_loss_scale * (1 + volatility_factor)
                
                # Take profit is risk-reward multiple of stop loss
                dynamic_take_profit = abs(dynamic_stop_loss) * self.risk_reward_ratio
                
                # Create new position with placeholder size to be determined by Kelly
                new_position = {
                    'direction': current_direction,
                    'is_new': True,  # Flag as new
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
        Applies the Kelly criterion to new positions only.
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

        # Separate new positions
        new_positions = [pos for pos in self.active_positions if pos.get('is_new', False)]

        for pos in new_positions:  # Apply the Kelly criterion only to new positions
            p = pos.get('win_prob', current_probability)
            q = 1 - p

            loss_pct = 1.0  
            win_pct = self.risk_reward_ratio  

            # Kelly Formula: f* = (p / loss_pct) - (q / win_pct)
            kelly_fraction = (p / loss_pct) - (q / win_pct)
            half_kelly = max(0, kelly_fraction * 0.5)  # Ensure non-negative Kelly fraction

            z_score_volatility = self.z_score.rolling(window=5).std()
            curr_z_volatility = z_score_volatility.loc[current_date]
            curr_z_score = self.z_score.loc[current_date]

            # Ensure volatility is nonzero to avoid division errors
            if curr_z_volatility == 0 or np.isnan(curr_z_volatility):
                curr_z_volatility = 1e-6  # Small value to prevent division by zero

            weight_kelly = 0.9  # This ensure the half kelly maintains dominance
            weight_adjustment = 1 - weight_kelly  # Keeps other parameters so they still contribute, but smaller variation

            # Calculate final position size
            final_position_size = (weight_kelly * half_kelly) + (weight_adjustment * half_kelly * (1 / np.sqrt(curr_z_volatility)) * self.position_scaling_factor)

            # Ensure position size is between 0.1 and 1
            pos['position_size'] = min(1, max(0.1, final_position_size))

        # Optionally mark positions as "not new" after sizing
        for pos in new_positions:
            pos['is_new'] = False  # Mark position as processed for future runs

        # Optional: update the direction_positions as needed
        # for direction, positions in positions_by_direction.items():
        #     self.direction_positions[direction] = len(positions)

    def calculate_rolling_metrics(self):
        """Calculate rolling performance metrics with updated Sharpe ratio calculation."""
        
        rolling_returns = pd.DataFrame(index=self.daily_returns.index)
        rolling_returns['1D'] = self.daily_returns
        rolling_returns['5D'] = self.daily_returns.rolling(window=5).sum()
        rolling_returns['10D'] = self.daily_returns.rolling(window=10).sum()
        rolling_returns['20D'] = self.daily_returns.rolling(window=20).sum()
        rolling_returns['60D'] = self.daily_returns.rolling(window=60).sum()

        # Custom rolling volatility calculation
        rolling_volatility = pd.DataFrame(index=self.daily_returns.index)
        for window in [10, 20, 60]:
            rolling_mean = self.daily_returns.rolling(window=window).mean()
            rolling_var = ((self.daily_returns - rolling_mean) ** 2).rolling(window=window).mean()
            rolling_volatility[f'{window}D'] = rolling_var

        # Updated rolling Sharpe ratio calculation
        rolling_sharpe = pd.DataFrame(index=self.daily_returns.index)
        for window in [10, 20, 60]:
            avg_return = self.daily_returns.rolling(window=window).mean()
            vol = rolling_volatility[f'{window}D']
            rolling_sharpe[f'{window}D'] = (avg_return / vol) * np.sqrt(252)

        # Rolling drawdown calculation
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
        """Calculate and store performance metrics using the updated Sharpe ratio formula."""
        
        if len(self.cum_returns) > 0:
            total_return = self.cum_returns.iloc[-1]
        else:
            total_return = 0.0

        if len(self.daily_returns) > 0:
            days = len(self.daily_returns)
            annualized_return = ((1 + total_return) ** (252 / days)) - 1
        else:
            annualized_return = 0.0

        # Updated Sharpe ratio calculation
        if len(self.daily_returns) > 0:
            avg_daily_return = self.daily_returns.mean()
            daily_volatility = np.sqrt(((self.daily_returns - avg_daily_return) ** 2).mean())
            
            if daily_volatility > 0:
                sharpe_ratio = (avg_daily_return / daily_volatility) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown calculation
        if len(self.daily_returns) > 0:
            cum_returns = (1 + self.daily_returns).cumprod()
            max_returns = cum_returns.cummax()
            drawdowns = (cum_returns / max_returns) - 1
            max_drawdown = drawdowns.min()
        else:
            max_drawdown = 0.0

        # Store the updated metrics
        self.metrics['sharpe_ratio'] = sharpe_ratio
        self.metrics['max_drawdown'] = max_drawdown


        # Calculate expectancy and win rate from completed trades
        if self.completed_trades:
            profitable_trades = [trade for trade in self.completed_trades if trade['pnl'] > 0]
            losing_trades = [trade for trade in self.completed_trades if trade['pnl'] <= 0]

            win_rate = len(profitable_trades) / len(self.completed_trades)

            avg_profit = sum(trade['pnl'] for trade in profitable_trades) / len(profitable_trades) if profitable_trades else 0.0
            avg_loss = sum(trade['pnl'] for trade in losing_trades) / len(losing_trades) if losing_trades else 0.0

            # Calculate expectancy
            loss_rate = 1 - win_rate
            expectancy = (win_rate * avg_profit) - (loss_rate * abs(avg_loss))

            # Calculate profit factor
            total_profit = sum(trade['pnl'] for trade in profitable_trades)
            total_loss = sum(abs(trade['pnl']) for trade in losing_trades)
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            expectancy = 0.0

        # Store the metrics
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'total_trades': len(self.completed_trades)
        }
        
        # Print the metrics
        print('📊 Performance Metrics:')
        print(f'Total Return: {total_return:.2%}')
        print(f'Annualized Return: {annualized_return:.2%}')
        print(f'Expectancy : {expectancy:.2%}')
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
        fig = plt.figure(figsize=(15, 25), dpi=100)  # Increased height further
        gs = GridSpec(6, 2, figure=fig, height_ratios=[0.5, 1, 1, 1, 1, 1])  # Added a new row
        
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
            ('Expectancy', f"{backtest_instance.metrics['expectancy']:.2%}"),
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
            
        # Plot stop loss levels
        for stop_loss in backtest_instance.stop_loss_levels:
            if stop_loss['direction'] == 1:  # Long position
                ax1.scatter(stop_loss['date'], stop_loss['entry_zscore'] + stop_loss['level'], 
                            marker='_', s=50, color=bright_red, 
                            edgecolor=bright_red, linewidth=1.5, alpha=0.7)
            else:  # Short position
                ax1.scatter(stop_loss['date'], stop_loss['entry_zscore'] - stop_loss['level'], 
                            marker='_', s=50, color=bright_red, 
                            edgecolor=bright_red, linewidth=1.5, alpha=0.7)

        # Plot take profit levels
        for take_profit in backtest_instance.take_profit_levels:
            if take_profit['direction'] == 1: # Long position
                ax1.scatter(take_profit['date'], take_profit['entry_zscore'] + take_profit['level'], 
                            marker='_', s=50, color=bright_green, 
                            edgecolor=bright_green, linewidth=1.5, alpha=0.7)
            else:
                ax1.scatter(take_profit['date'], take_profit['entry_zscore'] - take_profit['level'], 
                            marker='_', s=50, color=bright_green, 
                            edgecolor=bright_green, linewidth=1.5, alpha=0.7)

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
        
        
        # 5. TRADE RETURNS DISTRIBUTION and POSITION SIZES side by side (previously 5.)
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


        # 5. STOCK PRICES WITH TRADES
        ax_prices = fig.add_subplot(gs[5, :])
        ax_prices.set_facecolor(background_color)

        # Fetch stock prices for the two stocks
        stock1_prices = self.data[self.stock1]
        stock2_prices = self.data[self.stock2]

        # Normalize prices to start at 100 for comparison
        stock1_norm = stock1_prices / stock1_prices.iloc[0] * 100
        stock2_norm = stock2_prices / stock2_prices.iloc[0] * 100

        # Plot normalized prices
        ax_prices.plot(stock1_norm.index, stock1_norm, color=orange, label=self.stock1, linewidth=2, alpha=0.7)
        ax_prices.plot(stock2_norm.index, stock2_norm, color=pale_purple, label=self.stock2, linewidth=2, alpha=0.7)

        # Safely plot trade entry points
        if hasattr(backtest_instance, 'entry_points') and backtest_instance.entry_points:
            for entry in backtest_instance.entry_points:
                # Convert entry date to datetime and remove time component
                entry_date = pd.to_datetime(entry['date']).normalize()
                
                # Determine actual positions for each stock in the pair
                if entry['direction'] == 1:
                    stock1_marker, stock1_color = '^', bright_green  # Long Stock 1
                    stock2_marker, stock2_color = 'v', bright_red    # Short Stock 2
                else:  # entry['direction'] == -1
                    stock1_marker, stock1_color = 'v', bright_red    # Short Stock 1
                    stock2_marker, stock2_color = '^', bright_green  # Long Stock 2
                
                try:
                    # Explicitly find the closest index
                    def find_closest_index(index, target_date):
                        # Convert to numpy datetime64 if needed
                        if not isinstance(target_date, np.datetime64):
                            target_date = np.datetime64(target_date)
                        
                        # Convert index to numpy datetime64 array
                        dates_array = index.to_numpy()
                        
                        # Find the index of the closest date
                        closest_idx = np.abs(dates_array - target_date).argmin()
                        return closest_idx
                    
                    # Find closest indices
                    stock1_closest_idx = find_closest_index(stock1_norm.index, entry_date)
                    stock2_closest_idx = find_closest_index(stock2_norm.index, entry_date)
                    
                    # Get dates and values
                    stock1_closest_date = stock1_norm.index[stock1_closest_idx]
                    stock2_closest_date = stock2_norm.index[stock2_closest_idx]
                    
                    # Plot markers for both stocks based on actual positions
                    ax_prices.scatter(stock1_closest_date, stock1_norm.loc[stock1_closest_date], 
                                    marker=stock1_marker, s=50, color=stock1_color, 
                                    edgecolor='white', linewidth=0.8, alpha=0.8)
                    ax_prices.scatter(stock2_closest_date, stock2_norm.loc[stock2_closest_date], 
                                    marker=stock2_marker, s=50, color=stock2_color, 
                                    edgecolor='white', linewidth=0.8, alpha=0.8)
                except Exception as e:
                    print(f"Could not plot entry point for date {entry_date}: {e}")
                    continue
        
        # Set title and labels
        ax_prices.set_title('Normalized Stock Prices with Trade Entries', color=royal_blue, fontsize=14, fontweight='bold')
        ax_prices.set_ylabel('Normalized Price (Start = 100)', color='black', fontsize=12)
        ax_prices.grid(False)
        ax_prices.legend(loc='best')
        
        # Format x-axis
        ax_prices.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_prices.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax_prices.xaxis.get_majorticklabels(), rotation=45)
        
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

    """
    A portfolio backtesting class that aggregates and analyzes 
    performance across multiple trading pairs.
    
    This class combines individual pair backtest results into a unified portfolio
    analysis, providing portfolio-level performance metrics, visualizations, 
    and risk analytics. It supports equal weighting strategies and calculates
    advanced risk-adjusted returns including Sharpe, Sortino, and Calmar ratios.
    
    Attributes:
        pair_results (dict): Dictionary storing backtest results for each trading pair
        portfolio_daily_returns (pd.Series): Daily returns of the combined portfolio
        portfolio_cum_returns (pd.Series): Cumulative returns of the portfolio
        portfolio_equity_curve (pd.Series): Portfolio equity curve over time
        portfolio_metrics (dict): Dictionary of calculated performance metrics
        all_trades (list): Consolidated list of all trades across all pairs
    """

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
        """Calculate performance metrics for the overall portfolio with updated Sharpe ratio calculation."""
        
        if self.portfolio_daily_returns is None:
            print("No portfolio returns calculated yet")
            return

        returns = self.portfolio_daily_returns

        # Annual risk-free rate (e.g., 2% = 0.02) converted to a daily rate
        risk_free_rate = 0.00 / 252  

        # Basic metrics
        self.portfolio_metrics['total_return'] = self.portfolio_cum_returns.iloc[-1]
        self.portfolio_metrics['annualized_return'] = self.portfolio_cum_returns.iloc[-1] / (len(returns) / 252)

        # Custom volatility calculation
        avg_daily_return = returns.mean()
        volatility = np.sqrt(((returns - avg_daily_return) ** 2).mean())  

        self.portfolio_metrics['annualized_volatility'] = volatility * np.sqrt(252)

        # Updated Sharpe Ratio Calculation
        excess_returns = returns - risk_free_rate
        if volatility > 0:
            self.portfolio_metrics['sharpe_ratio'] = (excess_returns.mean() / volatility) * np.sqrt(252)
        else:
            self.portfolio_metrics['sharpe_ratio'] = np.nan  # Avoid division by zero

        # Sortino Ratio - calculate downside deviation first
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.sqrt(((negative_returns - negative_returns.mean()) ** 2).mean()) * np.sqrt(252)
            self.portfolio_metrics['sortino_ratio'] = (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else np.nan
        else:
            self.portfolio_metrics['sortino_ratio'] = np.nan

        # Drawdown analysis 
        wealth_index = (1 + returns).cumprod()
        running_max = wealth_index.cummax()
        drawdown = (wealth_index / running_max) - 1
        self.portfolio_metrics['max_drawdown'] = drawdown.min()

        # Calmar ratio: Annualized return / Max drawdown
        if abs(self.portfolio_metrics['max_drawdown']) > 0:
            self.portfolio_metrics['calmar_ratio'] = self.portfolio_metrics['annualized_return'] / abs(self.portfolio_metrics['max_drawdown'])
        else:
            self.portfolio_metrics['calmar_ratio'] = np.nan

        # Win rate and trade metrics
        if self.all_trades:
            profitable_trades = sum(1 for trade in self.all_trades if trade['pnl'] > 0)
            self.portfolio_metrics['win_rate'] = profitable_trades / len(self.all_trades)
            self.portfolio_metrics['total_trades'] = len(self.all_trades)
            self.portfolio_metrics['avg_trade_return'] = np.mean([trade['return'] for trade in self.all_trades])
            self.portfolio_metrics['avg_trade_pnl'] = np.mean([trade['pnl'] for trade in self.all_trades])
            self.portfolio_metrics['avg_holding_period'] = np.mean([trade['holding_period'] for trade in self.all_trades])
        else:
            self.portfolio_metrics.update({
                'win_rate': 0,
                'total_trades': 0,
                'avg_trade_return': 0,
                'avg_trade_pnl': 0,
                'avg_holding_period': 0
            })

        return self.portfolio_metrics
    
    def visualize_portfolio(self, portfolio_df, figsize=(20, 14)):
        """
        Creates a professional and visually appealing portfolio performance visualization
        with enhanced performance metrics including Sortino and Calmar ratios
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
            'highlight': '#FF8C00',   # Dark orange for drawdowns
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
        
        gs = GridSpec(3, 2, figure=fig, wspace=0.1, hspace=0.2)
        
        # Equity Curve with Drawdown
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.portfolio_equity_curve.index, self.portfolio_equity_curve, 
                color=color_palette['primary'], linewidth=3, label='Equity Curve')
        
        ax1.legend(loc='upper left', fontsize=12)
        
        # Drawdown Overlay
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
        
        # Calculate Sortino Ratio if not already in portfolio_metrics
        if 'sortino_ratio' not in self.portfolio_metrics:
            # Only consider negative returns for downside deviation
            negative_returns = self.portfolio_daily_returns[self.portfolio_daily_returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252)
            
            # Avoid division by zero
            if downside_deviation != 0:
                self.portfolio_metrics['sortino_ratio'] = (self.portfolio_metrics.get('annualized_return', 0) / downside_deviation)
            else:
                self.portfolio_metrics['sortino_ratio'] = np.nan
        
        # Enhanced Performance Metrics with Sortino and Calmar ratios
        ax_metrics = fig.add_subplot(gs[1, 0])
        metrics_data = [
            ["Performance Metrics", "Value"],
            ["Total Return", f"{self.portfolio_metrics.get('total_return', 0):.2%}"],
            ["Annualized Return", f"{self.portfolio_metrics.get('annualized_return', 0):.2%}"],
            ["Annualized Volatility", f"{self.portfolio_metrics.get('annualized_volatility', 0):.2%}"],
            ["Sharpe Ratio", f"{self.portfolio_metrics.get('sharpe_ratio', 0):.2f}"],
            ["Sortino Ratio", f"{self.portfolio_metrics.get('sortino_ratio', 0):.2f}"],
            ["Calmar Ratio", f"{self.portfolio_metrics.get('calmar_ratio', 0):.2f}"],
            ["Max Drawdown", f"{self.portfolio_metrics.get('max_drawdown', 0):.2%}"],
            ["Win Rate", f"{self.portfolio_metrics.get('win_rate', 0):.2%}"]
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

        # Boxplot positions and data initialization
        box_positions, box_data = [], []

        for i, (pair, data) in enumerate(self.pair_results.items()):
            print(f"Processing pair: {pair}")

            if pair in portfolio_df:  # Ensure portfolio_df contains the pair
                # Equity curve (cumulative sum of raw data, not daily returns)
                equity_data = portfolio_df[pair].cumsum()  # Cumulative sum for equity curve

                # Only plot equity curve if there is variation
                if len(equity_data) > 1 and equity_data.nunique() > 1:
                    try:
                        ax_equity.plot(equity_data.index, equity_data, 
                                        label=pair, linewidth=2, alpha=0.7)
                    except Exception as e:
                        print(f"  - Error plotting {pair} equity curve: {e}")
                else:
                    print(f"  - Skipping plot for {pair} - insufficient variation in data")

                # Calculate daily returns for the boxplot
                daily_returns = portfolio_df[pair].pct_change().dropna()  # Daily returns (percentage change)
                
                # Only include boxplot data if there is variation in daily returns
                if len(daily_returns) > 1 and daily_returns.nunique() > 1:
                    box_positions.append(i + 1)  # Position for boxplot (1-based index)
                    box_data.append(daily_returns.values)  # Store daily returns for the boxplot

            else:
                print(f"  - No data for {pair} in portfolio_df")

        # Plot settings for equity curves
        ax_equity.set_title('Pairwise Equity Curves', 
                            fontsize=14, 
                            color=color_palette['primary'], 
                            fontweight='bold')
        
        # Only set legend if we have data
        if len(ax_equity.get_lines()) > 0:
            ax_equity.legend(loc='best')
        else:
            print("No data for pairwise equity curves")
            
        ax_equity.grid(True, linestyle='--', linewidth=0.5, color=color_palette['grid'])
        
        # Box Plot - only if we have data
        if box_data and box_positions:
            box_props = dict(linestyle='-', linewidth=1.5, color=color_palette['secondary'])
            ax_box.boxplot(box_data, positions=box_positions, widths=0.5, 
                        patch_artist=True, boxprops=box_props)
            ax_box.set_xticks(box_positions)
            ax_box.set_xticklabels([pair for pair in self.pair_results.keys()], 
                                rotation=45, ha='right')
            ax_box.set_title('Pairwise Daily Returns Distribution', 
                            fontsize=14, 
                            color=color_palette['primary'], 
                            fontweight='bold')
            ax_box.grid(True, linestyle='--', linewidth=0.5, color=color_palette['grid'])
        else:
            print("No data for box plot")
            ax_box.set_title('No Data for Daily Returns Distribution', 
                            fontsize=14, 
                            color=color_palette['primary'], 
                            fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        print("Saving visualization...")
        plt.savefig('Portfolio/Portfolio_Performance_Visualization.png', 
                    dpi=300, 
                    bbox_inches='tight', 
                    facecolor=color_palette['background'])
        
        print("Visualization complete")
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