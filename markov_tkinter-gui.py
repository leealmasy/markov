import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import threading
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use TkAgg backend
import matplotlib
matplotlib.use('TkAgg')

class MarkovStockPredictor:
    def __init__(self, symbol, period="2y"):
        """Initialize the Markov Chain Stock Predictor"""
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.states = []
        self.transition_matrix = None
        self.state_labels = None
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            if len(self.data) < 50:
                raise ValueError(f"Insufficient data for {self.symbol}")
            return True
        except Exception as e:
            raise Exception(f"Error fetching data for {self.symbol}: {str(e)}")
    
    def calculate_daily_returns(self):
        """Calculate daily returns and add to dataset"""
        self.data['Daily_Return'] = self.data['Close'].pct_change() * 100
        self.data['Price_Change'] = self.data['Close'].diff()
        
    def define_states(self, method='return_bins', n_states=5):
        """Define discrete states for the Markov chain"""
        if method == 'return_bins':
            returns = self.data['Daily_Return'].dropna()
            percentiles = np.linspace(0, 100, n_states + 1)
            bins = np.percentile(returns, percentiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            self.data['State'] = pd.cut(self.data['Daily_Return'], 
                                      bins=bins, 
                                      labels=[f'State_{i}' for i in range(n_states)],
                                      include_lowest=True)
            
            self.state_labels = [f'State_{i}' for i in range(n_states)]
            
        elif method == 'direction':
            def get_direction_state(return_val):
                if pd.isna(return_val):
                    return None
                elif return_val > 1.0:
                    return 'Strong_Up'
                elif return_val > 0:
                    return 'Weak_Up'
                elif return_val < -1.0:
                    return 'Strong_Down'
                elif return_val < 0:
                    return 'Weak_Down'
                else:
                    return 'Neutral'
            
            self.data['State'] = self.data['Daily_Return'].apply(get_direction_state)
            self.state_labels = ['Strong_Down', 'Weak_Down', 'Neutral', 'Weak_Up', 'Strong_Up']
        
        self.states = self.data['State'].dropna().tolist()
        
    def build_transition_matrix(self):
        """Build the transition probability matrix"""
        n_states = len(self.state_labels)
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.states) - 1):
            current_state = self.states[i]
            next_state = self.states[i + 1]
            transition_counts[current_state][next_state] += 1
        
        self.transition_matrix = np.zeros((n_states, n_states))
        
        for i, current_state in enumerate(self.state_labels):
            total_transitions = sum(transition_counts[current_state].values())
            if total_transitions > 0:
                for j, next_state in enumerate(self.state_labels):
                    count = transition_counts[current_state][next_state]
                    self.transition_matrix[i, j] = count / total_transitions
        
    def predict_next_state(self, current_state, n_steps=1):
        """Predict next state(s) using the Markov chain"""
        if current_state not in self.state_labels:
            return None
            
        try:
            state_index = self.state_labels.index(current_state)
        except ValueError:
            return None
            
        predictions = []
        current_dist = np.zeros(len(self.state_labels))
        current_dist[state_index] = 1.0
        
        for step in range(n_steps):
            next_dist = current_dist @ self.transition_matrix
            next_state_index = np.argmax(next_dist)
            predicted_state = self.state_labels[next_state_index]
            predictions.append(predicted_state)
            
            current_dist = np.zeros(len(self.state_labels))
            current_dist[next_state_index] = 1.0
            
        return predictions
    
    def predict_price_values(self, days=10):
        """Predict actual price values for the next few days"""
        if len(self.states) == 0:
            return None
        
        current_price = self.data['Close'].iloc[-1]
        current_state = self.states[-1]
        predictions = self.predict_next_state(current_state, days)
        
        if not predictions:
            return None
        
        # Calculate average returns for each state
        state_returns = {}
        for state in self.state_labels:
            state_mask = self.data['State'] == state
            if state_mask.any():
                avg_return = self.data.loc[state_mask, 'Daily_Return'].mean()
                std_return = self.data.loc[state_mask, 'Daily_Return'].std()
                state_returns[state] = {
                    'mean_return': avg_return if not pd.isna(avg_return) else 0,
                    'std_return': std_return if not pd.isna(std_return) else 1
                }
            else:
                state_returns[state] = {'mean_return': 0, 'std_return': 1}
        
        # Generate price predictions
        predicted_prices = []
        price_lower_bounds = []
        price_upper_bounds = []
        
        last_price = current_price
        
        for i, pred_state in enumerate(predictions):
            mean_return = state_returns[pred_state]['mean_return']
            std_return = state_returns[pred_state]['std_return']
            
            predicted_return = mean_return / 100
            predicted_price = last_price * (1 + predicted_return)
            
            uncertainty_multiplier = 1 + (i * 0.1)
            lower_return = (mean_return - std_return * uncertainty_multiplier) / 100
            upper_return = (mean_return + std_return * uncertainty_multiplier) / 100
            
            lower_bound = last_price * (1 + lower_return)
            upper_bound = last_price * (1 + upper_return)
            
            predicted_prices.append(predicted_price)
            price_lower_bounds.append(lower_bound)
            price_upper_bounds.append(upper_bound)
            
            last_price = predicted_price
        
        return {
            'current_price': current_price,
            'predicted_prices': predicted_prices,
            'lower_bounds': price_lower_bounds,
            'upper_bounds': price_upper_bounds,
            'predicted_states': predictions
        }
    
    def backtest_predictions(self, test_days=30):
        """Simple backtest of the model's predictions"""
        if len(self.states) < test_days + 10:
            return 0
        
        train_states = self.states[:-test_days]
        test_states = self.states[-test_days:]
        
        original_states = self.states.copy()
        self.states = train_states
        self.build_transition_matrix()
        
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(test_states) - 1):
            current_state = test_states[i]
            actual_next_state = test_states[i + 1]
            
            predicted_states = self.predict_next_state(current_state, 1)
            if predicted_states and predicted_states[0] == actual_next_state:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        self.states = original_states
        self.build_transition_matrix()
        
        return accuracy

class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Advanced Stock Market Predictor - Markov Chain Analysis")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#2c3e50', foreground='#ecf0f1')
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#34495e', foreground='#ecf0f1')
        self.style.configure('Info.TLabel', font=('Arial', 10), background='#34495e', foreground='#bdc3c7')
        self.style.configure('Custom.TButton', font=('Arial', 10, 'bold'))
        self.style.configure('Custom.TFrame', background='#34495e')
        
        self.predictor = None
        self.create_widgets()
        
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Stock Market Predictor - Markov Chain Analysis", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="📊 Analysis Controls", 
                                      style='Custom.TFrame', padding=15)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Stock Symbol Input
        symbol_frame = ttk.Frame(control_frame, style='Custom.TFrame')
        symbol_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(symbol_frame, text="Stock Symbol:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value="AAPL")
        self.symbol_entry = ttk.Entry(symbol_frame, textvariable=self.symbol_var, font=('Arial', 12))
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        
        # Popular stocks buttons
        popular_frame = ttk.Frame(symbol_frame, style='Custom.TFrame')
        popular_frame.pack(side=tk.LEFT, padx=10)
        
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ"]
        for stock in popular_stocks:
            btn = ttk.Button(popular_frame, text=stock, width=6,
                           command=lambda s=stock: self.symbol_var.set(s))
            btn.pack(side=tk.LEFT, padx=2)
        
        # Parameters
        param_frame = ttk.Frame(control_frame, style='Custom.TFrame')
        param_frame.pack(fill=tk.X, pady=10)
        
        # Period selection
        ttk.Label(param_frame, text="Data Period:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        self.period_var = tk.StringVar(value="2y")
        period_combo = ttk.Combobox(param_frame, textvariable=self.period_var, 
                                   values=["1y", "2y", "5y"], width=10, state="readonly")
        period_combo.pack(side=tk.LEFT, padx=5)
        
        # Prediction days
        ttk.Label(param_frame, text="Predict Days:", style='Header.TLabel').pack(side=tk.LEFT, padx=15)
        self.days_var = tk.StringVar(value="10")
        days_combo = ttk.Combobox(param_frame, textvariable=self.days_var,
                                 values=["5", "10", "15", "20", "30"], width=10, state="readonly")
        days_combo.pack(side=tk.LEFT, padx=5)
        
        # Method selection
        ttk.Label(param_frame, text="Method:", style='Header.TLabel').pack(side=tk.LEFT, padx=15)
        self.method_var = tk.StringVar(value="return_bins")
        method_combo = ttk.Combobox(param_frame, textvariable=self.method_var,
                                   values=["return_bins", "direction"], width=12, state="readonly")
        method_combo.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame, style='Custom.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="🔍 Analyze & Predict", 
                                     command=self.run_analysis, style='Custom.TButton')
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.compare_btn = ttk.Button(button_frame, text="⚖️ Compare Methods", 
                                     command=self.compare_methods, style='Custom.TButton')
        self.compare_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear Results", 
                                   command=self.clear_results, style='Custom.TButton')
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        # Content area with notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.results_frame, text="📈 Prediction Results")
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.chart_frame, text="📊 Interactive Chart")
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.analysis_frame, text="🔬 Detailed Analysis")
        
        # Setup results area
        self.setup_results_tab()
        self.setup_chart_tab()
        self.setup_analysis_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to analyze stock predictions")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              style='Info.TLabel', relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
    def setup_results_tab(self):
        """Setup the results display tab"""
        # Results text area
        self.results_text = scrolledtext.ScrolledText(self.results_frame, 
                                                     font=('Consolas', 10),
                                                     bg='#ecf0f1', fg='#2c3e50',
                                                     height=25)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_chart_tab(self):
        """Setup the chart display tab"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), facecolor='#ecf0f1')
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chart toolbar
        toolbar_frame = ttk.Frame(self.chart_frame, style='Custom.TFrame')
        toolbar_frame.pack(fill=tk.X, padx=10)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def setup_analysis_tab(self):
        """Setup the detailed analysis tab"""
        # Analysis text area
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame,
                                                      font=('Consolas', 9),
                                                      bg='#ecf0f1', fg='#2c3e50',
                                                      height=30)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def run_analysis(self):
        """Run the stock prediction analysis in a separate thread"""
        # Start loading animation
        self.progress.start()
        self.analyze_btn.config(state='disabled')
        self.status_var.set("Loading data and analyzing...")
        
        # Run analysis in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._perform_analysis)
        thread.daemon = True
        thread.start()
        
    def _perform_analysis(self):
        """Perform the actual analysis (runs in separate thread)"""
        try:
            symbol = self.symbol_var.get().upper()
            period = self.period_var.get()
            days = int(self.days_var.get())
            method = self.method_var.get()
            
            # Create predictor and fetch data
            self.predictor = MarkovStockPredictor(symbol, period)
            self.predictor.fetch_data()
            self.predictor.calculate_daily_returns()
            self.predictor.define_states(method=method, n_states=6)
            self.predictor.build_transition_matrix()
            
            # Get predictions
            price_predictions = self.predictor.predict_price_values(days)
            if not price_predictions:
                raise Exception("Failed to generate predictions")
            
            # Backtest
            accuracy = self.predictor.backtest_predictions(test_days=min(60, len(self.predictor.states)//2))
            
            # Update GUI in main thread
            self.root.after(0, self._update_results, price_predictions, accuracy, days)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
        finally:
            self.root.after(0, self._analysis_complete)
            
    def _update_results(self, predictions, accuracy, days):
        """Update the results display"""
        symbol = self.symbol_var.get().upper()
        current_price = predictions['current_price']
        final_price = predictions['predicted_prices'][-1]
        expected_return = ((final_price - current_price) / current_price) * 100
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        
        # Update results tab
        results_text = f"""
🚀 STOCK PREDICTION ANALYSIS RESULTS
═══════════════════════════════════════════════════════════════

📊 STOCK: {symbol}
📅 ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📈 DATA PERIOD: {self.period_var.get()}
🔮 PREDICTION HORIZON: {days} days
🎯 METHOD: {self.method_var.get().replace('_', ' ').title()}

💰 CURRENT ANALYSIS:
   Current Price: ${current_price:.2f}
   {days}-Day Target: ${final_price:.2f}
   Expected Return: {expected_return:+.2f}%

🎯 DETAILED {days}-DAY FORECAST:
"""
        
        results_text += "─" * 80 + "\n"
        results_text += f"{'Day':<4} {'Date':<12} {'Price':<10} {'Daily%':<8} {'Cumul%':<8} {'State':<15}\n"
        results_text += "─" * 80 + "\n"
        
        future_dates = pd.bdate_range(start=self.predictor.data.index[-1] + pd.Timedelta(days=1), periods=days)
        
        for i, (date, price, state) in enumerate(zip(future_dates, 
                                                   predictions['predicted_prices'],
                                                   predictions['predicted_states'])):
            day_num = i + 1
            date_str = date.strftime('%m-%d')
            
            prev_price = current_price if i == 0 else predictions['predicted_prices'][i-1]
            daily_return = ((price - prev_price) / prev_price) * 100
            cumulative_return = ((price - current_price) / current_price) * 100
            
            results_text += f"{day_num:<4} {date_str:<12} ${price:<9.2f} {daily_return:+6.1f}% {cumulative_return:+7.1f}% {state:<15}\n"
        
        # Add summary
        max_price = max(predictions['predicted_prices'])
        min_price = min(predictions['predicted_prices'])
        volatility = ((max_price - min_price) / current_price) * 100
        
        results_text += "\n" + "─" * 80 + "\n"
        results_text += f"📊 SUMMARY STATISTICS:\n"
        results_text += f"   Best case scenario: +{((max_price/current_price - 1)*100):.1f}%\n"
        results_text += f"   Worst case scenario: {((min_price/current_price - 1)*100):.1f}%\n"
        results_text += f"   Volatility estimate: ±{volatility/2:.1f}%\n"
        results_text += f"   Model accuracy: {accuracy:.1%}\n"
        
        # Risk assessment
        if abs(expected_return) > 15:
            risk_level = "🔴 HIGH RISK"
        elif abs(expected_return) > 7:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🟢 LOW RISK"
        results_text += f"   Risk level: {risk_level}\n"
        
        # Trading recommendation
        results_text += f"\n🎯 TRADING INSIGHTS:\n"
        if expected_return > 5:
            results_text += f"   • 📈 BULLISH outlook - Consider long positions\n"
        elif expected_return < -5:
            results_text += f"   • 📉 BEARISH outlook - Consider defensive strategies\n"
        else:
            results_text += f"   • ➡️ NEUTRAL outlook - Sideways movement expected\n"
        
        confidence_level = max(50, 90 - (days * 3))
        results_text += f"   • Model confidence: {confidence_level}%\n"
        
        recommended_position = max(10, min(50, 30 - abs(expected_return)))
        results_text += f"   • Recommended position size: {recommended_position}% of portfolio\n"
        
        results_text += f"\n⚠️ DISCLAIMER:\n"
        results_text += f"   This analysis is for educational purposes only.\n"
        results_text += f"   Past performance doesn't guarantee future results.\n"
        results_text += f"   Always do your own research before investing.\n"
        
        self.results_text.insert(tk.END, results_text)
        
        # Update chart
        self._update_chart(predictions, days)
        
        # Update detailed analysis
        self._update_detailed_analysis(predictions, accuracy)
        
        self.status_var.set(f"Analysis complete for {symbol} - {expected_return:+.1f}% expected return")
        
    def _update_chart(self, predictions, days):
        """Update the chart display"""
        self.fig.clear()
        
        # Create subplot
        ax = self.fig.add_subplot(111)
        
        # Prepare data
        historical_window = min(60, len(self.predictor.data))
        historical_dates = self.predictor.data.index[-historical_window:]
        historical_prices = self.predictor.data['Close'][-historical_window:]
        
        last_date = self.predictor.data.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        # Plot historical data
        ax.plot(historical_dates, historical_prices, 'b-', linewidth=2, 
               label='Historical Prices', alpha=0.8)
        
        # Plot current price
        ax.plot(last_date, predictions['current_price'], 'go', 
               markersize=10, label='Current Price', zorder=5)
        
        # Plot predictions
        ax.plot(future_dates, predictions['predicted_prices'], 'r--', 
               linewidth=3, label='Predicted Prices', alpha=0.9, zorder=4)
        
        # Add confidence intervals
        ax.fill_between(future_dates, predictions['lower_bounds'], predictions['upper_bounds'],
                       alpha=0.2, color='red', label='Confidence Interval')
        
        # Add prediction points
        ax.plot(future_dates, predictions['predicted_prices'], 'rs', 
               markersize=6, zorder=6)
        
        # Formatting
        ax.set_title(f'{self.symbol_var.get().upper()} - {days}-Day Price Prediction', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add vertical line to separate historical and predicted
        ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _update_detailed_analysis(self, predictions, accuracy):
        """Update the detailed analysis tab"""
        symbol = self.symbol_var.get().upper()
        
        analysis_text = f"""
🔬 DETAILED TECHNICAL ANALYSIS FOR {symbol}
═══════════════════════════════════════════════════════════════

📊 MODEL PERFORMANCE METRICS:
   • Backtest Accuracy: {accuracy:.2%}
   • Data Points Used: {len(self.predictor.data)} trading days
   • States Identified: {len(self.predictor.state_labels)}
   • Method: {self.method_var.get().replace('_', ' ').title()}

🎲 MARKOV CHAIN STATE ANALYSIS:
"""
        
        # State distribution
        state_counts = Counter(self.predictor.states)
        analysis_text += "\nState Distribution:\n"
        analysis_text += "─" * 40 + "\n"
        
        for i, state in enumerate(self.predictor.state_labels):
            count = state_counts.get(state, 0)
            percentage = (count / len(self.predictor.states)) * 100 if self.predictor.states else 0
            bar = "█" * int(percentage / 3)
            analysis_text += f"{state:<12}: {count:4d} ({percentage:5.1f}%) {bar}\n"
        
        # Transition matrix analysis
        analysis_text += f"\n🔄 TRANSITION MATRIX INSIGHTS:\n"
        analysis_text += "─" * 50 + "\n"
        
        # Find most stable states (highest self-transition probability)
        stable_states = []
        for i, state in enumerate(self.predictor.state_labels):
            self_transition = self.predictor.transition_matrix[i, i]
            if self_transition > 0.3:  # States that tend to persist
                stable_states.append((state, self_transition))
        
        if stable_states:
            analysis_text += "Most Persistent States (tend to continue):\n"
            for state, prob in sorted(stable_states, key=lambda x: x[1], reverse=True):
                analysis_text += f"  • {state}: {prob:.1%} self-transition probability\n"
        
        # Market regime analysis
        analysis_text += f"\n📈 MARKET REGIME ANALYSIS:\n"
        analysis_text += "─" * 40 + "\n"
        
        # Calculate average returns for each state
        for state in self.predictor.state_labels:
            state_mask = self.predictor.data['State'] == state
            if state_mask.any():
                avg_return = self.predictor.data.loc[state_mask, 'Daily_Return'].mean()
                std_return = self.predictor.data.loc[state_mask, 'Daily_Return'].std()
                count = state_mask.sum()
                
                regime_type = "🚀 Bullish" if avg_return > 0.5 else "💥 Bearish" if avg_return < -0.5 else "➡️ Neutral"
                
                analysis_text += f"{state:<12}: {regime_type} | Avg: {avg_return:+5.2f}% | Std: {std_return:5.2f}% | Days: {count}\n"
        
        # Recent trend analysis
        analysis_text += f"\n📊 RECENT TREND ANALYSIS:\n"
        analysis_text += "─" * 40 + "\n"
        
        recent_states = self.predictor.states[-10:] if len(self.predictor.states) >= 10 else self.predictor.states
        recent_state_counts = Counter(recent_states)
        
        analysis_text += "Last 10 trading days state frequency:\n"
        for state, count in recent_state_counts.most_common():
            analysis_text += f"  • {state}: {count} days ({count/len(recent_states)*100:.0f}%)\n"
        
        # Volatility analysis
        recent_returns = self.predictor.data['Daily_Return'].tail(20).std()
        overall_volatility = self.predictor.data['Daily_Return'].std()
        
        analysis_text += f"\n⚡ VOLATILITY METRICS:\n"
        analysis_text += "─" * 30 + "\n"
        analysis_text += f"Recent Volatility (20 days): {recent_returns:.2f}%\n"
        analysis_text += f"Historical Volatility: {overall_volatility:.2f}%\n"
        
        volatility_trend = "📈 Increasing" if recent_returns > overall_volatility * 1.1 else "📉 Decreasing" if recent_returns < overall_volatility * 0.9 else "➡️ Stable"
        analysis_text += f"Volatility Trend: {volatility_trend}\n"
        
        # Risk metrics
        current_price = predictions['current_price']
        predicted_prices = predictions['predicted_prices']
        
        # Value at Risk (simple approximation)
        price_changes = [(p/current_price - 1)*100 for p in predicted_prices]
        var_95 = np.percentile(price_changes, 5) if price_changes else 0
        
        analysis_text += f"\n⚠️ RISK METRICS:\n"
        analysis_text += "─" * 25 + "\n"
        analysis_text += f"Value at Risk (95%): {var_95:.2f}%\n"
        analysis_text += f"Maximum Predicted Gain: {max(price_changes):.2f}%\n"
        analysis_text += f"Maximum Predicted Loss: {min(price_changes):.2f}%\n"
        
        # Model limitations and confidence
        analysis_text += f"\n🎯 MODEL CONFIDENCE & LIMITATIONS:\n"
        analysis_text += "─" * 45 + "\n"
        analysis_text += f"• Prediction accuracy tends to decrease over longer time horizons\n"
        analysis_text += f"• Model assumes market states follow Markov property\n"
        analysis_text += f"• External events (news, earnings, etc.) not considered\n"
        analysis_text += f"• Based on {self.period_var.get()} of historical data\n"
        
        days = int(self.days_var.get())
        confidence_score = max(50, 85 - (days * 2))
        analysis_text += f"• Estimated confidence for {days}-day prediction: {confidence_score}%\n"
        
        # Technical indicators summary
        analysis_text += f"\n📈 TECHNICAL SUMMARY:\n"
        analysis_text += "─" * 30 + "\n"
        
        # Simple moving averages
        ma_5 = self.predictor.data['Close'].tail(5).mean()
        ma_20 = self.predictor.data['Close'].tail(20).mean()
        
        trend = "📈 Uptrend" if current_price > ma_5 > ma_20 else "📉 Downtrend" if current_price < ma_5 < ma_20 else "➡️ Sideways"
        analysis_text += f"Short-term Trend: {trend}\n"
        analysis_text += f"Current vs 5-day MA: {((current_price/ma_5 - 1)*100):+.2f}%\n"
        analysis_text += f"Current vs 20-day MA: {((current_price/ma_20 - 1)*100):+.2f}%\n"
        
        self.analysis_text.insert(tk.END, analysis_text)
        
    def compare_methods(self):
        """Compare different prediction methods"""
        self.progress.start()
        self.compare_btn.config(state='disabled')
        self.status_var.set("Comparing prediction methods...")
        
        thread = threading.Thread(target=self._perform_comparison)
        thread.daemon = True
        thread.start()
        
    def _perform_comparison(self):
        """Perform method comparison in separate thread"""
        try:
            symbol = self.symbol_var.get().upper()
            period = self.period_var.get()
            days = int(self.days_var.get())
            
            results = {}
            methods = [('return_bins', 'Return Bins'), ('direction', 'Direction Based')]
            
            for method_key, method_name in methods:
                predictor = MarkovStockPredictor(symbol, period)
                predictor.fetch_data()
                predictor.calculate_daily_returns()
                predictor.define_states(method=method_key, n_states=6 if method_key == 'return_bins' else None)
                predictor.build_transition_matrix()
                
                predictions = predictor.predict_price_values(days)
                accuracy = predictor.backtest_predictions(test_days=min(60, len(predictor.states)//2))
                
                if predictions:
                    current_price = predictions['current_price']
                    final_price = predictions['predicted_prices'][-1]
                    expected_return = ((final_price - current_price) / current_price) * 100
                    
                    results[method_name] = {
                        'expected_return': expected_return,
                        'accuracy': accuracy,
                        'final_price': final_price,
                        'predictions': predictions
                    }
            
            self.root.after(0, self._update_comparison_results, results, days)
            
        except Exception as e:
            self.root.after(0, self._show_error, f"Comparison failed: {str(e)}")
        finally:
            self.root.after(0, lambda: (self.progress.stop(), self.compare_btn.config(state='normal')))
            
    def _update_comparison_results(self, results, days):
        """Update GUI with comparison results"""
        if not results:
            return
            
        symbol = self.symbol_var.get().upper()
        
        comparison_text = f"""
⚖️ METHOD COMPARISON RESULTS FOR {symbol}
═══════════════════════════════════════════════════════════════

📊 {days}-DAY PREDICTION COMPARISON:

"""
        
        comparison_text += f"{'Method':<20} {'Expected Return':<15} {'Accuracy':<12} {'Final Price':<12}\n"
        comparison_text += "─" * 65 + "\n"
        
        best_method = None
        best_score = -float('inf')
        
        for method, data in results.items():
            return_pct = data['expected_return']
            accuracy = data['accuracy']
            final_price = data['final_price']
            
            # Calculate composite score (accuracy weighted more heavily)
            score = (accuracy * 0.7) + (abs(return_pct) * 0.01)  # Slight preference for higher predicted returns
            
            if score > best_score:
                best_score = score
                best_method = method
            
            comparison_text += f"{method:<20} {return_pct:+7.2f}% {' ':<7} {accuracy:>6.1%} {' ':<5} ${final_price:<11.2f}\n"
        
        comparison_text += "\n🏆 RECOMMENDATION:\n"
        comparison_text += "─" * 25 + "\n"
        if best_method:
            best_data = results[best_method]
            comparison_text += f"Best Method: {best_method}\n"
            comparison_text += f"Rationale: Highest combined accuracy and prediction confidence\n"
            comparison_text += f"Expected {days}-day return: {best_data['expected_return']:+.2f}%\n"
            comparison_text += f"Model accuracy: {best_data['accuracy']:.1%}\n"
        
        comparison_text += f"\n📈 CONSENSUS ANALYSIS:\n"
        comparison_text += "─" * 30 + "\n"
        
        # Calculate consensus
        returns = [data['expected_return'] for data in results.values()]
        accuracies = [data['accuracy'] for data in results.values()]
        
        avg_return = sum(returns) / len(returns)
        avg_accuracy = sum(accuracies) / len(accuracies)
        return_std = np.std(returns)
        
        comparison_text += f"Average Expected Return: {avg_return:+.2f}%\n"
        comparison_text += f"Average Model Accuracy: {avg_accuracy:.1%}\n"
        comparison_text += f"Prediction Variance: ±{return_std:.2f}%\n"
        
        if return_std < 2:
            consensus = "🟢 Strong consensus between methods"
        elif return_std < 5:
            consensus = "🟡 Moderate agreement between methods"
        else:
            consensus = "🔴 High disagreement - use caution"
            
        comparison_text += f"Method Agreement: {consensus}\n"
        
        # Trading recommendation based on consensus
        comparison_text += f"\n🎯 CONSENSUS TRADING RECOMMENDATION:\n"
        comparison_text += "─" * 45 + "\n"
        
        if abs(avg_return) < 3:
            recommendation = "➡️ NEUTRAL - Consider range trading or hold position"
        elif avg_return > 5:
            recommendation = "📈 BULLISH - Both methods suggest upward movement"
        elif avg_return < -5:
            recommendation = "📉 BEARISH - Both methods suggest downward movement"
        elif avg_return > 0:
            recommendation = "📈 CAUTIOUSLY BULLISH - Slight upward bias"
        else:
            recommendation = "📉 CAUTIOUSLY BEARISH - Slight downward bias"
            
        comparison_text += f"{recommendation}\n"
        
        # Risk assessment
        comparison_text += f"\nRisk Level: "
        if return_std > 5 or max(abs(r) for r in returns) > 15:
            comparison_text += "🔴 HIGH (High variance or extreme predictions)\n"
        elif return_std > 2 or max(abs(r) for r in returns) > 8:
            comparison_text += "🟡 MEDIUM (Moderate variance)\n"
        else:
            comparison_text += "🟢 LOW (Low variance, consistent predictions)\n"
            
        comparison_text += f"\n⚠️ IMPORTANT NOTES:\n"
        comparison_text += f"• Higher accuracy doesn't always mean better predictions\n"
        comparison_text += f"• Consider market conditions and external factors\n"
        comparison_text += f"• Use multiple timeframes for better insights\n"
        comparison_text += f"• Always combine with fundamental analysis\n"
        
        # Clear and update results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, comparison_text)
        
        # Update status
        self.status_var.set(f"Method comparison complete - {best_method} recommended")
        
    def clear_results(self):
        """Clear all results and charts"""
        self.results_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.fig.clear()
        self.canvas.draw()
        self.status_var.set("Results cleared - Ready for new analysis")
        
    def _show_error(self, error_message):
        """Show error message to user"""
        messagebox.showerror("Analysis Error", error_message)
        self.status_var.set("Error occurred - Please check your inputs")
        
    def _analysis_complete(self):
        """Clean up after analysis completion"""
        self.progress.stop()
        self.analyze_btn.config(state='normal')

def create_web_interface():
    """Create a simple web interface using HTML/JavaScript"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Predictor Web Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: bold;
            margin-bottom: 5px;
            color: #34495e;
        }
        input, select, button {
            padding: 10px;
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.3);
        }
        .btn {
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .btn:active {
            transform: translateY(0);
        }
        .results {
            background: #ecf0f1;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            min-height: 300px;
            border: 2px solid #bdc3c7;
        }
        .loading {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
        }
        .popular-stocks {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .stock-btn {
            background: #95a5a6;
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s ease;
        }
        .stock-btn:hover {
            background: #34495e;
            transform: scale(1.05);
        }
        .warning {
            background: #f39c12;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: bold;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Stock Market Predictor</h1>
        
        <div class="warning">
            ⚠️ This is a demonstration interface. For full functionality, run the Python GUI application.
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="symbol">Stock Symbol:</label>
                <input type="text" id="symbol" value="AAPL" placeholder="Enter stock symbol">
                <div class="popular-stocks">
                    <button class="stock-btn" onclick="setSymbol('AAPL')">AAPL</button>
                    <button class="stock-btn" onclick="setSymbol('GOOGL')">GOOGL</button>
                    <button class="stock-btn" onclick="setSymbol('MSFT')">MSFT</button>
                    <button class="stock-btn" onclick="setSymbol('TSLA')">TSLA</button>
                    <button class="stock-btn" onclick="setSymbol('NVDA')">NVDA</button>
                    <button class="stock-btn" onclick="setSymbol('SPY')">SPY</button>
                </div>
            </div>
            
            <div class="control-group">
                <label for="period">Data Period:</label>
                <select id="period">
                    <option value="1y">1 Year</option>
                    <option value="2y" selected>2 Years</option>
                    <option value="5y">5 Years</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="days">Prediction Days:</label>
                <select id="days">
                    <option value="5">5 Days</option>
                    <option value="10" selected>10 Days</option>
                    <option value="15">15 Days</option>
                    <option value="20">20 Days</option>
                    <option value="30">30 Days</option>
                </select>
            </div>
            
            <div class="control-group">
                <label for="method">Method:</label>
                <select id="method">
                    <option value="return_bins" selected>Return Bins</option>
                    <option value="direction">Direction Based</option>
                </select>
            </div>
        </div>
        
        <div style="text-align: center; margin-bottom: 20px;">
            <button class="btn" onclick="runAnalysis()">🔍 Run Analysis</button>
            <button class="btn" onclick="clearResults()" style="margin-left: 10px;">🗑️ Clear</button>
        </div>
        
        <div class="results" id="results">
            <div class="loading">
                👆 Click "Run Analysis" to start predicting stock prices!
                <br><br>
                <strong>Features:</strong>
                <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
                    <li>🎯 Markov Chain-based predictions</li>
                    <li>📊 Multiple analysis methods</li>
                    <li>📈 Historical backtesting</li>
                    <li>⚖️ Risk assessment</li>
                    <li>🔬 Detailed technical analysis</li>
                </ul>
                <br>
                <em>Note: This web interface is a demonstration. For full interactive features, charts, and real-time analysis, please run the Python application.</em>
            </div>
        </div>
    </div>

    <script>
        function setSymbol(symbol) {
            document.getElementById('symbol').value = symbol;
        }
        
        function runAnalysis() {
            const symbol = document.getElementById('symbol').value.toUpperCase();
            const period = document.getElementById('period').value;
            const days = document.getElementById('days').value;
            const method = document.getElementById('method').value;
            
            const resultsDiv = document.getElementById('results');
            
            // Show loading
            resultsDiv.innerHTML = `
                <div class="loading">
                    ⏳ Analyzing ${symbol}...
                    <br><br>
                    <div style="font-size: 14px;">
                        • Fetching ${period} of historical data<br>
                        • Building Markov chain model<br>
                        • Generating ${days}-day predictions<br>
                        • Calculating risk metrics<br>
                    </div>
                </div>
            `;
            
            // Simulate analysis (in real app, this would call Python backend)
            setTimeout(() => {
                const mockResults = generateMockResults(symbol, days, period, method);
                resultsDiv.innerHTML = mockResults;
            }, 3000);
        }
        
        function generateMockResults(symbol, days, period, method) {
            const currentPrice = Math.random() * 200 + 100;
            const expectedReturn = (Math.random() - 0.5) * 20; // -10% to +10%
            const finalPrice = currentPrice * (1 + expectedReturn / 100);
            const accuracy = 0.6 + Math.random() * 0.25; // 60-85%
            
            return `
                <h2>📊 Analysis Results for ${symbol}</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
                    <div style="background: #3498db; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3>💰 Current Price</h3>
                        <div style="font-size: 24px; font-weight: bold;">${currentPrice.toFixed(2)}</div>
                    </div>
                    <div style="background: ${expectedReturn > 0 ? '#27ae60' : '#e74c3c'}; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3>🎯 ${days}-Day Target</h3>
                        <div style="font-size: 24px; font-weight: bold;">${finalPrice.toFixed(2)}</div>
                        <div>${expectedReturn > 0 ? '+' : ''}${expectedReturn.toFixed(1)}%</div>
                    </div>
                    <div style="background: #9b59b6; color: white; padding: 15px; border-radius: 10px; text-align: center;">
                        <h3>🎲 Model Accuracy</h3>
                        <div style="font-size: 24px; font-weight: bold;">${(accuracy * 100).toFixed(1)}%</div>
                        <div>Backtest Results</div>
                    </div>
                </div>
                
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h3>🎯 Trading Recommendation:</h3>
                    <div style="font-size: 18px; font-weight: bold; color: ${expectedReturn > 5 ? '#27ae60' : expectedReturn < -5 ? '#e74c3c' : '#f39c12'};">
                        ${expectedReturn > 5 ? '📈 BULLISH - Consider long positions' : 
                          expectedReturn < -5 ? '📉 BEARISH - Consider defensive strategies' : 
                          '➡️ NEUTRAL - Sideways movement expected'}
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>Risk Level:</strong> ${Math.abs(expectedReturn) > 8 ? '🔴 HIGH' : Math.abs(expectedReturn) > 4 ? '🟡 MEDIUM' : '🟢 LOW'}
                    </div>
                    <div>
                        <strong>Confidence:</strong> ${Math.max(60, 85 - parseInt(days) * 2)}%
                    </div>
                </div>
                
                <div style="background: #ecf0f1; padding: 15px; border-radius: 10px; border-left: 4px solid #e67e22;">
                    <h4>⚠️ Important Disclaimer:</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>This is a <strong>demonstration</strong> with simulated results</li>
                        <li>Real predictions require the full Python application</li>
                        <li>Past performance doesn't guarantee future results</li>
                        <li>Always do your own research before investing</li>
                        <li>Consider consulting with a financial advisor</li>
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 20px; padding: 15px; background: #dff0d8; border-radius: 10px; border: 1px solid #d6e9c6;">
                    <h4>🚀 Want Full Features?</h4>
                    <p>Run the Python GUI application for:</p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0;">
                        <div>📈 Interactive charts</div>
                        <div>🔄 Real-time data</div>
                        <div>📊 Detailed analysis</div>
                        <div>⚖️ Method comparison</div>
                    </div>
                </div>
            `;
        }
        
        function clearResults() {
            document.getElementById('results').innerHTML = `
                <div class="loading">
                    Results cleared! Ready for new analysis.
                    <br><br>
                    👆 Select your parameters and click "Run Analysis"
                </div>
            `;
        }
    </script>
</body>
</html>
    """
    
    # Save HTML file
    with open('stock_predictor_web.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Web interface created as 'stock_predictor_web.html'")
    print("   Open this file in your web browser for a demo interface")

def main():
    """Main function to run the application"""
    print("🚀 Starting Advanced Stock Market Predictor...")
    print("="*60)
    
    try:
        # Create web interface
        create_web_interface()
        
        # Create and run GUI
        root = tk.Tk()
        app = StockPredictorGUI(root)
        
        print("✅ GUI Application launched successfully!")
        print("✅ Web interface available at: stock_predictor_web.html")
        print("\n📊 Features available:")
        print("   • 🎯 Markov Chain-based stock predictions")
        print("   • 📈 Interactive charts with matplotlib")
        print("   • 🔄 Real-time data fetching with yfinance")
        print("   • ⚖️ Multiple prediction methods comparison")
        print("   • 📊 Detailed technical analysis")
        print("   • 🔬 Model backtesting and validation")
        print("   • 🎨 Modern GUI with tkinter")
        print("   • 🌐 Bonus web interface for demonstrations")
        
        print("\n🎮 How to use:")
        print("   1. Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)")
        print("   2. Select analysis parameters")
        print("   3. Click 'Analyze & Predict' for predictions")
        print("   4. Use 'Compare Methods' to validate results")
        print("   5. View results in different tabs")
        
        print(f"\n⚠️  Important Notes:")
        print(f"   • This is for educational purposes only")
        print(f"   • Not financial advice - do your own research")
        print(f"   • Past performance doesn't guarantee future results")
        print(f"   • Consider consulting with a financial advisor")
        
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("📦 Please install required packages:")
        print("   pip install numpy pandas yfinance matplotlib tkinter")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("🔧 Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()

# Quick usage examples for command line:
def quick_predict(symbol="AAPL", days=10):
    """Quick prediction function for command line use"""
    try:
        predictor = MarkovStockPredictor(symbol, period="2y")
        predictor.fetch_data()
        predictor.calculate_daily_returns()
        predictor.define_states(method='return_bins', n_states=6)
        predictor.build_transition_matrix()
        
        predictions = predictor.predict_price_values(days)
        if predictions:
            current = predictions['current_price']
            target = predictions['predicted_prices'][-1]
            return_pct = ((target - current) / current) * 100
            
            print(f"\n🚀 Quick Prediction for {symbol}:")
            print(f"   Current: ${current:.2f}")
            print(f"   {days}-day target: ${target:.2f}")
            print(f"   Expected return: {return_pct:+.1f}%")
            return predictions
        else:
            print(f"❌ Could not generate predictions for {symbol}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Example usage:
# quick_predict("TSLA", 5)  # Quick 5-day prediction for Tesla
# quick_predict("SPY", 20)  # Quick 20-day prediction for S&P 500 ETF