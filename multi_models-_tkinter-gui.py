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

# Neural Network imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
            'model_name': 'Markov Chain',
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

class NeuralNetworkPredictor:
    def __init__(self, symbol, period="2y"):
        """Initialize the Neural Network Stock Predictor"""
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.model_type = 'LSTM'
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if len(self.data) < 60:
                raise ValueError(f"Insufficient data for {self.symbol}")
            return True
        except Exception as e:
            raise Exception(f"Error fetching data for {self.symbol}: {str(e)}")
    
    def prepare_features(self, lookback_days=60):
        """Prepare features for neural network training"""
        # Technical indicators
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        self.data['MACD'] = self.calculate_macd(self.data['Close'])
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Price_Change'] = self.data['Close'].pct_change()
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        # Select features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volatility', 
                          'Volume_MA', 'Price_Change']
        
        features = self.data[feature_columns].values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback_days, len(features_scaled)):
            X.append(features_scaled[i-lookback_days:i])
            y.append(features_scaled[i, 3])  # Close price index
            
        return np.array(X), np.array(y)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def build_lstm_model(self, input_shape):
        """Build LSTM neural network model"""
        model = keras.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(100, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(50),
            layers.Dropout(0.3),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_gru_model(self, input_shape):
        """Build GRU neural network model"""
        model = keras.Sequential([
            layers.GRU(100, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.GRU(100, return_sequences=True),
            layers.Dropout(0.3),
            layers.GRU(50),
            layers.Dropout(0.3),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """Build CNN-LSTM hybrid model"""
        model = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.LSTM(100, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(50),
            layers.Dropout(0.3),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, model_type='LSTM', epochs=50):
        """Train the neural network model"""
        self.model_type = model_type
        X, y = self.prepare_features()
        
        # Split data
        split_ratio = 0.8
        split_index = int(len(X) * split_ratio)
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Build model based on type
        if model_type == 'LSTM':
            self.model = self.build_lstm_model((X.shape[1], X.shape[2]))
        elif model_type == 'GRU':
            self.model = self.build_gru_model((X.shape[1], X.shape[2]))
        elif model_type == 'CNN-LSTM':
            self.model = self.build_cnn_lstm_model((X.shape[1], X.shape[2]))
        
        # Train model
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate accuracy metrics
        y_pred = self.model.predict(X_test, verbose=0)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'history': history
        }
    
    def predict_price_values(self, days=10):
        """Predict future prices using the trained model"""
        if self.model is None:
            return None
        
        # Get the last 60 days of data for prediction
        lookback_days = 60
        last_sequence = self.data.tail(lookback_days)
        
        # Prepare features for the last sequence
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volatility', 
                          'Volume_MA', 'Price_Change']
        
        current_sequence = last_sequence[feature_columns].values
        current_sequence = self.scaler.transform(current_sequence)
        current_sequence = current_sequence.reshape(1, lookback_days, len(feature_columns))
        
        predicted_prices = []
        current_price = self.data['Close'].iloc[-1]
        
        # Generate predictions
        for _ in range(days):
            # Predict next normalized price
            next_pred_norm = self.model.predict(current_sequence, verbose=0)[0, 0]
            
            # Create a dummy array to denormalize
            dummy_array = np.zeros((1, len(feature_columns)))
            dummy_array[0, 3] = next_pred_norm  # Close price is at index 3
            
            # Denormalize
            denormalized = self.scaler.inverse_transform(dummy_array)
            next_price = denormalized[0, 3]
            
            predicted_prices.append(next_price)
            
            # Update sequence for next prediction
            # This is a simplified approach - in practice, you'd want to update all features
            new_row = current_sequence[0, -1:].copy()
            new_row[0, 3] = next_pred_norm  # Update close price
            
            current_sequence = np.concatenate([current_sequence[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)
        
        # Calculate confidence intervals (simplified)
        volatility = self.data['Close'].pct_change().std()
        lower_bounds = [p * (1 - volatility * np.sqrt(i+1)) for i, p in enumerate(predicted_prices)]
        upper_bounds = [p * (1 + volatility * np.sqrt(i+1)) for i, p in enumerate(predicted_prices)]
        
        return {
            'model_name': f'Neural Network ({self.model_type})',
            'current_price': current_price,
            'predicted_prices': predicted_prices,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds
        }

class MLPredictor:
    def __init__(self, symbol, period="2y"):
        """Initialize the Machine Learning Stock Predictor"""
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.scaler = StandardScaler()
        self.models = {}
        
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if len(self.data) < 100:
                raise ValueError(f"Insufficient data for {self.symbol}")
            return True
        except Exception as e:
            raise Exception(f"Error fetching data for {self.symbol}: {str(e)}")
    
    def prepare_features(self):
        """Prepare features for ML models"""
        # Technical indicators
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        self.data['MACD'] = self.calculate_macd(self.data['Close'])
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Price_Change'] = self.data['Close'].pct_change()
        self.data['High_Low_Ratio'] = self.data['High'] / self.data['Low']
        self.data['Open_Close_Ratio'] = self.data['Open'] / self.data['Close']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            self.data[f'Close_Lag_{lag}'] = self.data['Close'].shift(lag)
            self.data[f'Volume_Lag_{lag}'] = self.data['Volume'].shift(lag)
        
        # Target variable (next day's closing price)
        self.data['Target'] = self.data['Close'].shift(-1)
        
        # Drop NaN values
        self.data = self.data.dropna()
        
        return self.data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def train_models(self):
        """Train multiple ML models"""
        data = self.prepare_features()
        
        # Feature selection
        feature_columns = [col for col in data.columns if col not in ['Target', 'Close']]
        X = data[feature_columns].values
        y = data['Target'].values
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        split_ratio = 0.8
        split_index = int(len(X_scaled) * split_ratio)
        
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Train different models
        models_config = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models_config.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            self.models[name] = model
            results[name] = {
                'mae': mae,
                'mse': mse,
                'model': model
            }
        
        return results
    
    def predict_with_ensemble(self, days=10):
        """Predict using ensemble of ML models"""
        if not self.models:
            return None
        
        # Get latest features
        data = self.prepare_features()
        feature_columns = [col for col in data.columns if col not in ['Target', 'Close']]
        
        predictions_by_model = {}
        current_price = self.data['Close'].iloc[-1]
        
        for model_name, model in self.models.items():
            predictions = []
            
            # Use the latest data point for prediction
            latest_features = data[feature_columns].iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            for day in range(days):
                pred = model.predict(latest_features_scaled)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified)
                # In practice, you'd want to properly update all features
                latest_features_scaled[0, -1] = pred  # Update the most recent close price feature
            
            predictions_by_model[model_name] = predictions
        
        # Ensemble prediction (average of all models)
        ensemble_predictions = []
        for day in range(days):
            day_predictions = [predictions_by_model[model][day] for model in self.models.keys()]
            ensemble_pred = np.mean(day_predictions)
            ensemble_predictions.append(ensemble_pred)
        
        # Calculate confidence intervals
        prediction_std = np.std([predictions_by_model[model] for model in self.models.keys()], axis=0)
        lower_bounds = [pred - 1.96 * std for pred, std in zip(ensemble_predictions, prediction_std)]
        upper_bounds = [pred + 1.96 * std for pred, std in zip(ensemble_predictions, prediction_std)]
        
        return {
            'model_name': 'ML Ensemble',
            'current_price': current_price,
            'predicted_prices': ensemble_predictions,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'individual_predictions': predictions_by_model
        }

class MultiModelStockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Claude AI generated Multi-Model Stock Predictor - implemented by Lee Almasy")
        self.root.geometry("1600x1000")
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
        
        self.predictors = {}
        self.create_widgets()
        
    def create_widgets(self):
        """Create and layout all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚀 Multi-Model Stock Market Predictor", 
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
        
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ", "TSM"]
        for stock in popular_stocks:
            btn = ttk.Button(popular_frame, text=stock, width=6,
                           command=lambda s=stock: self.symbol_var.set(s))
            btn.pack(side=tk.LEFT, padx=2)
        
        # Model Selection
        model_frame = ttk.Frame(control_frame, style='Custom.TFrame')
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Select Models:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        
        self.model_vars = {
            'Markov Chain': tk.BooleanVar(value=True),
            'LSTM Neural Network': tk.BooleanVar(value=True),
            'GRU Neural Network': tk.BooleanVar(value=False),
            'CNN-LSTM Hybrid': tk.BooleanVar(value=False),
            'ML Ensemble': tk.BooleanVar(value=True)
        }
        
        for model_name, var in self.model_vars.items():
            cb = ttk.Checkbutton(model_frame, text=model_name, variable=var)
            cb.pack(side=tk.LEFT, padx=5)
        
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
                                 values=["3", "5", "10", "15", "20", "30"], width=10, state="readonly")
        days_combo.pack(side=tk.LEFT, padx=5)
        
        # Training epochs for neural networks
        ttk.Label(param_frame, text="NN Epochs:", style='Header.TLabel').pack(side=tk.LEFT, padx=15)
        self.epochs_var = tk.StringVar(value="50")
        epochs_combo = ttk.Combobox(param_frame, textvariable=self.epochs_var,
                                   values=["20", "50", "100"], width=10, state="readonly")
        epochs_combo.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame, style='Custom.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="🔍 Analyze & Predict", 
                                     command=self.run_analysis, style='Custom.TButton')
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.compare_btn = ttk.Button(button_frame, text="⚖️ Compare All Models", 
                                     command=self.compare_all_models, style='Custom.TButton')
        self.compare_btn.pack(side=tk.LEFT, padx=5)
        
        self.ensemble_btn = ttk.Button(button_frame, text="🎯 Ensemble Prediction", 
                                      command=self.ensemble_prediction, style='Custom.TButton')
        self.ensemble_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="🗑️ Clear Results", 
                                   command=self.clear_results, style='Custom.TButton')
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar and status
        progress_frame = ttk.Frame(button_frame, style='Custom.TFrame')
        progress_frame.pack(side=tk.RIGHT, padx=5)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(side=tk.TOP)
        
        self.training_status = ttk.Label(progress_frame, text="", style='Info.TLabel')
        self.training_status.pack(side=tk.BOTTOM)
        
        # Content area with notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.results_frame, text="📈 Prediction Results")
        
        # Chart tab
        self.chart_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.chart_frame, text="📊 Interactive Chart")
        
        # Model Comparison tab
        self.comparison_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.comparison_frame, text="⚖️ Model Comparison")
        
        # Technical Analysis tab
        self.analysis_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.analysis_frame, text="🔬 Technical Analysis")
        
        # Model Performance tab
        self.performance_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(self.performance_frame, text="📊 Model Performance")
        
        # Setup all tabs
        self.setup_results_tab()
        self.setup_chart_tab()
        self.setup_comparison_tab()
        self.setup_analysis_tab()
        self.setup_performance_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to analyze stock predictions with multiple models")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              style='Info.TLabel', relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
    def setup_results_tab(self):
        """Setup the results display tab"""
        self.results_text = scrolledtext.ScrolledText(self.results_frame, 
                                                     font=('Consolas', 10),
                                                     bg='#ecf0f1', fg='#2c3e50',
                                                     height=35)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_chart_tab(self):
        """Setup the chart display tab"""
        self.fig = Figure(figsize=(14, 10), facecolor='#ecf0f1')
        self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Chart toolbar
        toolbar_frame = ttk.Frame(self.chart_frame, style='Custom.TFrame')
        toolbar_frame.pack(fill=tk.X, padx=10)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def setup_comparison_tab(self):
        """Setup the model comparison tab"""
        self.comparison_text = scrolledtext.ScrolledText(self.comparison_frame,
                                                        font=('Consolas', 9),
                                                        bg='#ecf0f1', fg='#2c3e50',
                                                        height=35)
        self.comparison_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_analysis_tab(self):
        """Setup the technical analysis tab"""
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame,
                                                      font=('Consolas', 9),
                                                      bg='#ecf0f1', fg='#2c3e50',
                                                      height=35)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def setup_performance_tab(self):
        """Setup the model performance tab"""
        self.performance_text = scrolledtext.ScrolledText(self.performance_frame,
                                                         font=('Consolas', 9),
                                                         bg='#ecf0f1', fg='#2c3e50',
                                                         height=35)
        self.performance_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def run_analysis(self):
        """Run the multi-model stock prediction analysis"""
        self.progress.start()
        self.analyze_btn.config(state='disabled')
        self.status_var.set("Training models and analyzing...")
        
        thread = threading.Thread(target=self._perform_analysis)
        thread.daemon = True
        thread.start()
        
    def _perform_analysis(self):
        """Perform the actual analysis (runs in separate thread)"""
        try:
            symbol = self.symbol_var.get().upper()
            period = self.period_var.get()
            days = int(self.days_var.get())
            epochs = int(self.epochs_var.get())
            
            results = {}
            selected_models = [name for name, var in self.model_vars.items() if var.get()]
            
            if not selected_models:
                raise Exception("Please select at least one model")
            
            # Train and predict with each selected model
            for i, model_name in enumerate(selected_models):
                self.root.after(0, lambda m=model_name: self.training_status.config(text=f"Training {m}..."))
                
                try:
                    if model_name == 'Markov Chain':
                        predictor = MarkovStockPredictor(symbol, period)
                        predictor.fetch_data()
                        predictor.calculate_daily_returns()
                        predictor.define_states(method='return_bins', n_states=6)
                        predictor.build_transition_matrix()
                        
                        predictions = predictor.predict_price_values(days)
                        accuracy = predictor.backtest_predictions(test_days=min(60, len(predictor.states)//2))
                        
                        if predictions:
                            predictions['accuracy'] = accuracy
                            results[model_name] = predictions
                            self.predictors[model_name] = predictor
                    
                    elif 'Neural Network' in model_name:
                        nn_type = model_name.split()[0]  # LSTM, GRU, etc.
                        predictor = NeuralNetworkPredictor(symbol, period)
                        predictor.fetch_data()
                        
                        training_results = predictor.train_model(model_type=nn_type, epochs=epochs)
                        predictions = predictor.predict_price_values(days)
                        
                        if predictions:
                            predictions['accuracy'] = 1 - training_results['mae']  # Simplified accuracy
                            predictions['training_mae'] = training_results['mae']
                            predictions['training_mse'] = training_results['mse']
                            results[model_name] = predictions
                            self.predictors[model_name] = predictor
                    
                    elif model_name == 'ML Ensemble':
                        predictor = MLPredictor(symbol, period)
                        predictor.fetch_data()
                        
                        training_results = predictor.train_models()
                        predictions = predictor.predict_with_ensemble(days)
                        
                        if predictions:
                            # Calculate average accuracy from all ML models
                            avg_mae = np.mean([r['mae'] for r in training_results.values()])
                            predictions['accuracy'] = max(0, 1 - avg_mae / 100)  # Simplified
                            predictions['ml_results'] = training_results
                            results[model_name] = predictions
                            self.predictors[model_name] = predictor
                
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    continue
            
            if not results:
                raise Exception("No models were successfully trained")
            
            # Update GUI in main thread
            self.root.after(0, self._update_analysis_results, results, days)
            
        except Exception as e:
            self.root.after(0, self._show_error, str(e))
        finally:
            self.root.after(0, self._analysis_complete)
            
    def _update_analysis_results(self, results, days):
        """Update the results display with multi-model analysis"""
        symbol = self.symbol_var.get().upper()
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Generate comprehensive results
        results_text = f"""
🚀 MULTI-MODEL STOCK PREDICTION ANALYSIS
═══════════════════════════════════════════════════════════════

📊 STOCK: {symbol}
📅 ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📈 DATA PERIOD: {self.period_var.get()}
🔮 PREDICTION HORIZON: {days} days
🤖 MODELS ANALYZED: {len(results)}

💰 CURRENT PRICE: ${results[list(results.keys())[0]]['current_price']:.2f}

🎯 MODEL PREDICTIONS SUMMARY:
"""
        
        results_text += "─" * 90 + "\n"
        results_text += f"{'Model':<25} {'Final Price':<12} {'Return %':<10} {'Accuracy':<10} {'Confidence':<12}\n"
        results_text += "─" * 90 + "\n"
        
        current_price = results[list(results.keys())[0]]['current_price']
        model_predictions = {}
        
        for model_name, data in results.items():
            final_price = data['predicted_prices'][-1]
            return_pct = ((final_price - current_price) / current_price) * 100
            accuracy = data.get('accuracy', 0)
            
            # Calculate confidence based on model type and accuracy
            if 'Neural Network' in model_name:
                confidence = min(95, 60 + accuracy * 30)
            elif model_name == 'Markov Chain':
                confidence = min(90, 50 + accuracy * 40)
            else:  # ML Ensemble
                confidence = min(92, 55 + accuracy * 35)
            
            confidence_level = "🟢 High" if confidence > 80 else "🟡 Med" if confidence > 60 else "🔴 Low"
            
            results_text += f"{model_name:<25} ${final_price:<11.2f} {return_pct:+6.1f}% {' ':<3} {accuracy:<9.1%} {confidence_level:<12}\n"
            
            model_predictions[model_name] = {
                'final_price': final_price,
                'return_pct': return_pct,
                'accuracy': accuracy,
                'confidence': confidence,
                'predictions': data['predicted_prices']
            }
        
        # Ensemble analysis
        results_text += "\n" + "═" * 90 + "\n"
        results_text += "🎯 ENSEMBLE ANALYSIS:\n"
        results_text += "─" * 40 + "\n"
        
        all_returns = [data['return_pct'] for data in model_predictions.values()]
        all_final_prices = [data['final_price'] for data in model_predictions.values()]
        all_accuracies = [data['accuracy'] for data in model_predictions.values()]
        
        ensemble_return = np.mean(all_returns)
        ensemble_price = np.mean(all_final_prices)
        ensemble_accuracy = np.mean(all_accuracies)
        return_std = np.std(all_returns)
        
        results_text += f"Consensus Return: {ensemble_return:+.2f}%\n"
        results_text += f"Consensus Price: ${ensemble_price:.2f}\n"
        results_text += f"Average Accuracy: {ensemble_accuracy:.1%}\n"
        results_text += f"Prediction Variance: ±{return_std:.2f}%\n"
        
        # Agreement analysis
        if return_std < 2:
            agreement = "🟢 Strong Consensus"
        elif return_std < 5:
            agreement = "🟡 Moderate Agreement"
        else:
            agreement = "🔴 High Disagreement"
        
        results_text += f"Model Agreement: {agreement}\n"
        
        # Trading recommendation
        results_text += f"\n🎯 TRADING RECOMMENDATION:\n"
        results_text += "─" * 35 + "\n"
        
        if abs(ensemble_return) < 3:
            recommendation = "➡️ NEUTRAL - Range trading strategy recommended"
            risk_level = "🟢 LOW"
        elif ensemble_return > 8:
            recommendation = "📈 STRONG BUY - Multiple models show bullish signals"
            risk_level = "🟡 MEDIUM" if return_std < 3 else "🔴 HIGH"
        elif ensemble_return > 3:
            recommendation = "📈 BUY - Moderate upward consensus"
            risk_level = "🟢 LOW" if return_std < 3 else "🟡 MEDIUM"
        elif ensemble_return < -8:
            recommendation = "📉 STRONG SELL - Multiple models show bearish signals"
            risk_level = "🔴 HIGH"
        elif ensemble_return < -3:
            recommendation = "📉 SELL - Moderate downward consensus"
            risk_level = "🟡 MEDIUM"
        else:
            recommendation = "➡️ HOLD - Mixed signals, wait for clearer direction"
            risk_level = "🟡 MEDIUM"
        
        results_text += f"{recommendation}\n"
        results_text += f"Risk Level: {risk_level}\n"
        
        # Detailed day-by-day predictions
        results_text += f"\n📊 DETAILED {days}-DAY FORECAST:\n"
        results_text += "─" * 80 + "\n"
        
        future_dates = pd.bdate_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=days)
        
        # Show ensemble prediction for each day
        for day in range(days):
            date_str = future_dates[day].strftime('%m-%d')
            day_prices = [model_predictions[model]['predictions'][day] for model in model_predictions.keys()]
            avg_price = np.mean(day_prices)
            day_return = ((avg_price - current_price) / current_price) * 100
            
            results_text += f"Day {day+1:2d} ({date_str}): ${avg_price:7.2f} ({day_return:+5.1f}%)\n"
        
        # Risk metrics
        results_text += f"\n⚠️ RISK ANALYSIS:\n"
        results_text += "─" * 25 + "\n"
        
        max_return = max(all_returns)
        min_return = min(all_returns)
        
        results_text += f"Best case scenario: {max_return:+.1f}%\n"
        results_text += f"Worst case scenario: {min_return:+.1f}%\n"
        results_text += f"Risk-adjusted return: {ensemble_return/max(1, return_std):+.2f}\n"
        
        # Position sizing recommendation
        if return_std < 2 and abs(ensemble_return) > 5:
            position_size = "25-40%"
        elif return_std < 5 and abs(ensemble_return) > 3:
            position_size = "15-25%"
        else:
            position_size = "5-15%"
        
        results_text += f"Recommended position size: {position_size} of portfolio\n"
        
        results_text += f"\n⚠️ IMPORTANT DISCLAIMERS:\n"
        results_text += f"• These are AI-generated predictions for educational purposes only\n"
        results_text += f"• Past performance does not guarantee future results\n"
        results_text += f"• Multiple models reduce but don't eliminate prediction risk\n"
        results_text += f"• Always conduct your own research and consider professional advice\n"
        results_text += f"• Market conditions can change rapidly and invalidate predictions\n"
        
        self.results_text.insert(tk.END, results_text)
        
        # Update charts and other tabs
        self._update_multi_model_chart(results, days)
        self._update_performance_analysis(results)
        
        self.status_var.set(f"Multi-model analysis complete - Consensus: {ensemble_return:+.1f}% return")
        
    def _update_multi_model_chart(self, results, days):
        """Update chart with multiple model predictions"""
        self.fig.clear()
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        ax_main = self.fig.add_subplot(gs[0, :])
        ax_returns = self.fig.add_subplot(gs[1, 0])
        ax_accuracy = self.fig.add_subplot(gs[1, 1])
        
        # Get historical data from first available predictor
        predictor = list(self.predictors.values())[0]
        historical_window = min(60, len(predictor.data))
        historical_dates = predictor.data.index[-historical_window:]
        historical_prices = predictor.data['Close'][-historical_window:]
        
        current_price = historical_prices.iloc[-1]
        last_date = historical_dates[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        # Plot historical data
        ax_main.plot(historical_dates, historical_prices, 'b-', linewidth=2, 
                    label='Historical Prices', alpha=0.8)
        
        # Plot current price
        ax_main.plot(last_date, current_price, 'go', 
                    markersize=10, label='Current Price', zorder=5)
        
        # Plot predictions for each model
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, (model_name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            ax_main.plot(future_dates, data['predicted_prices'], '--', 
                        color=color, linewidth=2, label=f'{model_name}', alpha=0.8)
            
            # Add confidence intervals for first model
            if i == 0 and 'lower_bounds' in data and 'upper_bounds' in data:
                ax_main.fill_between(future_dates, data['lower_bounds'], data['upper_bounds'],
                                   alpha=0.1, color=color)
        
        # Calculate and plot ensemble average
        ensemble_predictions = []
        for day in range(days):
            day_predictions = [results[model]['predicted_prices'][day] for model in results.keys()]
            ensemble_predictions.append(np.mean(day_predictions))
        
        ax_main.plot(future_dates, ensemble_predictions, 'k-', 
                    linewidth=4, label='Ensemble Average', alpha=0.9, zorder=4)
        
        ax_main.set_title(f'{self.symbol_var.get().upper()} - Multi-Model {days}-Day Predictions', 
                         fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Price ($)')
        ax_main.legend(loc='upper left', fontsize=8)
        ax_main.grid(True, alpha=0.3)
        ax_main.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7)
        
        # Format dates
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_main.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot expected returns
        model_names = list(results.keys())
        returns = [((results[model]['predicted_prices'][-1] - current_price) / current_price) * 100 
                  for model in model_names]
        
        bars = ax_returns.bar(range(len(model_names)), returns, 
                             color=['green' if r > 0 else 'red' for r in returns], alpha=0.7)
        ax_returns.set_title('Expected Returns by Model', fontsize=10)
        ax_returns.set_ylabel('Return (%)')
        ax_returns.set_xticks(range(len(model_names)))
        ax_returns.set_xticklabels([name.replace(' Neural Network', '') for name in model_names], 
                                  rotation=45, fontsize=8)
        ax_returns.grid(True, alpha=0.3)
        ax_returns.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Plot model accuracies
        accuracies = [results[model].get('accuracy', 0) * 100 for model in model_names]
        ax_accuracy.bar(range(len(model_names)), accuracies, color='blue', alpha=0.7)
        ax_accuracy.set_title('Model Accuracy', fontsize=10)
        ax_accuracy.set_ylabel('Accuracy (%)')
        ax_accuracy.set_xticks(range(len(model_names)))
        ax_accuracy.set_xticklabels([name.replace(' Neural Network', '') for name in model_names], 
                                   rotation=45, fontsize=8)
        ax_accuracy.set_ylim(0, 100)
        ax_accuracy.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def _update_performance_analysis(self, results):
        """Update the performance analysis tab"""
        self.performance_text.delete(1.0, tk.END)
        
        performance_text = f"""
📊 MODEL PERFORMANCE ANALYSIS
═══════════════════════════════════════════════════════════════

🎯 INDIVIDUAL MODEL PERFORMANCE:
"""
        
        for model_name, data in results.items():
            performance_text += f"\n🤖 {model_name.upper()}:\n"
            performance_text += "─" * 50 + "\n"
            
            current_price = data['current_price']
            final_price = data['predicted_prices'][-1]
            return_pct = ((final_price - current_price) / current_price) * 100
            accuracy = data.get('accuracy', 0)
            
            performance_text += f"Expected Return: {return_pct:+.2f}%\n"
            performance_text += f"Model Accuracy: {accuracy:.1%}\n"
            
            # Model-specific metrics
            if 'training_mae' in data:
                performance_text += f"Training MAE: {data['training_mae']:.4f}\n"
                performance_text += f"Training MSE: {data['training_mse']:.4f}\n"
            
            if 'ml_results' in data:
                performance_text += "ML Component Results:\n"
                for ml_model, metrics in data['ml_results'].items():
                    performance_text += f"  • {ml_model}: MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}\n"
            
            # Prediction stability
            price_changes = [abs(data['predicted_prices'][i] - data['predicted_prices'][i-1]) 
                           for i in range(1, len(data['predicted_prices']))]
            avg_volatility = np.mean(price_changes) / current_price * 100
            
            performance_text += f"Prediction Volatility: {avg_volatility:.2f}%\n"
            
            # Model characteristics
            if model_name == 'Markov Chain':
                performance_text += "Characteristics: State-based, captures regime changes\n"
                performance_text += "Best for: Short-term patterns, market regimes\n"
            elif 'Neural Network' in model_name:
                performance_text += "Characteristics: Deep learning, complex patterns\n"
                performance_text += "Best for: Non-linear relationships, long sequences\n"
            elif model_name == 'ML Ensemble':
                performance_text += "Characteristics: Multiple algorithms, robust\n"
                performance_text += "Best for: Reducing overfitting, stable predictions\n"
        
        # Comparative analysis
        performance_text += f"\n🔬 COMPARATIVE ANALYSIS:\n"
        performance_text += "─" * 40 + "\n"
        
        returns = [((results[model]['predicted_prices'][-1] - results[model]['current_price']) / 
                   results[model]['current_price']) * 100 for model in results.keys()]
        accuracies = [results[model].get('accuracy', 0) for model in results.keys()]
        
        best_return_idx = np.argmax(np.abs(returns))
        best_accuracy_idx = np.argmax(accuracies)
        most_conservative_idx = np.argmin(np.abs(returns))
        
        model_names = list(results.keys())
        
        performance_text += f"Most Aggressive Prediction: {model_names[best_return_idx]} ({returns[best_return_idx]:+.1f}%)\n"
        performance_text += f"Highest Accuracy: {model_names[best_accuracy_idx]} ({accuracies[best_accuracy_idx]:.1%})\n"
        performance_text += f"Most Conservative: {model_names[most_conservative_idx]} ({returns[most_conservative_idx]:+.1f}%)\n"
        
        # Correlation analysis
        performance_text += f"\n📈 PREDICTION CORRELATION:\n"
        performance_text += "─" * 35 + "\n"
        
        if len(results) > 1:
            predictions_matrix = np.array([results[model]['predicted_prices'] for model in results.keys()])
            correlation_matrix = np.corrcoef(predictions_matrix)
            
            performance_text += "Model Correlation Matrix:\n"
            for i, model1 in enumerate(results.keys()):
                for j, model2 in enumerate(results.keys()):
                    if i < j:
                        corr = correlation_matrix[i, j]
                        performance_text += f"  {model1} vs {model2}: {corr:.3f}\n"
            
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            performance_text += f"\nAverage Correlation: {avg_correlation:.3f}\n"
            
            if avg_correlation > 0.8:
                performance_text += "Interpretation: High agreement between models\n"
            elif avg_correlation > 0.5:
                performance_text += "Interpretation: Moderate agreement between models\n"
            else:
                performance_text += "Interpretation: Low agreement - diverse model opinions\n"
        
        # Risk-adjusted performance
        performance_text += f"\n⚠️ RISK-ADJUSTED METRICS:\n"
        performance_text += "─" * 35 + "\n"
        
        return_std = np.std(returns)
        mean_return = np.mean(returns)
        
        if return_std > 0:
            sharpe_ratio = mean_return / return_std
            performance_text += f"Ensemble Sharpe Ratio: {sharpe_ratio:.2f}\n"
        
        performance_text += f"Return Volatility: ±{return_std:.2f}%\n"
        performance_text += f"Prediction Spread: {max(returns) - min(returns):.2f}%\n"
        
        # Model recommendations
        performance_text += f"\n🎯 MODEL SELECTION RECOMMENDATIONS:\n"
        performance_text += "─" * 45 + "\n"
        
        if return_std < 2:
            performance_text += "• All models show strong consensus - high confidence\n"
            performance_text += "• Recommended: Use ensemble average for final decision\n"
        elif return_std < 5:
            performance_text += "• Models show moderate agreement - medium confidence\n"
            performance_text += "• Recommended: Weight models by accuracy\n"
        else:
            performance_text += "• High model disagreement - proceed with caution\n"
            performance_text += "• Recommended: Use most conservative estimate\n"
        
        performance_text += f"\n📚 MODEL STRENGTHS & WEAKNESSES:\n"
        performance_text += "─" * 45 + "\n"
        performance_text += """
MARKOV CHAIN:
+ Captures market regime changes effectively
+ Computationally efficient and interpretable
+ Good for short-term state transitions
- Limited feature incorporation
- Assumes Markovian property

NEURAL NETWORKS (LSTM/GRU/CNN-LSTM):
+ Can learn complex non-linear patterns
+ Handles sequential dependencies well
+ Incorporates multiple technical indicators
- Requires large amounts of data
- Prone to overfitting
- Less interpretable

ML ENSEMBLE:
+ Combines multiple algorithms for robustness
+ Reduces overfitting through averaging
+ Good generalization performance
- May smooth out important signals
- Computationally intensive
- Complex hyperparameter tuning
"""
        
        self.performance_text.insert(tk.END, performance_text)
        
    def compare_all_models(self):
        """Compare all available models systematically"""
        self.progress.start()
        self.compare_btn.config(state='disabled')
        self.status_var.set("Running comprehensive model comparison...")
        
        # Select all models for comparison
        for var in self.model_vars.values():
            var.set(True)
        
        thread = threading.Thread(target=self._perform_comparison)
        thread.daemon = True
        thread.start()
        
    def _perform_comparison(self):
        """Perform comprehensive model comparison"""
        try:
            symbol = self.symbol_var.get().upper()
            period = self.period_var.get()
            
            comparison_results = {}
            test_periods = [5, 10, 20]  # Different prediction horizons
            
            for days in test_periods:
                self.root.after(0, lambda d=days: self.training_status.config(text=f"Testing {d}-day predictions..."))
                
                day_results = {}
                
                # Test each model
                models_to_test = [
                    ('Markov Chain', 'return_bins'),
                    ('LSTM Neural Network', 'LSTM'),
                    ('ML Ensemble', 'ensemble')
                ]
                
                for model_name, model_type in models_to_test:
                    try:
                        if model_name == 'Markov Chain':
                            predictor = MarkovStockPredictor(symbol, period)
                            predictor.fetch_data()
                            predictor.calculate_daily_returns()
                            predictor.define_states(method='return_bins', n_states=6)
                            predictor.build_transition_matrix()
                            
                            predictions = predictor.predict_price_values(days)
                            accuracy = predictor.backtest_predictions(test_days=30)
                            
                        elif 'Neural Network' in model_name:
                            predictor = NeuralNetworkPredictor(symbol, period)
                            predictor.fetch_data()
                            training_results = predictor.train_model(model_type='LSTM', epochs=30)
                            predictions = predictor.predict_price_values(days)
                            accuracy = 1 - training_results['mae']
                            
                        elif model_name == 'ML Ensemble':
                            predictor = MLPredictor(symbol, period)
                            predictor.fetch_data()
                            training_results = predictor.train_models()
                            predictions = predictor.predict_with_ensemble(days)
                            avg_mae = np.mean([r['mae'] for r in training_results.values()])
                            accuracy = max(0, 1 - avg_mae / 100)
                        
                        if predictions:
                            current_price = predictions['current_price']
                            final_price = predictions['predicted_prices'][-1]
                            return_pct = ((final_price - current_price) / current_price) * 100
                            
                            day_results[model_name] = {
                                'return_pct': return_pct,
                                'accuracy': accuracy,
                                'final_price': final_price,
                                'volatility': np.std([((p - current_price) / current_price) * 100 
                                                    for p in predictions['predicted_prices']])
                            }
                    
                    except Exception as e:
                        print(f"Error testing {model_name} for {days} days: {e}")
                        continue
                
                comparison_results[days] = day_results
            
            self.root.after(0, self._update_comparison_results, comparison_results)
            
        except Exception as e:
            self.root.after(0, self._show_error, f"Comparison failed: {str(e)}")
        finally:
            self.root.after(0, lambda: (self.progress.stop(), self.compare_btn.config(state='normal')))
    
    def _update_comparison_results(self, comparison_results):
        """Update the comparison tab with results"""
        self.comparison_text.delete(1.0, tk.END)
        
        symbol = self.symbol_var.get().upper()
        
        comparison_text = f"""
⚖️ COMPREHENSIVE MODEL COMPARISON FOR {symbol}
═══════════════════════════════════════════════════════════════

📊 MULTI-HORIZON ANALYSIS:
Testing prediction accuracy across different time horizons...

"""
        
        # Create comparison table
        for days, results in comparison_results.items():
            if not results:
                continue
                
            comparison_text += f"\n🎯 {days}-DAY PREDICTIONS:\n"
            comparison_text += "─" * 70 + "\n"
            comparison_text += f"{'Model':<20} {'Return %':<10} {'Accuracy':<10} {'Volatility':<12} {'Score':<8}\n"
            comparison_text += "─" * 70 + "\n"
            
            scores = {}
            for model_name, data in results.items():
                return_pct = data['return_pct']
                accuracy = data['accuracy']
                volatility = data['volatility']
                
                # Calculate composite score (accuracy weighted heavily, prefer moderate volatility)
                score = (accuracy * 0.6) + (min(abs(return_pct), 10) / 50 * 0.3) + (max(0, 10 - volatility) / 10 * 0.1)
                scores[model_name] = score
                
                comparison_text += f"{model_name:<20} {return_pct:+6.1f}% {' ':<3} {accuracy:>6.1%} {' ':<3} {volatility:>7.2f}% {' ':<4} {score:>5.2f}\n"
            
            # Find best model for this horizon
            if scores:
                best_model = max(scores, key=scores.get)
                comparison_text += f"\n🏆 Best for {days} days: {best_model} (Score: {scores[best_model]:.2f})\n"
        
        # Overall recommendations
        comparison_text += f"\n📋 HORIZON-SPECIFIC RECOMMENDATIONS:\n"
        comparison_text += "─" * 50 + "\n"
        
        comparison_text += """
SHORT-TERM (3-5 days):
• Markov Chain often performs best
• Captures immediate regime changes
• Lower computational requirements

MEDIUM-TERM (10-15 days):
• Neural Networks show strength
• Can capture complex patterns
• Balance between accuracy and stability

LONG-TERM (20-30 days):
• ML Ensemble recommended
• More stable predictions
• Reduces overfitting risk

GENERAL GUIDELINES:
• Use multiple models for robustness
• Weight by recent performance
• Consider market volatility
• Validate with out-of-sample testing
"""
        
        # Model selection matrix
        comparison_text += f"\n📊 MODEL SELECTION MATRIX:\n"
        comparison_text += "─" * 40 + "\n"
        comparison_text += f"{'Scenario':<25} {'Recommended Model':<20}\n"
        comparison_text += "─" * 45 + "\n"
        comparison_text += f"{'High Volatility':<25} {'Markov Chain':<20}\n"
        comparison_text += f"{'Trending Market':<25} {'Neural Network':<20}\n"
        comparison_text += f"{'Sideways Market':<25} {'ML Ensemble':<20}\n"
        comparison_text += f"{'Earnings Season':<25} {'Conservative Blend':<20}\n"
        comparison_text += f"{'News-Heavy Period':<25} {'Short-term Models':<20}\n"
        
        self.comparison_text.insert(tk.END, comparison_text)
        
    def ensemble_prediction(self):
        """Create an intelligent ensemble prediction"""
        if not self.predictors:
            messagebox.showwarning("No Models", "Please run analysis first to train models")
            return
        
        self.progress.start()
        self.ensemble_btn.config(state='disabled')
        
        thread = threading.Thread(target=self._create_ensemble)
        thread.daemon = True
        thread.start()
        
    def _create_ensemble(self):
        """Create weighted ensemble prediction"""
        try:
            days = int(self.days_var.get())
            
            # Get predictions from all available models
            ensemble_data = {}
            weights = {}
            
            for model_name, predictor in self.predictors.items():
                if hasattr(predictor, 'predict_price_values'):
                    predictions = predictor.predict_price_values(days)
                    if predictions:
                        ensemble_data[model_name] = predictions
                        
                        # Calculate weight based on accuracy and model type
                        accuracy = predictions.get('accuracy', 0.5)
                        
                        # Model-specific weight adjustments
                        if model_name == 'Markov Chain':
                            base_weight = 0.3
                        elif 'Neural Network' in model_name:
                            base_weight = 0.4
                        elif model_name == 'ML Ensemble':
                            base_weight = 0.3
                        else:
                            base_weight = 0.25
                        
                        # Adjust weight by accuracy
                        weights[model_name] = base_weight * (0.5 + accuracy)
            
            if not ensemble_data:
                raise Exception("No valid predictions available for ensemble")
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Create weighted ensemble
            current_price = list(ensemble_data.values())[0]['current_price']
            ensemble_predictions = []
            ensemble_lower = []
            ensemble_upper = []
            
            for day in range(days):
                day_predictions = []
                day_lower = []
                day_upper = []
                
                for model_name, data in ensemble_data.items():
                    weight = weights[model_name]
                    day_predictions.append(data['predicted_prices'][day] * weight)
                    
                    if 'lower_bounds' in data and 'upper_bounds' in data:
                        day_lower.append(data['lower_bounds'][day] * weight)
                        day_upper.append(data['upper_bounds'][day] * weight)
                
                ensemble_predictions.append(sum(day_predictions))
                ensemble_lower.append(sum(day_lower) if day_lower else ensemble_predictions[-1] * 0.95)
                ensemble_upper.append(sum(day_upper) if day_upper else ensemble_predictions[-1] * 1.05)
            
            ensemble_result = {
                'model_name': 'Intelligent Ensemble',
                'current_price': current_price,
                'predicted_prices': ensemble_predictions,
                'lower_bounds': ensemble_lower,
                'upper_bounds': ensemble_upper,
                'weights': weights,
                'component_models': list(ensemble_data.keys())
            }
            
            self.root.after(0, self._display_ensemble_result, ensemble_result)
            
        except Exception as e:
            self.root.after(0, self._show_error, f"Ensemble creation failed: {str(e)}")
        finally:
            self.root.after(0, lambda: (self.progress.stop(), self.ensemble_btn.config(state='normal')))
    
    def _display_ensemble_result(self, ensemble_result):
        """Display ensemble prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        symbol = self.symbol_var.get().upper()
        current_price = ensemble_result['current_price']
        final_price = ensemble_result['predicted_prices'][-1]
        return_pct = ((final_price - current_price) / current_price) * 100
        days = len(ensemble_result['predicted_prices'])
        
        results_text = f"""
🎯 INTELLIGENT ENSEMBLE PREDICTION FOR {symbol}
═══════════════════════════════════════════════════════════════

📊 ENSEMBLE CONFIGURATION:
• Current Price: ${current_price:.2f}
• Prediction Horizon: {days} days
• Target Price: ${final_price:.2f}
• Expected Return: {return_pct:+.2f}%
• Component Models: {len(ensemble_result['component_models'])}

🏆 MODEL WEIGHTS:
"""
        
        for model, weight in ensemble_result['weights'].items():
            results_text += f"  • {model}: {weight:.1%}\n"
        
        results_text += f"\n📈 ENSEMBLE ADVANTAGES:\n"
        results_text += "─" * 35 + "\n"
        results_text += "• Reduces individual model bias\n"
        results_text += "• Improves prediction stability\n"
        results_text += "• Weights models by performance\n"
        results_text += "• Captures diverse market aspects\n"
        
        # Daily predictions
        results_text += f"\n📊 DAILY ENSEMBLE FORECAST:\n"
        results_text += "─" * 50 + "\n"
        results_text += f"{'Day':<4} {'Price':<10} {'Return':<8} {'Lower':<10} {'Upper':<10}\n"
        results_text += "─" * 50 + "\n"
        
        future_dates = pd.bdate_range(start=pd.Timestamp.now() + pd.Timedelta(days=1), periods=days)
        
        for i, (date, price, lower, upper) in enumerate(zip(
            future_dates, 
            ensemble_result['predicted_prices'],
            ensemble_result['lower_bounds'],
            ensemble_result['upper_bounds']
        )):
            day_return = ((price - current_price) / current_price) * 100
            results_text += f"{i+1:<4} ${price:<9.2f} {day_return:+5.1f}% ${lower:<9.2f} ${upper:<9.2f}\n"
        
        # Risk analysis
        max_price = max(ensemble_result['predicted_prices'])
        min_price = min(ensemble_result['predicted_prices'])
        volatility = ((max_price - min_price) / current_price) * 100
        
        results_text += f"\n⚠️ ENSEMBLE RISK METRICS:\n"
        results_text += "─" * 35 + "\n"
        results_text += f"Best Case: +{((max_price/current_price - 1)*100):.1f}%\n"
        results_text += f"Worst Case: {((min_price/current_price - 1)*100):.1f}%\n"
        results_text += f"Volatility Range: ±{volatility/2:.1f}%\n"
        
        # Confidence scoring
        prediction_range = max(ensemble_result['upper_bounds']) - min(ensemble_result['lower_bounds'])
        confidence = max(50, 90 - (prediction_range / current_price * 100))
        
        results_text += f"Ensemble Confidence: {confidence:.0f}%\n"
        
        if confidence > 80:
            confidence_level = "🟢 HIGH"
        elif confidence > 60:
            confidence_level = "🟡 MEDIUM"
        else:
            confidence_level = "🔴 LOW"
        
        results_text += f"Confidence Level: {confidence_level}\n"
        
        # Trading recommendation
        results_text += f"\n🎯 ENSEMBLE TRADING RECOMMENDATION:\n"
        results_text += "─" * 45 + "\n"
        
        if abs(return_pct) < 2:
            recommendation = "➡️ HOLD - Minimal expected movement"
        elif return_pct > 5:
            recommendation = "📈 BUY - Strong bullish consensus"
        elif return_pct > 2:
            recommendation = "📈 LEAN BUY - Moderate upside"
        elif return_pct < -5:
            recommendation = "📉 SELL - Strong bearish consensus"
        elif return_pct < -2:
            recommendation = "📉 LEAN SELL - Moderate downside"
        else:
            recommendation = "➡️ NEUTRAL - Mixed signals"
        
        results_text += f"{recommendation}\n"
        
        # Position sizing
        if volatility < 5 and abs(return_pct) > 3:
            position_size = "20-35%"
        elif volatility < 10 and abs(return_pct) > 2:
            position_size = "10-20%"
        else:
            position_size = "5-10%"
        
        results_text += f"Suggested Position Size: {position_size} of portfolio\n"
        
        results_text += f"\n📚 ENSEMBLE METHODOLOGY:\n"
        results_text += "─" * 35 + "\n"
        results_text += "• Weights based on historical accuracy\n"
        results_text += "• Model diversity ensures robustness\n"
        results_text += "• Confidence intervals from model variance\n"
        results_text += "• Dynamic weighting by model type\n"
        
        results_text += f"\n⚠️ ENSEMBLE LIMITATIONS:\n"
        results_text += "• Can mask important minority signals\n"
        results_text += "• May be overly conservative\n"
        results_text += "• Depends on component model quality\n"
        results_text += "• Not immune to systematic market shifts\n"
        
        self.results_text.insert(tk.END, results_text)
        
        # Update status
        self.status_var.set(f"Ensemble prediction complete - {return_pct:+.1f}% expected return")
        
    def clear_results(self):
        """Clear all results and charts"""
        for text_widget in [self.results_text, self.comparison_text, 
                           self.analysis_text, self.performance_text]:
            text_widget.delete(1.0, tk.END)
        
        self.fig.clear()
        self.canvas.draw()
        self.predictors.clear()
        self.status_var.set("Results cleared - Ready for new analysis")
        
    def _show_error(self, error_message):
        """Show error message to user"""
        messagebox.showerror("Analysis Error", error_message)
        self.status_var.set("Error occurred - Please check your inputs")
        
    def _analysis_complete(self):
        """Clean up after analysis completion"""
        self.progress.stop()
        self.analyze_btn.config(state='normal')
        self.training_status.config(text="")

def main():
    """Main function to run the multi-model application"""
    print("🚀 Starting Advanced Multi-Model Stock Market Predictor...")
    print("="*80)
    
    try:
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            print(f"✅ TensorFlow {tf.__version__} detected")
        except ImportError:
            print("⚠️  TensorFlow not found - Neural Network models will be unavailable")
        
        # Check scikit-learn availability
        try:
            import sklearn
            print(f"✅ Scikit-learn {sklearn.__version__} detected")
        except ImportError:
            print("⚠️  Scikit-learn not found - ML Ensemble models will be unavailable")
        
        root = tk.Tk()
        app = MultiModelStockPredictorGUI(root)
        
        print("✅ Multi-Model GUI Application launched successfully!")
        print("\n🤖 Available Models:")
        print("   • 🎲 Markov Chain - State-based predictions")
        print("   • 🧠 LSTM Neural Network - Long short-term memory")
        print("   • 🔄 GRU Neural Network - Gated recurrent unit")
        print("   • 🔗 CNN-LSTM Hybrid - Convolutional + LSTM")
        print("   • 📊 ML Ensemble - Multiple algorithms combined")
        
        print("\n🎯 Key Features:")
        print("   • 🔄 Multi-model comparison and validation")
        print("   • 🎯 Intelligent ensemble predictions")
        print("   • 📈 Advanced technical analysis")
        print("   • 📊 Comprehensive performance metrics")
        print("   • 🎨 Interactive visualization")
        print("   • ⚖️ Risk-adjusted recommendations")
        
        print("\n🎮 How to use:")
        print("   1. Select which models to use")
        print("   2. Enter stock symbol and parameters")
        print("   3. Click 'Analyze & Predict' for individual models")
        print("   4. Use 'Compare All Models' for comprehensive analysis")
        print("   5. Try 'Ensemble Prediction' for weighted combination")
        print("   6. Review results across multiple tabs")
        
        print(f"\n📋 Recommended Workflow:")
        print(f"   • Start with 'Compare All Models' for overview")
        print(f"   • Use 'Ensemble Prediction' for final decision")
        print(f"   • Check 'Model Performance' tab for validation")
        print(f"   • Review 'Technical Analysis' for context")
        
        print(f"\n⚠️  Important Notes:")
        print(f"   • Neural networks require significant training time")
        print(f"   • Start with shorter prediction horizons (5-10 days)")
        print(f"   • Multiple models provide better validation")
        print(f"   • This is for educational purposes only")
        print(f"   • Always do your own research before investing")
        
        root.mainloop()
        
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("📦 Please install required packages:")
        print("   pip install numpy pandas yfinance matplotlib tkinter")
        print("   pip install tensorflow scikit-learn")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("🔧 Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()

# Quick usage examples for command line:
def quick_multi_predict(symbol="AAPL", days=10, models=['markov', 'lstm']):
    """Quick multi-model prediction function"""
    try:
        results = {}
        
        if 'markov' in models:
            predictor = MarkovStockPredictor(symbol, period="2y")
            predictor.fetch_data()
            predictor.calculate_daily_returns()
            predictor.define_states(method='return_bins', n_states=6)
            predictor.build_transition_matrix()
            
            predictions = predictor.predict_price_values(days)
            if predictions:
                results['Markov Chain'] = predictions
        
        if 'lstm' in models:
            try:
                predictor = NeuralNetworkPredictor(symbol, period="2y")
                predictor.fetch_data()
                predictor.train_model(model_type='LSTM', epochs=30)
                
                predictions = predictor.predict_price_values(days)
                if predictions:
                    results['LSTM'] = predictions
            except ImportError:
                print("TensorFlow not available for LSTM model")
        
        if 'ensemble' in models:
            try:
                predictor = MLPredictor(symbol, period="2y")
                predictor.fetch_data()
                predictor.train_models()
                
                predictions = predictor.predict_with_ensemble(days)
                if predictions:
                    results['ML Ensemble'] = predictions
            except ImportError:
                print("Scikit-learn not available for ML Ensemble")
        
        if results:
            print(f"\n🚀 Multi-Model Prediction for {symbol}:")
            for model, data in results.items():
                current = data['current_price']
                target = data['predicted_prices'][-1]
                return_pct = ((target - current) / current) * 100
                print(f"   {model}: ${target:.2f} ({return_pct:+.1f}%)")
            
            # Ensemble average
            avg_target = np.mean([data['predicted_prices'][-1] for data in results.values()])
            avg_return = ((avg_target - results[list(results.keys())[0]]['current_price']) / 
                         results[list(results.keys())[0]]['current_price']) * 100
            print(f"   Ensemble Average: ${avg_target:.2f} ({avg_return:+.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Example usage:
# quick_multi_predict("TSLA", 5, ['markov', 'lstm'])

# quick_multi_predict("SPY", 20, ['markov', 'ensemble'])
