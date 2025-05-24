# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
from pandas import DataFrame
import sys
import os
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Any, Literal
import numpy as np
import pandas as pd
from enum import Enum

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

# TA imports
import talib.abstract as ta

# Configure logger
logger = logging.getLogger(__name__)

# Log phases
class LogPhase:
    GENERATE_IMAGES = "📊 GENERATE_IMAGES"
    LLM_ANALYSIS = "🧠 LLM_ANALYSIS"
    UPDATE_ORDER = "📝 UPDATE_ORDER"
    SIGNAL = "🚦 SIGNAL"
    INDICATORS = "📈 INDICATORS"

# Icon types for logs
class IconType(Enum):
    NONE = ""
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    MONEY = "💰"
    CHART = "📈"
    DOWN = "📉"
    BRAIN = "🧠"
    ROCKET = "🚀"
    FIRE = "🔥"
    EYES = "👀"
    CLOCK = "⏰"
    ROBOT = "🤖"
    BUY = "🛒"
    SELL = "💸"

# Safely import plotting module
PLOT_CANDLES_AVAILABLE = False
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(current_dir, 'utils')
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    if os.path.exists(os.path.join(utils_dir, 'plot_candles.py')):
        import plot_candles
        PLOT_CANDLES_AVAILABLE = True
        logger.info("Plot candles module loaded successfully")
    else:
        logger.warning("plot_candles.py not found in utils directory")
except Exception as e:
    logger.error(f"Error importing plot_candles module: {e}")

# Safely import LLM client
LLM_CLIENT_AVAILABLE = False
try:
    if os.path.exists(os.path.join(utils_dir, 'llm_client.py')):
        from utils.llm_client import LLMClient, OPENAI_AVAILABLE as LLM_CLIENT_AVAILABLE
        logger.info("LLM client module loaded successfully")
    else:
        logger.warning("llm_client.py not found in utils directory")
except Exception as e:
    logger.error(f"Error importing LLM client module: {e}")

# Safely import Discord webhook
DISCORD_WEBHOOK_AVAILABLE = False
try:
    if os.path.exists(os.path.join(utils_dir, 'discord_webhook.py')):
        from utils.discord_webhook import DiscordWebhook
        DISCORD_WEBHOOK_AVAILABLE = True
        logger.info("Discord webhook module loaded successfully")
    else:
        logger.warning("discord_webhook.py not found in utils directory")
except Exception as e:
    logger.error(f"Error importing Discord webhook module: {e}")

class SampleStrategy(IStrategy):
    """
    Simple trading strategy that uses LLM analysis of chart images to make decisions
    """
    INTERFACE_VERSION = 3
    can_short = False

    # Default ROI and stoploss (will be overridden by LLM recommendations)
    minimal_roi = {
        "0": 0.5,
        "120": 0.2,
        "180": 0.1,
    }
    # stoploss is a percentage of the entry price
    # 0.02 is 2%
    stoploss = -0.02
    # trailing_stop_positive is a percentage of the entry price
    # 0.01 is 1%
    trailing_stop_positive = 0.02

    # No trailing stop by default
    trailing_stop = True

    # Trading timeframe
    timeframe = "1h"

    # Only process new candles
    process_only_new_candles = True

    # Exit settings
    use_exit_signal = True    
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles required
    startup_candle_count = 100

    # Order settings
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Directory for saving chart images
    candle_plot_dir = "user_data/plot"
    
    # LLM settings
    llm_model = "gpt-4.1"
    llm_temperature = 0.7
    llm_max_tokens = 4096

    def __init__(self, config: dict) -> None:
        """Initialize the strategy"""
        super().__init__(config)
        
        # Create plot directory
        os.makedirs(os.path.join(config['user_data_dir'], 'plot'), exist_ok=True)
        self.absolute_plot_dir = os.path.join(config['user_data_dir'], 'plot')
        
        # Get OpenAI API key
        self.openai_api_key = config.get('openai_api_key') or os.environ.get("OPENAI_API_KEY")
        
        # Initialize LLM client
        self.llm_client = None
        if LLM_CLIENT_AVAILABLE and self.openai_api_key:
            try:
                self.llm_client = LLMClient(api_key=self.openai_api_key)
                logger.info(f"LLM client initialized with model {self.llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
        
        # Dictionary to store custom stop loss and take profit values
        self.custom_sl_tp = {}
        
        # Path to LLM system prompt file
        self.llm_prompt_file = os.path.join(config['user_data_dir'], 'config', 'llm_system_prompt.md')

    def load_llm_system_prompt(self) -> str:
        """Load the LLM system prompt from markdown file"""
        with open(self.llm_prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove markdown formatting for plain text prompt
            import re
            # Remove headers (##, ###, etc.)
            content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)
            # Remove bold markers (**)
            content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
            # Remove bullet points (- )
            content = re.sub(r'^-\s+', '', content, flags=re.MULTILINE)
            # Clean up extra whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)
            return content.strip()

    def log_formatted(self, 
                     pair: str, 
                     phase: str, 
                     level: Literal["INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL"], 
                     message: str,
                     icon: IconType = IconType.NONE) -> None:
        """
        Log a message with standardized formatting
        
        Parameters:
        -----------
        pair : str
            Trading pair (e.g., 'BTC/USDT')
        phase : str
            Processing phase (use LogPhase constants)
        level : str
            Log level (INFO, WARNING, ERROR, DEBUG, CRITICAL)
        message : str
            The log message
        icon : IconType
            Optional icon to prepend to the message (default: IconType.NONE)
        """
        icon_str = icon.value + " " if icon.value else ""
        formatted_message = f"[{pair:10}][{phase}]: {icon_str}{message}"
        
        if level == "INFO":
            logger.info(formatted_message)
        elif level == "WARNING":
            logger.warning(formatted_message)
        elif level == "ERROR":
            logger.error(formatted_message)
        elif level == "DEBUG":
            logger.debug(formatted_message)
        elif level == "CRITICAL":
            logger.critical(formatted_message)

    def bollinger_bands(self, dataframe: DataFrame) -> DataFrame:
        """Calculate Bollinger Bands"""
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe['close'],
            timeperiod=20,
            nbdevup=2.0,
            nbdevdn=2.0,
            matype=0
        )
        return dataframe

    def calculate_vwap(self, dataframe: DataFrame, window: int = 20) -> DataFrame:
        """Calculate Volume Weighted Average Price (VWAP) manually"""
        # Make sure the dataframe has volume and typical price
        if 'volume' not in dataframe.columns:
            return dataframe
            
        # Calculate typical price: (high + low + close) / 3
        if 'typical_price' not in dataframe.columns:
            dataframe['typical_price'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
            
        # Calculate VWAP
        dataframe['vp'] = dataframe['typical_price'] * dataframe['volume']
        dataframe['vwap'] = dataframe['vp'].rolling(window=window).sum() / dataframe['volume'].rolling(window=window).sum()
        
        # Clean up temporary columns
        dataframe = dataframe.drop(columns=['vp', 'typical_price'], errors='ignore')
        
        return dataframe

    def calculate_volume_price_heatmap(self, dataframe: DataFrame, bins: int = 20, window: int = 14) -> DataFrame:
        """
        Calculate a heatmap showing where the money is concentrated based on volume and price amplitude.
        Adds a 'volume_price_heatmap' column to the dataframe.
        
        Parameters:
        -----------
        dataframe : DataFrame
            DataFrame containing OHLCV data
        bins : int, optional
            Number of price bins to divide the price range into (default: 20)
        window : int, optional
            Rolling window size for calculating the heatmap intensity (default: 14)
            
        Returns:
        --------
        DataFrame
            The input dataframe with added 'price_bin_centers', 'price_bin_values', and 'volume_price_intensity' columns
        """
        try:
            # Make a copy of the dataframe to avoid modifying the original
            df = dataframe.copy()
            
            # Calculate price amplitude (high - low) for each candle
            df['price_amplitude'] = df['high'] - df['low']
            
            # Calculate typical price (média do OHLC)
            df['typical_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            # Calculate volume * price amplitude as money intensity
            df['money_intensity'] = df['volume'] * df['price_amplitude']
            
            # Create a rolling window for recent price range
            recent_min = df['low'].rolling(window=window).min()
            recent_max = df['high'].rolling(window=window).max()
            
            # Initialize arrays to hold the heatmap data - explicitly set as object dtypes
            df['price_bin_centers'] = pd.Series(index=df.index, dtype='object')
            df['price_bin_values'] = pd.Series(index=df.index, dtype='object')
            df['volume_price_intensity'] = pd.Series(index=df.index, dtype='object')
            
            # For each row in the active calculation range (after the window)
            for i in range(window, len(df)):
                if np.isnan(recent_min[i]) or np.isnan(recent_max[i]):
                    continue
                    
                # Define price range for this window
                price_min = recent_min[i]
                price_max = recent_max[i]
                
                # Avoid division by zero
                if price_max <= price_min:
                    price_max = price_min * 1.001  # Add a small buffer
                    
                # Create price bins
                price_bins = np.linspace(price_min, price_max, bins + 1)
                price_bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
                
                # Initialize bin values
                bin_values = np.zeros(bins)
                
                # Get relevant data window
                window_data = df.iloc[i-window+1:i+1]
                
                # For each candle in window, distribute money intensity across price range
                for _, candle in window_data.iterrows():
                    # Find which bins this candle's price range covers
                    low_idx = np.searchsorted(price_bins, candle['low']) - 1
                    high_idx = np.searchsorted(price_bins, candle['high'])
                    
                    # Find the bin where the typical price falls
                    typical_idx = np.searchsorted(price_bins, candle['typical_price']) - 1
                    
                    # Keep indices within valid range
                    low_idx = max(0, low_idx)
                    high_idx = min(bins, high_idx)
                    typical_idx = max(0, min(bins-1, typical_idx))
                    
                    # Skip if invalid range
                    if high_idx <= low_idx:
                        continue
                        
                    # Calculate covered price range
                    covered_range = price_bins[high_idx] - price_bins[low_idx]
                    
                    # Avoid division by zero
                    if covered_range <= 0:
                        continue
                        
                    # Distribute money intensity proportionally to bins
                    base_intensity = candle['money_intensity'] / (high_idx - low_idx)
                    
                    # Add base intensity to each covered bin
                    for j in range(low_idx, high_idx):
                        # Add base intensity to all bins in the range
                        bin_values[j] += base_intensity * 0.7  # 70% of intensity distributed equally
                        
                        # If this is the bin containing the typical price, add extra weight
                        if j == typical_idx:
                            # Add extra 30% to the typical price bin
                            bin_values[j] += base_intensity * 0.3
                
                # Normalize bin values (0 to 1)
                if np.max(bin_values) > 0:
                    bin_values = bin_values / np.max(bin_values)
                
                # Store the results in the dataframe
                df.at[i, 'price_bin_centers'] = str(price_bin_centers.tolist())
                df.at[i, 'price_bin_values'] = str(bin_values.tolist())
                df.at[i, 'volume_price_intensity'] = str(list(zip(price_bin_centers.tolist(), bin_values.tolist())))
            
            # Copy the calculated columns back to the original dataframe
            dataframe['price_bin_centers'] = df['price_bin_centers'].copy()
            dataframe['price_bin_values'] = df['price_bin_values'].copy()
            dataframe['volume_price_intensity'] = df['volume_price_intensity'].copy()
                    
            return dataframe
            
        except Exception as e:
            logger.error(f"Error calculating volume-price heatmap: {e}", exc_info=True)
            return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate indicators and generate chart for analysis"""
        pair = metadata['pair']
        
        self.log_formatted(pair, LogPhase.INDICATORS, "INFO", "Calculating technical indicators")
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Medias moveis exponenciais
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        
        # Calculate VWAP manually instead of using ta.VWAP
        dataframe = self.calculate_vwap(dataframe, window=20)
        
        # Calculate MACD - Fix: MACD returns tuple of arrays, not a dictionary
        macd, macdsignal, macdhist = ta.MACD(dataframe['close'])
        dataframe['macd'] = macd
        dataframe['macdsignal'] = macdsignal
        dataframe['macdhist'] = macdhist
        
        # Bollinger Bands
        dataframe = self.bollinger_bands(dataframe)
        
        # Calculate volume-price heatmap
        dataframe = self.calculate_volume_price_heatmap(dataframe, bins=20, window=self.startup_candle_count)
        
        # Generate chart image for LLM analysis
        chart_emas_result = None
        chart_bollinger_result = None
        chart_vwap_result = None
        chart_heatmap_result = None
        if self.process_only_new_candles and not dataframe.empty and PLOT_CANDLES_AVAILABLE:
            try:
                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Starting chart generation process")
                
                # Plot candles with indicators
                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Generating EMA chart")
                chart_emas_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='emas',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=self.startup_candle_count,
                    output_dir=self.absolute_plot_dir,
                    indicators=[
                        {'name': 'ema_9', 'color': 'red', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'ema_21', 'color': 'blue', 'panel': 0, 'width': 1.5, 'type': 'line'},
                    ],
                    indicators_below=[
                        {'name': 'rsi', 'color': 'purple', 'width': 1.0, 'type': 'line'},
                        {'name': 'macd', 'color': 'blue', 'width': 1.5, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macdsignal', 'color': 'red', 'width': 1.0, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macdhist', 'color': 'green', 'width': 0.8, 'panel': 'MACD', 'type': 'bar'},
                    ],
                    title=f"{pair} - {self.timeframe}",
                    subtitle=f"Moving Average Exponencial 9 e 21, Convergence Divergence (MACD) e Relative Strength Index (RSI)",
                    question=f"What is the trend for {pair} in the {self.timeframe} timeframe? What is the trend for the EMA 9 and EMA 21?"
                )
                if chart_emas_result:
                    self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", f"EMA chart generated: {chart_emas_result.get('filepath', 'unknown')}")

                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Generating Bollinger Bands chart")
                chart_bollinger_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='bollinger',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=self.startup_candle_count,
                    output_dir=self.absolute_plot_dir,
                    indicators=[
                        {'name': 'bb_upper', 'color': 'red', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'bb_middle', 'color': 'blue', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'bb_lower', 'color': 'green', 'panel': 0, 'width': 1.5, 'type': 'line'},
                    ],
                    indicators_below=[
                        {'name': 'rsi', 'color': 'purple', 'width': 1.0, 'type': 'line'},
                        {'name': 'macd', 'color': 'blue', 'width': 1.5, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macdsignal', 'color': 'red', 'width': 1.0, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macdhist', 'color': 'green', 'width': 0.8, 'panel': 'MACD', 'type': 'bar'},
                    ],
                    title=f"{pair} - {self.timeframe}",
                    subtitle=f"Bollinger Bands 20, 2 desvios padrao, Media Movel Exponencial 9 e 21, Convergence Divergence (MACD) e Relative Strength Index (RSI)",
                    question=f"What is the trend for {pair} in the {self.timeframe} timeframe? What is the trend for the Bollinger Bands?"
                )
                if chart_bollinger_result:
                    self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", f"Bollinger chart generated: {chart_bollinger_result.get('filepath', 'unknown')}")
                
                # Plot VWAP chart
                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Generating VWAP chart")
                chart_vwap_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='vwap',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=self.startup_candle_count,
                    output_dir=self.absolute_plot_dir,
                    indicators=[
                        {'name': 'vwap', 'color': 'blue', 'panel': 0, 'width': 1.5, 'type': 'line'},
                    ],
                    indicators_below=[
                        {'name': 'rsi', 'color': 'purple', 'width': 1.0, 'type': 'line'}
                    ],
                    title=f"{pair} - {self.timeframe} - VWAP",
                    subtitle=f"Volume Weighted Average Price (VWAP) 20",
                    question=f"What is the trend for {pair} in the {self.timeframe} timeframe? What is the trend for the VWAP?"
                )
                if chart_vwap_result:
                    self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", f"VWAP chart generated: {chart_vwap_result.get('filepath', 'unknown')}")
                
                # Plot heatmap chart
                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Generating Volume-Price Heatmap chart")
                chart_heatmap_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='heatmap',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=self.startup_candle_count,
                    output_dir=self.absolute_plot_dir,
                    indicators=[
                        {'name': 'ema_9', 'color': 'red', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'ema_21', 'color': 'blue', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'volume_price_intensity', 'panel': 0, 'type': 'heatmap'},
                    ],
                    indicators_below=[],
                    title=f"{pair} - {self.timeframe} - Volume-Price Heatmap",
                    subtitle=f"Heatmap showing where the money is concentrated based on volume and price amplitude",
                    question=f"Where is the money concentrated in {pair} price levels? Which price levels have the highest volume and price activity? Please identify the most important price ranges from the heatmap and sort them by significance (from most to least important). Compare these key price levels with the current price and indicate if the price is currently trading within, above, or below these significant zones."
                )
                if chart_heatmap_result:
                    self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", f"Heatmap chart generated: {chart_heatmap_result.get('filepath', 'unknown')}")
                
                # Get LLM analysis if charts were created and client is available
                if (chart_emas_result or chart_bollinger_result or chart_vwap_result or chart_heatmap_result) and self.llm_client and LLM_CLIENT_AVAILABLE:
                    # Create a list of all chart images to analyze
                    chart_images = []
                    chart_base64_images = []
                    
                    if chart_emas_result:
                        if chart_emas_result.get('filepath') and os.path.exists(chart_emas_result['filepath']):
                            chart_images.append(chart_emas_result['filepath'])
                            if chart_emas_result.get('base64'):
                                chart_base64_images.append({
                                    "path": chart_emas_result['filepath'],
                                    "base64": chart_emas_result['base64'],
                                    "question": chart_emas_result['question']
                                })
                                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "EMA chart encoded for LLM analysis")
                            
                    if chart_bollinger_result:
                        if chart_bollinger_result.get('filepath') and os.path.exists(chart_bollinger_result['filepath']):
                            chart_images.append(chart_bollinger_result['filepath'])
                            if chart_bollinger_result.get('base64'):
                                chart_base64_images.append({
                                    "path": chart_bollinger_result['filepath'],
                                    "base64": chart_bollinger_result['base64'],
                                    "question": chart_bollinger_result['question']
                                })
                                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Bollinger chart encoded for LLM analysis")

                    if chart_vwap_result:
                        if chart_vwap_result.get('filepath') and os.path.exists(chart_vwap_result['filepath']):
                            chart_images.append(chart_vwap_result['filepath'])
                            if chart_vwap_result.get('base64'):
                                chart_base64_images.append({
                                    "path": chart_vwap_result['filepath'],
                                    "base64": chart_vwap_result['base64'],
                                    "question": chart_vwap_result['question']
                                })
                                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "VWAP chart encoded for LLM analysis")

                    if chart_heatmap_result:
                        if chart_heatmap_result.get('filepath') and os.path.exists(chart_heatmap_result['filepath']):
                            chart_images.append(chart_heatmap_result['filepath'])
                            if chart_heatmap_result.get('base64'):
                                chart_base64_images.append({
                                    "path": chart_heatmap_result['filepath'],
                                    "base64": chart_heatmap_result['base64'],
                                    "question": chart_heatmap_result['question']
                                })
                                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "INFO", "Heatmap chart encoded for LLM analysis")
                                
                    if chart_images and chart_base64_images:
                        self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Starting LLM analysis with {len(chart_images)} charts")
                        llm_analysis = self.analyze_chart_with_llm(chart_images, chart_base64_images, pair, dataframe)
                        if llm_analysis:
                            # Store analysis for use in signal generation
                            if not hasattr(self, 'llm_analyses'):
                                self.llm_analyses = {}
                            self.llm_analyses[pair] = llm_analysis
                            
                            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", 
                                           f"Analysis complete: {llm_analysis['recommendation']} (confidence: {llm_analysis['confidence']})")
                    else:
                        self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", "No chart images available to analyze with LLM")
            except Exception as e:
                self.log_formatted(pair, LogPhase.GENERATE_IMAGES, "ERROR", f"Error in chart generation: {e}")
                
        return dataframe

    def analyze_chart_with_llm(self, image_paths: List[str], image_base64_list: List[Dict[str, str]], pair: str, dataframe: DataFrame) -> Optional[Dict[str, Any]]:
        """Send chart images to LLM for analysis"""
        if not self.llm_client or not all(os.path.exists(path) for path in image_paths):
            missing_paths = [path for path in image_paths if not os.path.exists(path)]
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", 
                            f"Cannot analyze chart: {'LLM client not available' if not self.llm_client else f'Images not found: {missing_paths}'}", 
                            IconType.ERROR)
            return None
            
        try:
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"STEP 1: Starting analysis with {len(image_paths)} images", IconType.BRAIN)
            
            # No need to encode images, as we already have the base64 from plot_candles
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"STEP 2: Using pre-encoded images for {len(image_base64_list)} charts", IconType.CHART)
            
            # Recent price data for context
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 3: Preparing price context", IconType.MONEY)
            recent_prices = dataframe.tail(5)[['date', 'close']].to_dict('records')
            price_context = ", ".join([f"{row['date']}: {row['close']:.2f}" for row in recent_prices])
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"Price context: {price_context}")
            
            # Get open trade information for this pair
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 4: Getting open trade information", IconType.EYES)
            trade_info = self._get_open_trade_info(pair)
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"Trade info: {trade_info}")
            
            # System message for LLM - Load from markdown file
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 5: Loading system message from markdown file", IconType.ROBOT)
            system_content = self.load_llm_system_prompt()
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"System prompt loaded: {len(system_content)} characters")
            
            # User message with chart and trade context
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 6: Creating user message with charts", IconType.CHART)
            user_content = f"Analyze these {pair} charts. Recent closing prices: {price_context}\n\n"
            
            # Add open trade information if available
            if trade_info:
                user_content += (
                    f"OPEN TRADE INFORMATION:\n"
                    f"- Entry price: {trade_info['entry_price']}\n"
                    f"- Current price: {trade_info['current_price']}\n"
                    f"- Current profit/loss: {trade_info['profit_pct']}%\n"
                    f"- Time in trade: {trade_info['time_in_trade']} hours\n"
                    f"- Stop loss set at: {trade_info['stop_loss_pct']}%\n"
                    f"- Take profit target: {trade_info['take_profit_pct']}%\n\n"
                )
                user_content += ("Based on the charts, current open position, and open orders, provide your analysis and recommendation. "
                                "If you recommend selling, explain why. If you recommend holding, explain for how much longer and what conditions would change your recommendation.")
            else:
                user_content += ("No open position currently exists. Based on the charts, provide your analysis and recommendation. "
                                "If you recommend buying, explain why and suggest appropriate stop loss and take profit levels.")
                
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"User message prepared: {user_content[:100]}...")
            
            # Format messages for OpenAI API with multiple images
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"STEP 7: Formatting messages with {len(image_base64_list)} images", IconType.CHART)
            
            # Start with text in the content
            content_list = [{"type": "text", "text": user_content}]
            
            # Add all images to content
            for img in image_base64_list:
                content_list.append({"type": "text", "text": img['question']})
                content_list.append(                    
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"}}
                )
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content_list}
            ]
            
            # Log the request
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", "REQUEST TO LLM:")
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"System content: {system_content[:100]}...")
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"Including {len(image_base64_list)} images")
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "DEBUG", f"Model: {self.llm_model}, Temperature: {self.llm_temperature}")
            
            # Send to LLM
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"STEP 8: Sending request to LLM with model {self.llm_model}", IconType.ROCKET)
            response = self.llm_client.send_message(
                messages=messages,
                model=self.llm_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Process response
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 9: Processing LLM response", IconType.BRAIN)
            if not response["success"]:
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", f"LLM request failed: {response.get('error', 'Unknown error')}", IconType.ERROR)
                return None
                
            if not isinstance(response["content"], dict):
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", f"LLM response is not a valid dictionary: {response['content']}", IconType.ERROR)
                return None
                
            content = response["content"]
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Received LLM response successfully", IconType.SUCCESS)
            
            # Validate required fields
            required_fields = ["trend", "confidence", "recommendation", "stop_loss", "take_profit"]
            missing_fields = [field for field in required_fields if field not in content]
            
            if missing_fields:
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", f"LLM response missing required fields: {missing_fields}", IconType.ERROR)
                return None
            
            # Save LLM analysis to JSON file with same name as the first chart image but with .json extension
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 10: Saving LLM analysis to JSON file", IconType.INFO)
            try:
                # Create JSON file path from the first image path
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
                json_file_path = os.path.join(os.path.dirname(image_paths[0]), f"{pair.replace('/', '_')}_{self.timeframe}_{current_datetime}.json")
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Saving analysis to: {json_file_path}")
                
                # Create a complete record with both request and response
                json_data = {
                    "request": messages,
                    "response": content
                }
                
                # Write the JSON file
                with open(json_file_path, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)
                    
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Analysis saved to JSON file: {json_file_path}", IconType.SUCCESS)
                
                # STEP 11: Send analysis to Discord if webhook URL is configured
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 11: Sending analysis to Discord", IconType.INFO)
                try:
                    # Get Discord webhook URL from config or environment
                    discord_webhook_url = self.config.get('discord_webhook_url') or os.environ.get('DISCORD_WEBHOOK_URL')
                    
                    if discord_webhook_url and DISCORD_WEBHOOK_AVAILABLE:
                        # Initialize Discord webhook client
                        discord = DiscordWebhook(webhook_url=discord_webhook_url)
                        
                        # Send analysis to Discord
                        result = discord.send_analysis(
                            pair=pair,
                            timeframe=self.timeframe,
                            analysis_json_path=json_file_path,
                            image_paths=image_paths,
                            username=f"FreqTrade {self.__class__.__name__}",
                            open_trade_info=self._get_open_trade_info(pair)
                        )
                        
                        if result and result.get('success'):
                            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "Analysis sent to Discord successfully", IconType.SUCCESS)
                        else:
                            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "WARNING", f"Failed to send to Discord: {result.get('error', 'Unknown error')}", IconType.WARNING)
                    elif not discord_webhook_url:
                        self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "Discord webhook URL not configured, skipping Discord integration", IconType.INFO)
                    elif not DISCORD_WEBHOOK_AVAILABLE:
                        self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "WARNING", "Discord webhook module not available, skipping Discord integration", IconType.WARNING)
                        
                except Exception as e:
                    self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", f"Error sending analysis to Discord: {e}", IconType.ERROR)
                    # Continue even if Discord sending fails
            except Exception as e:
                self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", f"Error saving LLM analysis to JSON file: {e}", IconType.ERROR)
                # Continue even if saving fails
            
            # Log detailed analysis
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "STEP 12: Analysis results:", IconType.CHART)
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Trend: {content.get('trend')}", 
                           IconType.CHART if content.get('trend') == 'neutral' else 
                           (IconType.ROCKET if content.get('trend') == 'bullish' else IconType.DOWN))
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Confidence: {content.get('confidence')}", IconType.INFO)
            
            # Choose icon based on recommendation
            rec_icon = IconType.INFO
            if content.get('recommendation') == 'buy':
                rec_icon = IconType.BUY
            elif content.get('recommendation') == 'sell':
                rec_icon = IconType.SELL
            elif content.get('recommendation') == 'hold':
                rec_icon = IconType.CLOCK
                
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Recommendation: {content.get('recommendation')}", rec_icon)
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", f"Stop Loss: {content.get('stop_loss')}%, Take Profit: {content.get('take_profit')}%", IconType.MONEY)
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "INFO", "Analysis validated and complete", IconType.SUCCESS)
                
            return content
                
        except Exception as e:
            self.log_formatted(pair, LogPhase.LLM_ANALYSIS, "ERROR", f"Error in LLM analysis: {e}", IconType.ERROR)
            return None

    def _get_open_trade_info(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get information about an open trade for a specific pair"""
        try:
            # Check for open trades for this pair
            open_trades = Trade.get_trades_proxy(is_open=True)
            for trade in open_trades:
                if trade.pair == pair:
                    # We found an open trade for this pair
                    current_price = self.dp.ticker(pair)['last']
                    profit_pct = ((current_price / trade.open_rate) - 1) * 100
                    time_in_trade = (datetime.now() - trade.open_date).total_seconds() / 3600  # hours
                    
                    # Get stop loss and take profit settings
                    stop_loss_pct = self.stoploss * 100
                    take_profit_pct = 0.0
                    
                    if pair in self.custom_sl_tp:
                        stop_loss_pct = self.custom_sl_tp[pair]['stop_loss_pct']
                        take_profit_pct = self.custom_sl_tp[pair]['take_profit_pct']
                    
                    self.log_formatted(pair, LogPhase.UPDATE_ORDER, "DEBUG", f"Found open trade with profit: {profit_pct:.2f}%, time: {time_in_trade:.1f}h")
                    
                    return {
                        'entry_price': trade.open_rate,
                        'current_price': current_price,
                        'profit_pct': round(profit_pct, 2),
                        'time_in_trade': round(time_in_trade, 1),
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct,
                        'stake_amount': trade.stake_amount,
                        'trade_id': trade.id,
                        'open_date': trade.open_date.strftime('%Y-%m-%d %H:%M:%S')
                    }
            
            # No open trade found
            self.log_formatted(pair, LogPhase.UPDATE_ORDER, "DEBUG", "No open trade found")
            return None
            
        except Exception as e:
            self.log_formatted(pair, LogPhase.UPDATE_ORDER, "ERROR", f"Error getting open trade info: {e}")
            return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate buy signals based on LLM recommendations"""
        pair = metadata['pair']
        
        # Initialize with no buy signals
        dataframe['enter_long'] = 0
        
        # Skip if in backtest/hyperopt or not processing new candles
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"Skipping LLM entry signals - running in {self.dp.runmode.value} mode")
            return dataframe
            
        if not self.process_only_new_candles:
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Skipping LLM entry signals - not processing new candles")
            return dataframe
            
        # Get LLM analysis
        self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Checking for LLM analysis", IconType.EYES)
        llm_analysis = getattr(self, 'llm_analyses', {}).get(pair)
        if not llm_analysis:
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "No LLM analysis available", IconType.INFO)
            return dataframe
            
        # Check recommendation and confidence
        self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Evaluating buy recommendation", IconType.BRAIN)
        recommendation = llm_analysis.get('recommendation', '').lower()
        confidence = float(llm_analysis.get('confidence', 0))
        
        self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"LLM recommendation: {recommendation.upper()} with {confidence:.2f} confidence")
        
        # Generate buy signal if LLM recommends with sufficient confidence
        if recommendation == 'buy' and confidence >= 0.6:
            # Set buy signal on last candle
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Generating BUY signal", IconType.BUY)
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"BUY SIGNAL GENERATED with {confidence:.2f} confidence", IconType.SUCCESS)
            
            # Store custom stop loss and take profit
            self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", "Setting stop loss and take profit parameters", IconType.MONEY)
            try:
                stop_loss = float(llm_analysis.get('stop_loss', 0))
                take_profit = float(llm_analysis.get('take_profit', 0))
                
                self.log_formatted(pair, LogPhase.UPDATE_ORDER, "DEBUG", f"Raw SL: {stop_loss}%, TP: {take_profit}%")
                
                current_price = dataframe['close'].iloc[-1]
                
                # Store for use in other methods
                self.custom_sl_tp[pair] = {
                    'stop_loss_pct': stop_loss,  # Already negative
                    'take_profit_pct': take_profit,
                    'entry_price': current_price,
                    'time_set': datetime.now()
                }
                
                self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", f"Set SL: {stop_loss}%, TP: {take_profit}% at price {current_price}", IconType.SUCCESS)
            except Exception as e:
                self.log_formatted(pair, LogPhase.UPDATE_ORDER, "ERROR", f"Error setting SL/TP: {e}", IconType.ERROR)
        else:
            if recommendation == 'hold':
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"HOLDING based on LLM recommendation with {confidence:.2f} confidence", IconType.CLOCK)
                
                # Update stop loss and take profit for hold recommendation
                try:
                    stop_loss = float(llm_analysis.get('stop_loss', 0))
                    take_profit = float(llm_analysis.get('take_profit', 0))
                    
                    self.log_formatted(pair, LogPhase.UPDATE_ORDER, "DEBUG", f"Trying to update SL/TP, values: SL={stop_loss}%, TP={take_profit}%")
                    self.log_formatted(pair, LogPhase.UPDATE_ORDER, "DEBUG", f"Pairs in custom_sl_tp: {list(self.custom_sl_tp.keys())}")
                    
                    # Check if pair exists in dictionary
                    if pair in self.custom_sl_tp:
                        old_sl = self.custom_sl_tp[pair]['stop_loss_pct']
                        old_tp = self.custom_sl_tp[pair]['take_profit_pct']
                        
                        # Update values
                        self.custom_sl_tp[pair]['stop_loss_pct'] = stop_loss
                        self.custom_sl_tp[pair]['take_profit_pct'] = take_profit
                        
                        self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", f"Updated SL/TP: SL {old_sl}% → {stop_loss}%, TP {old_tp}% → {take_profit}%", IconType.SUCCESS)
                    else:
                        # Initialize the pair if it doesn't exist
                        current_price = dataframe['close'].iloc[-1]
                        
                        # Store for use in other methods
                        self.custom_sl_tp[pair] = {
                            'stop_loss_pct': stop_loss,
                            'take_profit_pct': take_profit,
                            'entry_price': current_price,
                            'time_set': datetime.now()
                        }
                        
                        self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", f"Created new SL/TP: SL {stop_loss}%, TP {take_profit}% at price {current_price}", IconType.SUCCESS)
                except Exception as e:
                    self.log_formatted(pair, LogPhase.UPDATE_ORDER, "ERROR", f"Error updating SL/TP on hold: {e}", IconType.ERROR)
            elif recommendation == 'sell':
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"No buy - LLM recommends SELL with {confidence:.2f} confidence", IconType.SELL)
            else:
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"No buy - confidence too low: {confidence:.2f} < 0.6", IconType.WARNING)
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate sell signals based on LLM recommendations"""
        pair = metadata['pair']
        
        # Initialize with no sell signals
        dataframe['exit_long'] = 0
        
        # Skip if in backtest/hyperopt or not processing new candles
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"Skipping LLM exit signals - running in {self.dp.runmode.value} mode")
            return dataframe
            
        if not self.process_only_new_candles:
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Skipping LLM exit signals - not processing new candles")
            return dataframe
            
        # Get LLM analysis
        self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Checking for LLM analysis for exit", IconType.EYES)
        llm_analysis = getattr(self, 'llm_analyses', {}).get(pair)
        if not llm_analysis:
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "No LLM analysis available for exit", IconType.INFO)
            return dataframe
            
        # Check recommendation and confidence
        self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Evaluating exit recommendation", IconType.BRAIN)
        recommendation = llm_analysis.get('recommendation', '').lower()
        confidence = float(llm_analysis.get('confidence', 0))
        
        self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"LLM exit recommendation: {recommendation.upper()} with {confidence:.2f} confidence")
        
        # Generate sell signal if LLM recommends with sufficient confidence
        if recommendation == 'sell' and confidence >= 0.6:
            # Check if we have an open trade for this pair
            self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Checking for open trade", IconType.EYES)
            open_trades = Trade.get_trades_proxy(is_open=True)
            open_trade_found = False
            
            for trade in open_trades:
                if trade.pair == pair:
                    open_trade_found = True
                    # Set sell signal on last candle
                    self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "Generating SELL signal", IconType.SELL)
                    dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                    self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"SELL SIGNAL GENERATED with {confidence:.2f} confidence", IconType.SUCCESS)
                    
                    # Log trade details
                    current_price = dataframe['close'].iloc[-1]
                    profit = ((current_price / trade.open_rate) - 1) * 100
                    self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"Entry: {trade.open_rate}, Current: {current_price}, Profit: {profit:.2f}%", IconType.MONEY)
                    break
            
            if not open_trade_found:
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", "No open trade found - no sell signal generated", IconType.INFO)
        else:
            if recommendation == 'hold':
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"HOLDING based on LLM exit recommendation with {confidence:.2f} confidence", IconType.CLOCK)
                
                # Update stop loss and take profit for hold recommendation
                try:
                    stop_loss = float(llm_analysis.get('stop_loss', 0))
                    take_profit = float(llm_analysis.get('take_profit', 0))
                    
                    self.log_formatted(pair, LogPhase.UPDATE_ORDER, "DEBUG", f"Trying to update SL/TP, values: SL={stop_loss}%, TP={take_profit}%")
                    
                    # Check if pair exists in dictionary
                    if pair in self.custom_sl_tp:
                        old_sl = self.custom_sl_tp[pair]['stop_loss_pct']
                        old_tp = self.custom_sl_tp[pair]['take_profit_pct']
                        
                        # Update values
                        self.custom_sl_tp[pair]['stop_loss_pct'] = stop_loss
                        self.custom_sl_tp[pair]['take_profit_pct'] = take_profit
                        
                        self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", f"Updated SL/TP: SL {old_sl}% → {stop_loss}%, TP {old_tp}% → {take_profit}%", IconType.SUCCESS)
                    else:
                        # Initialize the pair if it doesn't exist
                        current_price = dataframe['close'].iloc[-1]
                        
                        # Store for use in other methods
                        self.custom_sl_tp[pair] = {
                            'stop_loss_pct': stop_loss,
                            'take_profit_pct': take_profit,
                            'entry_price': current_price,
                            'time_set': datetime.now()
                        }
                        
                        self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", f"Created new SL/TP: SL {stop_loss}%, TP {take_profit}% at price {current_price}", IconType.SUCCESS)
                except Exception as e:
                    self.log_formatted(pair, LogPhase.UPDATE_ORDER, "ERROR", f"Error updating SL/TP on hold: {e}", IconType.ERROR)
            elif recommendation == 'buy':
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"No sell - LLM recommends BUY with {confidence:.2f} confidence", IconType.BUY)
            else:
                self.log_formatted(pair, LogPhase.SIGNAL, "INFO", f"No sell - confidence too low: {confidence:.2f} < 0.6", IconType.WARNING)
        
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> bool:
        """Check for take profit level from LLM recommendation"""
        if pair in self.custom_sl_tp:
            tp_percentage = self.custom_sl_tp[pair]['take_profit_pct'] / 100
            entry_price = self.custom_sl_tp[pair]['entry_price']
            tp_price = entry_price * (1 + tp_percentage)
            
            # Calculate distance to target
            progress = min(100, max(0, (current_profit / tp_percentage) * 100)) if tp_percentage > 0 else 0
            
            # Create a visual progress bar
            bar_length = 20
            progress_chars = int((progress / 100) * bar_length)
            progress_bar = "=" * progress_chars + " " * (bar_length - progress_chars)
            
            # Choose icon based on progress
            icon = IconType.EYES
            if progress >= 90:
                icon = IconType.FIRE  # Almost at target
            elif progress >= 70:
                icon = IconType.ROCKET  # Making good progress
            elif progress >= 50:
                icon = IconType.CHART  # Halfway there
            elif progress >= 30:
                icon = IconType.MONEY  # Some progress
            elif progress <= 0:
                icon = IconType.DOWN  # Moving away from target
                
            # Format the log message
            progress_msg = f" [{progress_bar}] {progress:.0f}% {current_rate:.4f} → {tp_price:.4f} (profit: {current_profit:.2%}, target: {tp_percentage:.2%})"
            self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", progress_msg, icon)
            
            # Return True if we've hit the take profit level
            if current_profit >= tp_percentage:
                self.log_formatted(pair, LogPhase.UPDATE_ORDER, "INFO", f"TAKE PROFIT TRIGGERED at {current_profit:.2%}", IconType.SUCCESS)
                return True
                
        return False
