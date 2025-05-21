# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
from pandas import DataFrame
import sys
import os
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Any

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

# TA imports
import talib.abstract as ta

# Configure logger
logger = logging.getLogger(__name__)

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
        "30": 0.025,
        "60": 0.015,
        "90": 0.01
    }
    # stoploss is a percentage of the entry price
    # 0.02 is 2%
    stoploss = -0.02
    # trailing_stop_positive is a percentage of the entry price
    # 0.01 is 1%
    trailing_stop_positive = 0.01

    # No trailing stop by default
    trailing_stop = True

    # Trading timeframe
    timeframe = "30m"

    # Only process new candles
    process_only_new_candles = True

    # Exit settings
    use_exit_signal = True    
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles required
    startup_candle_count = 50

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
    llm_max_tokens = 1024

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate indicators and generate chart for analysis"""
        pair = metadata['pair']
        
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
        
        # Generate chart image for LLM analysis
        chart_emas_result = None
        chart_bollinger_result = None
        if self.process_only_new_candles and not dataframe.empty and PLOT_CANDLES_AVAILABLE:
            try:
                logger.info(f"Generating chart for {pair}")
                
                # Plot candles with indicators
                chart_emas_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='emas',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=50,
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

                chart_bollinger_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='bollinger',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=50,
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

                chart_vwap_result = plot_candles.plot_last_candles(
                    pair=pair,
                    sulfix_filename='vwap',
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=50,
                    output_dir=self.absolute_plot_dir,
                    indicators=[
                        {'name': 'vwap', 'color': 'blue', 'panel': 0, 'width': 1.5, 'type': 'line'},
                    ],
                    indicators_below=[
                        {'name': 'rsi', 'color': 'purple', 'width': 1.0, 'type': 'line'},
                        {'name': 'macd', 'color': 'blue', 'width': 1.5, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macdsignal', 'color': 'red', 'width': 1.0, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macdhist', 'color': 'green', 'width': 0.8, 'panel': 'MACD', 'type': 'bar'},
                    ],
                    title=f"{pair} - {self.timeframe}",
                    subtitle=f"Volume Weighted Average Price (VWAP) 20",
                    question=f"What is the trend for {pair} in the {self.timeframe} timeframe? What is the trend for the VWAP?"
                )
                
                # Get LLM analysis if charts were created and client is available
                if (chart_emas_result or chart_bollinger_result or chart_vwap_result) and self.llm_client and LLM_CLIENT_AVAILABLE:
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
                            logger.info(f"EMA chart generated: {chart_emas_result['filepath']}")
                            
                    if chart_bollinger_result:
                        if chart_bollinger_result.get('filepath') and os.path.exists(chart_bollinger_result['filepath']):
                            chart_images.append(chart_bollinger_result['filepath'])
                            if chart_bollinger_result.get('base64'):
                                chart_base64_images.append({
                                    "path": chart_bollinger_result['filepath'],
                                    "base64": chart_bollinger_result['base64'],
                                    "question": chart_bollinger_result['question']
                                })
                            logger.info(f"Bollinger chart generated: {chart_bollinger_result['filepath']}")

                    if chart_vwap_result:
                        if chart_vwap_result.get('filepath') and os.path.exists(chart_vwap_result['filepath']):
                            chart_images.append(chart_vwap_result['filepath'])
                            if chart_vwap_result.get('base64'):
                                chart_base64_images.append({
                                    "path": chart_vwap_result['filepath'],
                                    "base64": chart_vwap_result['base64'],
                                    "question": chart_vwap_result['question']
                                })
                            logger.info(f"VWAP chart generated: {chart_vwap_result['filepath']}")
                                
                    if chart_images and chart_base64_images:
                        llm_analysis = self.analyze_chart_with_llm(chart_images, chart_base64_images, pair, dataframe)
                        if llm_analysis:
                            # Store analysis for use in signal generation
                            if not hasattr(self, 'llm_analyses'):
                                self.llm_analyses = {}
                            self.llm_analyses[pair] = llm_analysis
                            
                            logger.info(f"LLM analysis for {pair}: {llm_analysis['recommendation']} (confidence: {llm_analysis['confidence']})")
                    else:
                        logger.error(f"No chart images available for {pair} to analyze with LLM")
            except Exception as e:
                logger.error(f"Error in chart generation: {e}", exc_info=True)
                
        return dataframe

    def analyze_chart_with_llm(self, image_paths: List[str], image_base64_list: List[Dict[str, str]], pair: str, dataframe: DataFrame) -> Optional[Dict[str, Any]]:
        """Send chart images to LLM for analysis"""
        if not self.llm_client or not all(os.path.exists(path) for path in image_paths):
            missing_paths = [path for path in image_paths if not os.path.exists(path)]
            logger.error(f"Cannot analyze chart: {'LLM client not available' if not self.llm_client else f'Images not found: {missing_paths}'}")
            return None
            
        try:
            logger.info(f"🔄 STEP 1: Starting LLM analysis for {pair} with {len(image_paths)} images")
            
            # No need to encode images, as we already have the base64 from plot_candles
            logger.info(f"🔄 STEP 2: Using pre-encoded images for {len(image_base64_list)} charts")
            
            # Recent price data for context
            logger.info(f"🔄 STEP 3: Preparing price context for {pair}")
            recent_prices = dataframe.tail(5)[['date', 'close']].to_dict('records')
            price_context = ", ".join([f"{row['date']}: {row['close']:.2f}" for row in recent_prices])
            logger.info(f"📊 Price context: {price_context}")
            
            # Get open trade information for this pair
            logger.info(f"🔄 STEP 4: Getting open trade information for {pair}")
            trade_info = self._get_open_trade_info(pair)
            logger.info(f"📊 Trade info: {trade_info}")
            
            # System message for LLM
            logger.info(f"🔄 STEP 5: Creating system message for LLM")
            system_content = (
                "You are a cryptocurrency trading assistant. Analyze the chart images and provide "
                "professional technical analysis. You must analyze all indicators, candlestick patterns, and market structures. "
                "Pay close attention to open orders and positions, profit levels, and make clear decisions on whether to sell or continue holding. "
                "Your response must be a JSON object with the following fields: "
                "\"analysis\": a brief explanation of your analysis with key indicators and patterns, "
                "\"trend\": \"bullish\", \"bearish\", or \"neutral\", "
                "\"confidence\": a number between 0 and 1, "
                "\"recommendation\": \"buy\", \"sell\", or \"hold\", "
                "\"stop_loss\": a suggested stop-loss percentage (negative value), "
                "\"take_profit\": a suggested take-profit percentage (positive value)"
            )
            
            # User message with chart and trade context
            logger.info(f"🔄 STEP 6: Creating user message with charts for {pair}")
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
                
            logger.info(f"📝 User message prepared: {user_content}")
            
            # Format messages for OpenAI API with multiple images
            logger.info(f"🔄 STEP 7: Formatting messages with {len(image_base64_list)} images for LLM API")
            
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
            logger.info(f"📤 REQUEST TO LLM:")
            logger.info(f"📤 System content: {system_content}")
            logger.info(f"📤 User content: {user_content}")
            logger.info(f"📤 Including {len(image_base64_list)} images")
            logger.info(f"📤 Model: {self.llm_model}, Temperature: {self.llm_temperature}")
            
            # Send to LLM
            logger.info(f"🔄 STEP 8: Sending request to LLM with model {self.llm_model}")
            response = self.llm_client.send_message(
                messages=messages,
                model=self.llm_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Process response
            logger.info(f"🔄 STEP 9: Processing LLM response")
            if not response["success"]:
                logger.error(f"❌ LLM request failed: {response.get('error', 'Unknown error')}")
                return None
                
            if not isinstance(response["content"], dict):
                logger.error(f"❌ LLM response is not a valid dictionary: {response['content']}")
                return None
                
            content = response["content"]
            logger.info(f"✅ Received LLM response: {content}")
            
            # Validate required fields
            required_fields = ["trend", "confidence", "recommendation", "stop_loss", "take_profit"]
            missing_fields = [field for field in required_fields if field not in content]
            
            if missing_fields:
                logger.error(f"❌ LLM response missing required fields: {missing_fields}")
                return None
            
            # Save LLM analysis to JSON file with same name as the first chart image but with .json extension
            logger.info(f"🔄 STEP 10: Saving LLM analysis to JSON file")
            try:
                # Create JSON file path from the first image path
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
                json_file_path = os.path.join(os.path.dirname(image_paths[0]), f"{pair.replace('/', '_')}_{self.timeframe}_{current_datetime}.json")
                logger.info(f"📄 Saving analysis to: {json_file_path}")
                
                # Create a complete record with both request and response
                json_data = {
                    "request": messages,
                    "response": content
                }
                
                # Write the JSON file
                with open(json_file_path, 'w') as json_file:
                    json.dump(json_data, json_file, indent=4)
                    
                logger.info(f"✅ Analysis saved to JSON file: {json_file_path}")
                
                # STEP 11: Send analysis to Discord if webhook URL is configured
                logger.info(f"🔄 STEP 11: Sending analysis to Discord")
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
                            logger.info(f"✅ Analysis sent to Discord successfully")
                        else:
                            logger.warning(f"⚠️ Failed to send analysis to Discord: {result.get('error', 'Unknown error')}")
                    elif not discord_webhook_url:
                        logger.info("ℹ️ Discord webhook URL not configured, skipping Discord integration")
                    elif not DISCORD_WEBHOOK_AVAILABLE:
                        logger.warning("⚠️ Discord webhook module not available, skipping Discord integration")
                        
                except Exception as e:
                    logger.error(f"❌ Error sending analysis to Discord: {e}", exc_info=True)
                    # Continue even if Discord sending fails
            except Exception as e:
                logger.error(f"❌ Error saving LLM analysis to JSON file: {e}")
                # Continue even if saving fails
            
            # Log detailed analysis
            logger.info(f"📈 LLM ANALYSIS FOR {pair}:")
            logger.info(f"   Analysis: {content.get('analysis', 'No analysis provided')}")
            logger.info(f"   Trend: {content.get('trend')}")
            logger.info(f"   Confidence: {content.get('confidence')}")
            logger.info(f"   Recommendation: {content.get('recommendation')}")
            logger.info(f"   Stop Loss: {content.get('stop_loss')}%")
            logger.info(f"   Take Profit: {content.get('take_profit')}%")
            logger.info(f"✅ STEP 12: Analysis validated and complete for {pair}")
                
            return content
                
        except Exception as e:
            logger.error(f"❌ Error in LLM analysis: {e}", exc_info=True)
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
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting open trade info for {pair}: {e}")
            return None

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate buy signals based on LLM recommendations"""
        pair = metadata['pair']
        
        # Initialize with no buy signals
        dataframe['enter_long'] = 0
        
        # Skip if in backtest/hyperopt or not processing new candles
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            logger.info(f"Skipping LLM entry signals for {pair} - running in {self.dp.runmode.value} mode")
            return dataframe
            
        if not self.process_only_new_candles:
            logger.info(f"Skipping LLM entry signals for {pair} - not processing new candles")
            return dataframe
            
        # Get LLM analysis
        logger.info(f"ud83dudd04 ENTRY STEP 1: Checking for LLM analysis for {pair}")
        llm_analysis = getattr(self, 'llm_analyses', {}).get(pair)
        if not llm_analysis:
            logger.info(f"u274c No LLM analysis available for {pair}")
            return dataframe
            
        # Check recommendation and confidence
        logger.info(f"ud83dudd04 ENTRY STEP 2: Evaluating recommendation for {pair}")
        recommendation = llm_analysis.get('recommendation', '').lower()
        confidence = float(llm_analysis.get('confidence', 0))
        
        logger.info(f"ud83dudcc8 {pair} LLM recommendation: {recommendation.upper()} with {confidence:.2f} confidence")
        
        # Generate buy signal if LLM recommends with sufficient confidence
        if recommendation == 'buy' and confidence >= 0.6:
            # Set buy signal on last candle
            logger.info(f"ud83dudd04 ENTRY STEP 3: Generating BUY signal for {pair}")
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
            logger.info(f"u2705 BUY SIGNAL GENERATED for {pair} with {confidence:.2f} confidence")
            
            # Store custom stop loss and take profit
            logger.info(f"ud83dudd04 ENTRY STEP 4: Setting stop loss and take profit for {pair}")
            try:
                stop_loss = float(llm_analysis.get('stop_loss', 0))
                take_profit = float(llm_analysis.get('take_profit', 0))
                
                logger.info(f"   Raw SL: {stop_loss}%, TP: {take_profit}%")
                
                # Use reasonable values
                if -15 <= stop_loss <= -0.5 and 1 <= take_profit <= 20:
                    current_price = dataframe['close'].iloc[-1]
                    
                    # Store for use in other methods
                    self.custom_sl_tp[pair] = {
                        'stop_loss_pct': stop_loss,  # Already negative
                        'take_profit_pct': take_profit,
                        'entry_price': current_price,
                        'time_set': datetime.now()
                    }
                    
                    logger.info(f"u2705 Set SL: {stop_loss}%, TP: {take_profit}% for {pair} at price {current_price}")
                else:
                    logger.warning(f"u26a0ufe0f Invalid SL/TP values: SL={stop_loss}%, TP={take_profit}% - using defaults")
            except Exception as e:
                logger.error(f"u274c Error setting SL/TP: {e}", exc_info=True)
        else:
            if recommendation == 'hold':
                logger.info(f"u2139ufe0f HOLDING {pair} based on LLM recommendation with {confidence:.2f} confidence")
            elif recommendation == 'sell':
                logger.info(f"u2139ufe0f No buy for {pair} - LLM recommends SELL with {confidence:.2f} confidence")
            else:
                logger.info(f"u2139ufe0f No buy for {pair} - confidence too low: {confidence:.2f} < 0.6")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate sell signals based on LLM recommendations"""
        pair = metadata['pair']
        
        # Initialize with no sell signals
        dataframe['exit_long'] = 0
        
        # Skip if in backtest/hyperopt or not processing new candles
        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            logger.info(f"Skipping LLM exit signals for {pair} - running in {self.dp.runmode.value} mode")
            return dataframe
            
        if not self.process_only_new_candles:
            logger.info(f"Skipping LLM exit signals for {pair} - not processing new candles")
            return dataframe
            
        # Get LLM analysis
        logger.info(f"ud83dudd04 EXIT STEP 1: Checking for LLM analysis for {pair}")
        llm_analysis = getattr(self, 'llm_analyses', {}).get(pair)
        if not llm_analysis:
            logger.info(f"u274c No LLM analysis available for {pair}")
            return dataframe
            
        # Check recommendation and confidence
        logger.info(f"ud83dudd04 EXIT STEP 2: Evaluating exit recommendation for {pair}")
        recommendation = llm_analysis.get('recommendation', '').lower()
        confidence = float(llm_analysis.get('confidence', 0))
        
        logger.info(f"ud83dudcc8 {pair} LLM exit recommendation: {recommendation.upper()} with {confidence:.2f} confidence")
        
        # Generate sell signal if LLM recommends with sufficient confidence
        if recommendation == 'sell' and confidence >= 0.6:
            # Check if we have an open trade for this pair
            logger.info(f"ud83dudd04 EXIT STEP 3: Checking for open trade for {pair}")
            open_trades = Trade.get_trades_proxy(is_open=True)
            open_trade_found = False
            
            for trade in open_trades:
                if trade.pair == pair:
                    open_trade_found = True
                    # Set sell signal on last candle
                    logger.info(f"ud83dudd04 EXIT STEP 4: Generating SELL signal for {pair}")
                    dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                    logger.info(f"u2705 SELL SIGNAL GENERATED for {pair} with {confidence:.2f} confidence")
                    
                    # Log trade details
                    current_price = dataframe['close'].iloc[-1]
                    profit = ((current_price / trade.open_rate) - 1) * 100
                    logger.info(f"   Entry: {trade.open_rate}, Current: {current_price}, Profit: {profit:.2f}%")
                    break
            
            if not open_trade_found:
                logger.info(f"u2139ufe0f No open trade found for {pair} - no sell signal generated")
        else:
            if recommendation == 'hold':
                logger.info(f"u2139ufe0f HOLDING {pair} based on LLM exit recommendation with {confidence:.2f} confidence")
            elif recommendation == 'buy':
                logger.info(f"u2139ufe0f No sell for {pair} - LLM recommends BUY with {confidence:.2f} confidence")
            else:
                logger.info(f"u2139ufe0f No sell for {pair} - confidence too low: {confidence:.2f} < 0.6")
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                      current_rate: float, current_profit: float, **kwargs) -> float:
        """Use stop loss from LLM recommendation"""
        logger.info(f"ud83dudd04 SL check for {pair}: Current profit {current_profit:.2%}")
        if pair in self.custom_sl_tp:
            sl_value = self.custom_sl_tp[pair]['stop_loss_pct']
            entry_price = self.custom_sl_tp[pair]['entry_price']
            sl_price = entry_price * (1 + (sl_value / 100))
            logger.info(f"u2139ufe0f Using LLM stop loss: {sl_value:.2f}% (price: {sl_price:.4f})")
            return sl_value
            
        logger.info(f"u2139ufe0f Using default stop loss: {self.stoploss}")
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> bool:
        """Check for take profit level from LLM recommendation"""
        logger.info(f"ud83dudd04 TP check for {pair}: Current profit {current_profit:.2%}")
        if pair in self.custom_sl_tp:
            tp_percentage = self.custom_sl_tp[pair]['take_profit_pct'] / 100
            entry_price = self.custom_sl_tp[pair]['entry_price']
            tp_price = entry_price * (1 + tp_percentage)
            
            logger.info(f"u2139ufe0f Take profit target: {tp_percentage:.2%} (price: {tp_price:.4f})")
            
            if current_profit >= tp_percentage:
                logger.info(f"u2705 TAKE PROFIT TRIGGERED for {pair} at {current_profit:.2%} (target: {tp_percentage:.2%})")
                return True
            else:
                distance_to_tp = ((tp_price / current_rate) - 1) * 100
                logger.info(f"u2139ufe0f {distance_to_tp:.2f}% away from take profit target")
                
        return False

    # def bot_loop_start(self, **kwargs) -> None:
    #     """Log open trade status at the start of each bot loop"""
    #     if self.dp.runmode.value in ('backtest', 'hyperopt'):
    #         return
            
    #     try:
    #         logger.info(f"ud83dudd04 MONITORING: Checking open trades status")
    #         open_trades = Trade.get_trades_proxy(is_open=True)
    #         if not open_trades:
    #             logger.info("ud83dudcc3 No open trades to monitor")
    #             return
                
    #         logger.info(f"ud83dudcc8 MONITORING {len(open_trades)} OPEN TRADES:")
            
    #         for trade in open_trades:
    #             # Get current price
    #             try:
    #                 current_price = self.dp.ticker(trade.pair)['last']
    #                 profit_pct = ((current_price / trade.open_rate) - 1) * 100
    #                 time_in_trade = (datetime.now() - trade.open_date).total_seconds() / 3600  # hours
                    
    #                 logger.info(f"ud83dudd39 {trade.pair}:")
    #                 logger.info(f"   Open for: {time_in_trade:.1f} hours")
    #                 logger.info(f"   Entry: {trade.open_rate:.4f}, Current: {current_price:.4f}")
    #                 logger.info(f"   Profit: {profit_pct:.2f}%")
                    
    #                 if trade.pair in self.custom_sl_tp:
    #                     sl_pct = self.custom_sl_tp[trade.pair]['stop_loss_pct']
    #                     tp_pct = self.custom_sl_tp[trade.pair]['take_profit_pct']
    #                     entry_price = self.custom_sl_tp[trade.pair]['entry_price']
                        
    #                     # Calculate distances
    #                     sl_price = entry_price * (1 + (sl_pct / 100))
    #                     tp_price = entry_price * (1 + (tp_pct / 100))
                        
    #                     sl_distance = ((current_price / sl_price) - 1) * 100
    #                     tp_distance = ((tp_price / current_price) - 1) * 100
                        
    #                     logger.info(f"   SL: {sl_pct:.2f}% ({sl_price:.4f}), {sl_distance:.2f}% away")
    #                     logger.info(f"   TP: {tp_pct:.2f}% ({tp_price:.4f}), {tp_distance:.2f}% away")
                        
    #                     # Risk assessment
    #                     if sl_distance < 1.0:
    #                         logger.warning(f"u26a0ufe0f {trade.pair} is CLOSE TO STOP LOSS (only {sl_distance:.2f}% away)")
    #                     if tp_distance < 1.0:
    #                         logger.info(f"ud83dudd34 {trade.pair} is CLOSE TO TAKE PROFIT (only {tp_distance:.2f}% away)")
    #                 else:
    #                     logger.info(f"   No custom SL/TP defined - using strategy defaults")
                    
    #             except Exception as e:
    #                 logger.error(f"u274c Error getting price data for {trade.pair}: {e}")
            
    #         logger.info("ud83dudd04 MONITORING complete")
    #     except Exception as e:
    #         logger.error(f"u274c Error in bot_loop_start monitoring: {e}", exc_info=True)
