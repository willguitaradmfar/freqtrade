# Freqtrade Strategy Utilities

This directory contains utility modules for use with Freqtrade strategies.

## LLM Client Module

The `llm_client.py` module provides functionality to interact with OpenAI's language models (LLMs) for enhanced trading strategies.

### Features

- Send text and image messages to OpenAI models
- Generate structured JSON responses for trading decisions
- Support for multimodal inputs (text + images)
- Error handling and logging
- Compatible with various OpenAI models

### Installation

Install the required package:

```bash
pip install openai>=1.0.0
```

### Usage

In your strategy, import and use the module:

```python
from utils.llm_client import LLMClient

# Initialize the client with your API key
llm_client = LLMClient(api_key="your_openai_api_key")

# Example: Send messages to get market analysis
response = llm_client.send_message(
    messages=[
        {"role": "system", "text": "You are a trading assistant."},
        {"role": "user", "text": "Analyze this market data", "url_image": "https://example.com/chart.jpg"}
    ],
    model="gpt-4-vision-preview",
    response_format={"type": "json_object"}
)

# Check if successful and use the response
if response["success"]:
    analysis = response["content"]
    print(f"Market trend: {analysis['trend']}")
    print(f"Recommendation: {analysis['recommendation']}")
```

### Parameters

#### LLMClient.__init__
- `api_key`: OpenAI API key (optional, can also use OPENAI_API_KEY environment variable)

#### LLMClient.send_message
- `messages`: List of message objects with:
  - `role`: "system" or "user"
  - `text`: The message content
  - `url_image`: (Optional) URL to an image
- `model`: OpenAI model to use (default: "gpt-4-turbo")
- `temperature`: Controls randomness (0-1, default: 0.7)
- `max_tokens`: Maximum response length (default: 1024)
- `response_format`: Optional format specification, e.g. {"type": "json_object"}

### Return Value

The function returns a dictionary with:
- `success`: Boolean indicating if the call was successful
- `model`: The model used for the response
- `content`: Parsed JSON content (if requested) or raw text
- `raw_response`: The original text response
- `tokens`: Token usage information
- `error`: Error message if unsuccessful

## Plot Candles Module

The `plot_candles.py` module provides functionality to generate candlestick charts with indicators for Freqtrade strategies.

### Features

- Static candlestick charts with mplfinance
- Optional interactive charts with Plotly
- Support for multiple indicators with customizable:
  - Colors
  - Line width/thickness
  - Panel placement
- Automatic legends showing indicator names
- Volume subplot with colored bars matching candle direction (green for bullish, red for bearish)
  - Bars have proper spacing/margins between them for better readability
- Customizable output formats (PNG, JPG, SVG, PDF)

### Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

In your strategy, import and use the module:

```python
from utils.plot_candles import plot_last_candles

# Example usage in a strategy's populate_indicators method
result = plot_last_candles(
    pair=pair,
    dataframe=dataframe,
    timeframe=self.timeframe,
    num_candles=30,
    output_dir=absolute_plot_dir,
    indicators=[
        {'name': 'fast_ma', 'color': 'red', 'panel': 0, 'width': 2.0},
        {'name': 'slow_ma', 'color': 'blue', 'panel': 0, 'width': 1.5},
    ],
    indicators_below=[
        {'name': 'rsi', 'color': 'purple', 'width': 1.0},
    ],
    volume_spacing='sparse'  # Para maior separação entre barras de volume
)
```

### Parameters

- `pair`: Trading pair name (e.g., 'BTC/USDT')
- `dataframe`: DataFrame containing OHLCV data
- `timeframe`: Timeframe of the data (e.g., '5m', '1h', '1d')
- `num_candles`: Number of candles to plot (default: 30)
- `output_dir`: Directory to save the plot (default: 'user_data/plot')
- `indicators`: List of indicators to plot on main price chart, each as a dict with:
  - `name`: Column name in dataframe (required) - also used as legend label
  - `color`: Line color (required)
  - `panel`: Panel index (0 = main chart, optional, default: 0)
  - `width`: Line width/thickness (optional, default: 1.5)
- `indicators_below`: List of indicators to plot in separate panels below, each as a dict with:
  - `name`: Column name in dataframe (required) - also used as panel title
  - `color`: Line color (required)
  - `width`: Line width/thickness (optional, default: 1.5)
- `save_format`: Format to save the image ('png', 'jpg', 'svg', 'pdf')
- `use_plotly`: Use plotly for interactive charts if True, otherwise use mplfinance (default: False)
- `save_html`: When using plotly, also save an interactive HTML file (default: False)
- `volume_spacing`: Control spacing between volume bars (default: 'auto')
  - 'auto': Automatic spacing based on number of candles
  - 'sparse': Maximum spacing between bars
  - 'none': Minimum spacing between bars

### Output

The function generates and saves:
1. A static image file (PNG by default)
2. An interactive HTML file (only when using plotly with save_html=True)

The files are saved in the specified output directory with a filename pattern:
`{pair}_{timeframe}_{timestamp}.{format}`

## Available Utilities

### LLM Client
The `LLMClient` class provides integration with OpenAI's API for trading analysis. It allows you to send messages to OpenAI's models and parse the responses.

### Plot Candles
The `plot_last_candles` function allows you to generate charts with various indicators.

### Discord Webhook Integration
The `DiscordWebhook` class provides integration with Discord's webhook API, allowing you to:

- Send simple messages to Discord channels
- Post trading analysis with chart images and LLM-generated insights
- Customize username and avatar for each message

#### Usage Example

```python
from freqtrade_project.strategies.utils import DiscordWebhook

# Initialize the webhook client
webhook = DiscordWebhook(webhook_url="https://discord.com/api/webhooks/your-webhook-url")

# Send a simple message
webhook.send_message(
    content="Hello from FreqTrade!",
    username="Trading Bot"
)

# Send trading analysis with images
webhook.send_analysis(
    pair="BTC/USDT",
    timeframe="1h",
    analysis_json_path="path/to/analysis.json",
    image_paths=["path/to/chart1.png", "path/to/chart2.png"],
    username="FreqTrade Analysis"
)
```

#### Configuration
You can set the webhook URL either directly in the constructor or via the `DISCORD_WEBHOOK_URL` environment variable:

```bash
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-webhook-url"
```

You can also configure the webhook URL in your FreqTrade strategy config file:

```json
{
  "max_open_trades": 5,
  "stake_currency": "USDT",
  "stake_amount": 100,
  "tradable_balance_ratio": 0.99,
  "fiat_display_currency": "USD",
  "dry_run": true,
  "cancel_open_orders_on_exit": false,
  "unfilledtimeout": {
    "entry": 10,
    "exit": 10,
    "unit": "minutes"
  },
  "bid_strategy": {
    "price_side": "ask",
    "ask_last_balance": 0.0,
    "use_order_book": false,
    "order_book_top": 1,
    "check_depth_of_market": {
      "enabled": false,
      "bids_to_ask_delta": 1
    }
  },
  "ask_strategy": {
    "price_side": "bid",
    "bid_last_balance": 0.0,
    "use_order_book": false,
    "order_book_top": 1
  },
  
  /* Discord Integration Configuration */
  "discord_webhook_url": "https://discord.com/api/webhooks/your-webhook-url",
  
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": "",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": [
      "BTC/USDT",
      "ETH/USDT",
      "BNB/USDT"
    ],
    "pair_blacklist": []
  },
  "pairlists": [
    {"method": "StaticPairList"}
  ],
  "telegram": {
    "enabled": false,
    "token": "",
    "chat_id": ""
  }
} 