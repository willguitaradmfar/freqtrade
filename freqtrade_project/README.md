# FreqTrade Project

A modular FreqTrade project structure for developing and running cryptocurrency trading strategies.

## Project Structure

```
freqtrade_project/
├── config/                    # Configuration files
│   └── config.json            # Main configuration
├── data/                      # Data directory for downloaded historical data
├── strategies/                # Trading strategies
│   ├── indicators/            # Custom indicators
│   │   ├── __init__.py        # Package initialization
│   │   ├── trend_indicators.py # Trend analysis indicators
│   │   └── volume_indicators.py # Volume analysis indicators
│   ├── __init__.py            # Strategies package initialization
│   ├── base_strategy.py       # Base strategy with common functionality
│   ├── simple_ma_crossover.py # Simple Moving Average strategy
│   └── ichimoku_strategy.py   # Ichimoku Cloud strategy
└── README.md                  # This file
```

## Prerequisites

Before running this project, you need to install FreqTrade:

```bash
# Clone FreqTrade repository
git clone https://github.com/freqtrade/freqtrade.git

# Enter the directory
cd freqtrade

# Install dependencies
./setup.sh -i
```

## Installation

1. Clone this repository or download it
2. Install the additional dependencies:

```bash
pip install numpy pandas ta-lib
```

## Usage

### Download Historical Data

Before running backtesting or live trading, you need to download historical data:

```bash
# Activate FreqTrade environment
cd freqtrade
source .env/bin/activate

# Download data
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT XRP/USDT SOL/USDT ADA/USDT BNB/USDT DOT/USDT DOGE/USDT --timeframes 5m 15m 1h 4h 1d
```

### Backtesting

```bash
# Run backtesting with SimpleMAStrategy
freqtrade backtesting --strategy SimpleMAStrategy --strategy-path /path/to/freqtrade_project/strategies --config /path/to/freqtrade_project/config/config.json

# Run backtesting with IchimokuStrategy
freqtrade backtesting --strategy IchimokuStrategy --strategy-path /path/to/freqtrade_project/strategies --config /path/to/freqtrade_project/config/config.json
```

### Hyperopt (Strategy Optimization)

```bash
# Optimize SimpleMAStrategy
freqtrade hyperopt --hyperopt-loss SharpeHyperOptLoss --strategy SimpleMAStrategy --strategy-path /path/to/freqtrade_project/strategies --config /path/to/freqtrade_project/config/config.json --epochs 100
```

### Start Trading Bot

```bash
# Run in dry-run mode (paper trading)
freqtrade trade --strategy SimpleMAStrategy --strategy-path /path/to/freqtrade_project/strategies --config /path/to/freqtrade_project/config/config.json

# Run with real trading (update config.json with your API keys first and set dry_run to false)
freqtrade trade --strategy SimpleMAStrategy --strategy-path /path/to/freqtrade_project/strategies --config /path/to/freqtrade_project/config/config.json
```

## Creating Your Own Strategies

To create a new strategy:

1. Create a new Python file in the `strategies` directory
2. Extend the `BaseStrategy` class
3. Implement at minimum the `populate_entry_trend` and `populate_exit_trend` methods
4. Update the `__init__.py` file to include your new strategy

Example:

```python
from freqtrade_project.strategies.base_strategy import BaseStrategy
from pandas import DataFrame

class MyCustomStrategy(BaseStrategy):
    # Define your strategy parameters here
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define your buy signals here
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define your sell signals here
        return dataframe
```

## Customizing Indicators

To add custom indicators:

1. Add your indicator functions to one of the existing files in the `indicators` directory or create a new file
2. Update the `__init__.py` file in the indicators directory to export your new indicator
3. Import and use your indicators in your strategies

## License

This project is free to use and modify. 