# FreqTrade Project - Modular Trading Bot

A modular project structure for FreqTrade, a free and open-source cryptocurrency trading bot.

## Overview

This project provides a structured approach to developing cryptocurrency trading strategies using FreqTrade. It separates concerns into different modules, making it easier to maintain, test, and develop new strategies.

## Features

- **Modular Structure**: Well-organized code with separation of concerns
- **Base Strategy**: Common functionality in a base class that other strategies inherit from
- **Custom Indicators**: Pre-built collection of technical indicators
- **Multiple Strategy Examples**:
  - SimpleMAStrategy: A simple strategy based on Moving Average crossovers
  - IchimokuStrategy: A more complex strategy using Ichimoku Cloud with volume confirmation

## Getting Started

1. Install FreqTrade following the [official installation instructions](https://www.freqtrade.io/en/stable/installation/)
2. Clone this repository
3. Install the additional requirements
4. Configure your API keys in the config/config.json file
5. Run the bot

For detailed instructions, see the [project README](freqtrade_project/README.md).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/freqtrade_project.git
cd freqtrade_project

# Install the project
pip install -e .

# Download historical data for backtesting
freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT --timeframes 5m 15m 1h

# Run backtesting
freqtrade backtesting --strategy SimpleMAStrategy --strategy-path ./freqtrade_project/strategies --config ./freqtrade_project/config/config.json

# Start trading (dry-run)
freqtrade trade --strategy SimpleMAStrategy --strategy-path ./freqtrade_project/strategies --config ./freqtrade_project/config/config.json
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is free to use and modify.

## Disclaimer

Trading cryptocurrencies carries significant risk. Use this software at your own risk. The author(s) take no responsibility for financial losses incurred using this software. 