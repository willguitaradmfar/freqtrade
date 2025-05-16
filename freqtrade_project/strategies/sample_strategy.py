# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
from pandas import DataFrame
import sys
import os
import importlib.util
import logging
import numpy as np
from datetime import datetime

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
)
from freqtrade.vendor.qtpylib.indicators import crossed_above, crossed_below

# TA imports
import talib.abstract as ta

# Configurar logger
logger = logging.getLogger(__name__)

# Importar o módulo de plotagem de forma segura
PLOT_CANDLES_AVAILABLE = False
try:
    # Adicionar o diretório do módulo ao path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    utils_dir = os.path.join(current_dir, 'utils')
    
    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    
    # Verificar se o módulo existe antes de importar
    if os.path.exists(os.path.join(utils_dir, 'plot_candles.py')):
        import plot_candles
        PLOT_CANDLES_AVAILABLE = True
        logger.info("Módulo plot_candles importado com sucesso!")
    else:
        logger.warning("Arquivo plot_candles.py não encontrado no diretório utils")
except Exception as e:
    logger.error(f"Erro ao importar módulo plot_candles: {e}")

class SampleStrategy(IStrategy):
    """
    Simple moving average crossover strategy for FreqTrade
    """
    INTERFACE_VERSION = 3
    can_short: bool = False

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.05  # 5% profit target
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.10  # 10% stop loss

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptimizable parameters for moving averages
    fast_ma_period = IntParameter(low=5, high=20, default=10, space="buy", optimize=True, load=True)
    slow_ma_period = IntParameter(low=20, high=50, default=30, space="buy", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    # Diretório para salvar as imagens dos candles
    candle_plot_dir = "user_data/plot"

    def __init__(self, config: dict) -> None:
        """
        Inicializa a estratégia
        """
        super().__init__(config)
        # Verificar e criar o diretório para os plot
        os.makedirs(os.path.join(config['user_data_dir'], 'plot'), exist_ok=True)
        logger.info(f"Estratégia {self.__class__.__name__} inicializada")
        logger.info(f"Diretório de plot configurado: {os.path.join(config['user_data_dir'], 'plot')}")
        if PLOT_CANDLES_AVAILABLE:
            logger.info("Módulo de plotagem disponível - candles serão salvos")
        else:
            logger.warning("Módulo de plotagem não está disponível - candles não serão salvos")

    def calculate_vwap(self, dataframe, window=14):
        """
        Calcula o Volume Weighted Average Price (VWAP)
        """
        # Típico é (high + low + close) / 3, mas também pode-se usar close
        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        # VWAP é o preço típico ponderado pelo volume
        vwap = (typical_price * dataframe['volume']).rolling(window=window).sum() / dataframe['volume'].rolling(window=window).sum()
        return vwap

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate moving averages
        """
        # Moving Averages
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=self.fast_ma_period.value)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=self.slow_ma_period.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # VWAP - implementação personalizada
        dataframe['vwap'] = self.calculate_vwap(dataframe, window=14)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']
        
        # Cálculo correto do MACD
        macd, macdsignal, macdhist = ta.MACD(dataframe['close'], 
                                            fastperiod=12, 
                                            slowperiod=26, 
                                            signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macd_signal'] = macdsignal
        dataframe['macd_hist'] = macdhist
        
        # Log para debug
        logger.info(f"MACD criado com sucesso: {len(macd)} pontos")

        # Plotar gráfico dos últimos 10 candles quando um novo candle é processado
        if self.process_only_new_candles and not dataframe.empty and PLOT_CANDLES_AVAILABLE:
            try:
                pair = metadata['pair']
                logger.info(f"Tentando plotar candles para {pair}")
                
                # Criar diretório absoluto para salvar os plot
                absolute_plot_dir = os.path.join(self.config['user_data_dir'], 'plot')
                
                # Chamar a função de plotagem
                result = plot_candles.plot_last_candles(
                    pair=pair,
                    dataframe=dataframe,
                    timeframe=self.timeframe,
                    num_candles=50,
                    output_dir=absolute_plot_dir,
                    indicators=[
                        {'name': 'fast_ma', 'color': 'red', 'panel': 0, 'width': 2.0, 'type': 'line'},
                        {'name': 'slow_ma', 'color': 'blue', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'vwap', 'color': 'green', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'bb_upper', 'color': 'orange', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'bb_middle', 'color': 'yellow', 'panel': 0, 'width': 1.5, 'type': 'line'},
                        {'name': 'bb_lower', 'color': 'purple', 'panel': 0, 'width': 1.5, 'type': 'line'},
                    ],
                    indicators_below=[
                        {'name': 'rsi', 'color': 'purple', 'width': 1.0, 'type': 'line'},
                        {'name': 'macd', 'color': 'blue', 'width': 1.5, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macd_signal', 'color': 'red', 'width': 1.0, 'panel': 'MACD', 'type': 'line'},
                        {'name': 'macd_hist', 'color': 'green', 'width': 0.8, 'panel': 'MACD', 'type': 'bar'},
                    ],
                    volume_spacing='none',  # Maior espaçamento entre barras de volume
                    title=f"Análise Técnica - {pair}",
                    subtitle=f"Timeframe: {self.timeframe} - Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                )
                
                if result:
                    logger.info(f"Candles plotados com sucesso para {pair}: {result}")
                else:
                    logger.warning(f"Não foi possível plotar candles para {pair}")
            except Exception as e:
                logger.error(f"Erro ao plotar candles para {pair}: {e}", exc_info=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal: fast MA crosses above slow MA
        """
        dataframe.loc[
            (
                crossed_above(dataframe['fast_ma'], dataframe['slow_ma']) &
                (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal: fast MA crosses below slow MA
        """
        dataframe.loc[
            (
                crossed_below(dataframe['fast_ma'], dataframe['slow_ma']) &
                (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        return dataframe
