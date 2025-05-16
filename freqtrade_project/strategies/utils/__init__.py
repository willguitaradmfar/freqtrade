"""
Pacote de utilidades para estratégias do Freqtrade
"""

from .plot_candles import plot_last_candles
from .llm_client import LLMClient, encode_image_to_base64
from .discord_webhook import DiscordWebhook

__all__ = ['plot_last_candles', 'LLMClient', 'encode_image_to_base64', 'DiscordWebhook']
