#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Discord webhook integration for freqtrade strategies
Allows posting analysis images and LLM reports to Discord channels
"""

import json
import logging
import os
import requests
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class DiscordWebhook:
    """
    Discord webhook client for sending messages, embeds, and files to Discord channels
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize the Discord webhook client
        
        :param webhook_url: Discord webhook URL, if None will try to get from DISCORD_WEBHOOK_URL env variable
        """
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            raise ValueError("Discord webhook URL is required. Provide it as a parameter or set DISCORD_WEBHOOK_URL environment variable")
        
        logger.info("Discord webhook client initialized")
    
    def send_message(self, 
                    content: str,
                    username: Optional[str] = "FreqTrade Bot",
                    avatar_url: Optional[str] = None,
                    embeds: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Send a simple message to Discord
        
        :param content: Message content (up to 2000 characters)
        :param username: Override the webhook's default username
        :param avatar_url: Override the webhook's default avatar
        :param embeds: List of embed objects
        
        :return: Dictionary with response details
        """
        if not content and not embeds:
            logger.error("No content or embeds to send")
            return {
                "success": False,
                "error": "No content or embeds to send",
                "response": None
            }
        
        payload = {
            "content": content[:2000] if content else "",  # Discord limit is 2000 chars
        }
        
        if username:
            payload["username"] = username
            
        if avatar_url:
            payload["avatar_url"] = avatar_url
            
        if embeds:
            payload["embeds"] = embeds
            
        try:
            response = requests.post(
                self.webhook_url,
                json=payload
            )
            
            if response.status_code == 204:
                logger.info("Message sent successfully to Discord")
                return {
                    "success": True,
                    "response": response
                }
            else:
                logger.error(f"Failed to send message to Discord: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response": response
                }
                
        except Exception as e:
            logger.error(f"Error sending message to Discord: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def send_analysis(self,
                     pair: str,
                     timeframe: str,
                     analysis_json_path: str,
                     image_paths: List[str],
                     username: Optional[str] = "FreqTrade Analysis",
                     avatar_url: Optional[str] = None,
                     open_trade_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send trading analysis with images to Discord
        
        :param pair: Trading pair (e.g., 'BTC/USDT')
        :param timeframe: Timeframe of the analysis (e.g., '1h', '4h', '1d')
        :param analysis_json_path: Path to the JSON file containing LLM analysis
        :param image_paths: List of paths to chart images
        :param username: Override the webhook's default username
        :param avatar_url: Override the webhook's default avatar
        :param open_trade_info: Information about any open trade for this pair (optional)
        
        :return: Dictionary with response details
        """
        # Make sure the JSON file exists
        if not os.path.exists(analysis_json_path):
            logger.error(f"Analysis file not found: {analysis_json_path}")
            return {
                "success": False,
                "error": f"Analysis file not found: {analysis_json_path}",
                "response": None
            }
            
        # Load the analysis JSON
        try:
            with open(analysis_json_path, 'r') as file:
                analysis_data = json.load(file)
        except Exception as e:
            logger.error(f"Error loading analysis JSON: {e}")
            return {
                "success": False,
                "error": f"Error loading analysis JSON: {e}",
                "response": None
            }
        
        # Check if images exist
        valid_image_paths = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                valid_image_paths.append(img_path)
            else:
                logger.warning(f"Image not found, skipping: {img_path}")
        
        if not valid_image_paths:
            logger.error("No valid images to send")
            return {
                "success": False,
                "error": "No valid images to send",
                "response": None
            }
        
        # Get the actual LLM response content from the JSON structure
        llm_content = None
        if isinstance(analysis_data, dict):
            if "response" in analysis_data:
                llm_content = analysis_data["response"]
            else:
                llm_content = analysis_data
        else:
            llm_content = analysis_data
            
        # Prepare the message content with improved formatting
        timestamp = Path(analysis_json_path).stem.split('_')[-2:]
        generation_time = f"{timestamp[0]} {timestamp[1]}" if len(timestamp) >= 2 else "Unknown"
        
        # Create a well-formatted Discord message with emojis and formatting
        content = f"# 📊 Trading Analysis: {pair} ({timeframe})\n\n"
        
        # Add open trade information if available
        if open_trade_info:
            entry_price = open_trade_info.get('entry_price', 0)
            current_price = open_trade_info.get('current_price', 0)
            profit_pct = open_trade_info.get('profit_pct', 0)
            time_in_trade = open_trade_info.get('time_in_trade', 0)
            stop_loss_pct = open_trade_info.get('stop_loss_pct', 0)
            take_profit_pct = open_trade_info.get('take_profit_pct', 0)
            
            # Choose profit emoji based on the current profit
            profit_emoji = "🟢" if profit_pct > 0 else "🔴" if profit_pct < 0 else "⚪"
            
            content += f"## 💰 Posição Aberta\n"
            content += f"Entrada: **{entry_price:.4f}**\n"
            content += f"Atual: **{current_price:.4f}**\n"
            content += f"Lucro/Prejuízo: {profit_emoji} **{profit_pct:.2f}%**\n"
            content += f"Tempo na posição: **{time_in_trade:.1f}h**\n"
            
            if stop_loss_pct or take_profit_pct:
                content += f"Stop Loss: **{stop_loss_pct:.2f}%** | Take Profit: **{take_profit_pct:.2f}%**\n"
                
                # Calculate distances to SL/TP if available
                if entry_price and current_price:
                    if stop_loss_pct:
                        sl_price = entry_price * (1 + (stop_loss_pct / 100))
                        sl_distance = ((current_price / sl_price) - 1) * 100
                        content += f"Distância ao SL: **{sl_distance:.2f}%**\n"
                        
                    if take_profit_pct:
                        tp_price = entry_price * (1 + (take_profit_pct / 100))
                        tp_distance = ((tp_price / current_price) - 1) * 100
                        content += f"Distância ao TP: **{tp_distance:.2f}%**\n"
            
            content += "\n"
        else:
            # Explicitly mention that there are no open trades
            content += f"## 📝 Status da Posição\n"
            content += "**Sem posição aberta** para este par no momento.\n\n"
        
        # Add recommendation section (highlighted and with emoji)
        if isinstance(llm_content, dict):
            recommendation = llm_content.get('recommendation', '').lower()
            confidence = llm_content.get('confidence', 0)
            
            # Add recommendation emoji based on the type
            rec_emoji = "🟢" if recommendation == 'buy' else "🔴" if recommendation == 'sell' else "🟡"
            
            content += f"## {rec_emoji} Recomendação: **{recommendation.upper()}**\n"
            content += f"Confiança: **{confidence:.2f}**\n\n"
            
            # Add financial targets if available
            if 'stop_loss' in llm_content or 'take_profit' in llm_content:
                content += "## 🎯 Alvos de Trading\n"
                if 'stop_loss' in llm_content:
                    content += f"Stop Loss: **{llm_content['stop_loss']}%**\n"
                if 'take_profit' in llm_content:
                    content += f"Take Profit: **{llm_content['take_profit']}%**\n"
                content += "\n"
            
            # Add trend information
            if 'trend' in llm_content:
                trend = llm_content['trend'].lower()
                trend_emoji = "📈" if trend == 'bullish' else "📉" if trend == 'bearish' else "➡️"
                content += f"## {trend_emoji} Tendência: **{trend.upper()}**\n\n"
                
            # Add detailed analysis
            if 'analysis' in llm_content:
                content += f"## 📝 Análise\n{llm_content['analysis'][:1500]}\n\n"
            elif 'summary' in llm_content:
                content += f"## 📝 Resumo\n{llm_content['summary'][:1500]}\n\n"
        else:
            # If it's not properly structured, use as is (truncated)
            content += f"## 📝 Análise\n{str(llm_content)[:1500]}\n\n"
            
        # Add timestamp
        content += f"*Gerado em: {generation_time}*"
        
        # Prepare the multipart form data
        files = [
            ("file" + str(i), (os.path.basename(path), open(path, "rb"), "image/png"))
            for i, path in enumerate(valid_image_paths[:10])  # Discord allows up to 10 attachments
        ]
        
        payload = {
            "content": content[:2000],  # Discord limit is 2000 chars
            "username": username,
        }
        
        if avatar_url:
            payload["avatar_url"] = avatar_url
        
        try:
            response = requests.post(
                self.webhook_url,
                data=payload,
                files=files
            )
            
            # Close all file handlers
            for _, file_tuple in files:
                file_tuple[1].close()
            
            if response.status_code == 204 or response.status_code == 200:
                logger.info(f"Analysis for {pair} ({timeframe}) sent successfully to Discord")
                return {
                    "success": True,
                    "response": response
                }
            else:
                logger.error(f"Failed to send analysis to Discord: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response": response
                }
                
        except Exception as e:
            # Close all file handlers in case of exception
            for _, file_tuple in files:
                try:
                    file_tuple[1].close()
                except:
                    pass
                
            logger.error(f"Error sending analysis to Discord: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            } 