"""
Plot the last N candles of a trading pair with indicators.
"""
import os
import logging
import base64
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
from PIL import Image
from typing import Dict, Optional, Tuple, Union, Any, List

# Configure logger
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64
    
    :param image_path: Path to the image file
    :return: Base64 encoded string or None if encoding fails
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

def plot_last_candles(pair, dataframe, timeframe, num_candles=30, output_dir="user_data/plot", 
                     indicators=None, indicators_below=None, save_format='png', use_plotly=False, 
                     save_html=False, volume_spacing='none', title=None, subtitle=None, sulfix_filename=None, 
                     question=None):
    """
    Plot the last N candles of a trading pair with indicators.
    
    Parameters:
    -----------
    pair : str
        Trading pair name (e.g., 'BTC/USDT')
    dataframe : pandas.DataFrame
        DataFrame containing OHLCV data
    timeframe : str
        Timeframe of the data (e.g., '5m', '1h', '1d')
    num_candles : int, optional
        Number of candles to plot (default: 30)
    output_dir : str, optional
        Directory to save the plot (default: 'user_data/plot')
    indicators : list, optional
        List of indicators to plot on main panel, each as a dict with 'name', 'color', 'panel', and 'width'
        Example: [{'name': 'sma', 'color': 'red', 'panel': 0, 'width': 1.5}]
    indicators_below : list, optional
        List of indicators to plot in separate panels below, each as a dict with 'name', 'color', and 'width'
        Example: [{'name': 'rsi', 'color': 'gray', 'width': 1.0}]
    save_format : str, optional
        Format to save the image ('png', 'jpg', 'svg', 'pdf')
    use_plotly : bool, optional
        Use plotly for interactive charts if True, otherwise use mplfinance (default: False)
    save_html : bool, optional
        When using plotly, also save an interactive HTML file (default: False)
    volume_spacing : str, optional
        Method to space volume bars: 'auto', 'sparse', or 'none' (default: 'auto')
    title : str, optional
        Main title for the chart (default: None, will use pair-timeframe)
    subtitle : str, optional
        Subtitle for the chart (default: None)
    sulfix_filename : str, optional
        Sulfix for the filename (default: None)    
        
    Returns:
    --------
    Union[Dict[str, Any], None]
        Dictionary with keys 'filepath' (path to the saved image) and 'base64' (base64-encoded image),
        or None if plot failed
    """
    
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare pair name for filename (replace / with _)
        safe_pair = pair.replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if sulfix_filename:
            filename = f"{safe_pair}_{timeframe}_{sulfix_filename}_{timestamp}.{save_format}"
        else:
            filename = f"{safe_pair}_{timeframe}_{timestamp}.{save_format}"
        filepath = os.path.join(output_dir, filename)
        
        # Copy the dataframe to avoid modifying the original
        df = dataframe.copy()
        
        # Ensure the index is a datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Get only the last N candles
        if len(df) > num_candles:
            df = df.iloc[-num_candles:]
            
        # Check if we have data
        if df.empty:
            logger.warning(f"No data available for {pair} in the specified timeframe")
            return None
            
        # Generate the plot using the selected method
        result_path = None
        if use_plotly:
            result_path = _plot_with_plotly(df, pair, timeframe, filepath, indicators, indicators_below, save_html, volume_spacing, title, subtitle)
        else:
            result_path = _plot_with_mplfinance(df, pair, timeframe, filepath, indicators, indicators_below, volume_spacing, title, subtitle)
        
        # If plotting succeeded, encode to base64
        if result_path:
            base64_image = encode_image_to_base64(result_path)
            if base64_image:
                logger.info(f"Successfully encoded image to base64: {result_path}")
                return {
                    "filepath": result_path,
                    "base64": base64_image,
                    "question": question
                }
            else:
                logger.error(f"Failed to encode image to base64: {result_path}")
                return {
                    "filepath": result_path,
                    "base64": None,
                    "question": question
                }
        
        return None
    
    except Exception as e:
        logger.error(f"Error plotting candles for {pair}: {e}", exc_info=True)
        return None

def _plot_with_mplfinance(df, pair, timeframe, filepath, indicators=None, indicators_below=None, volume_spacing='auto', title=None, subtitle=None):
    """
    Plot candles using mplfinance library
    """
    try:
        # Prepare the plot
        title = title or f"{pair} - {timeframe}"
        
        # Aplicar o espaçamento do volume conforme solicitado
        df_original = df.copy()  # Guardar cópia original
        
        if volume_spacing == 'sparse' or (volume_spacing == 'auto' and len(df) > 20 and timeframe in ['1m', '3m', '5m', '15m', '30m', '1h']):
            # Técnica de espaçamento por linhas vazias
            df_sparse = df.copy()
            # Adicionar linhas vazias para criar espaço visual
            for i in range(0, len(df), 2):
                if i < len(df) - 1:
                    # Inserir linha entre barras
                    try:
                        # Tentar calcular um índice intermediário
                        if isinstance(df.index, pd.DatetimeIndex):
                            mid_time = df.index[i] + (df.index[i+1] - df.index[i]) / 2
                            mid_idx = mid_time
                        else:
                            mid_idx = df.index[i] + 0.5
                            
                        # Criar linha vazia para espaçamento
                        empty_row = pd.Series({
                            'open': df['close'].iloc[i], 
                            'high': df['close'].iloc[i], 
                            'low': df['close'].iloc[i], 
                            'close': df['close'].iloc[i],
                            'volume': 0
                        }, name=mid_idx)
                        
                        # Adicionar à cópia esparsa
                        df_sparse = pd.concat([df_sparse, pd.DataFrame([empty_row])])
                    except Exception as e:
                        logger.warning(f"Erro ao criar linha vazia para espaçamento: {e}")
                        continue
            
            # Ordenar pelo índice para manter a cronologia
            try:
                df_sparse = df_sparse.sort_index()
                df = df_sparse
            except Exception as e:
                logger.warning(f"Erro ao ordenar DataFrame com espaçamento: {e}")
                df = df_original  # Voltar para o original em caso de erro
        
        # Set up colors and style
        mc = mpf.make_marketcolors(up='green', down='red',
                                  wick={'up':'green', 'down':'red'},
                                  volume={'up': 'green', 'down': 'red'},
                                  edge={'up': 'white', 'down': 'white'},  # Bordas brancas
                                  ohlc='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)
        
        # Ajuste do estilo para barras
        rc_params = {
            'axes.linewidth': 0.8, 
            'grid.linewidth': 0.6,
            # Adicionar bordas nas barras de volume
            'patch.linewidth': 1.5,
            'patch.edgecolor': 'white',
            'patch.force_edgecolor': True
        }
        s.update(rc=rc_params)
        
        # Prepare additional plots for indicators
        apds = []
        
        # Add main panel indicators
        if indicators:
            for ind in indicators:
                if ind['name'] in df.columns:
                    # Verificar o tipo de plotagem
                    plot_type = ind.get('type', 'line')
                    
                    if plot_type == 'line':
                        # Add to main panel (row=1)
                        apds.append(
                            mpf.make_addplot(
                                df[ind['name']], 
                                color=ind['color'], 
                                panel=0,  # Force main panel (price panel)
                                width=ind.get('width', 1.5),
                                ylabel=ind['name']  # Use column name as label
                            )
                        )
                    elif plot_type == 'bar':
                        # Para barras com cores diferentes por valor (positivo/negativo)
                        # Precisamos criar dois conjuntos de dados e plotar separadamente
                        positive_data = df[ind['name']].copy()
                        negative_data = df[ind['name']].copy()
                        
                        # Manter apenas os valores positivos/negativos
                        positive_data[positive_data <= 0] = float('nan')  # Remove negativos
                        negative_data[negative_data >= 0] = float('nan')  # Remove positivos
                        
                        # Cor positiva (verde ou a cor especificada)
                        apds.append(
                            mpf.make_addplot(
                                positive_data,
                                color=ind.get('positive_color', 'green') if 'positive_color' in ind else ind['color'],
                                panel=0,
                                type='bar',
                                width=ind.get('width', 1.5),
                                ylabel=ind['name'],
                                secondary_y=True  # Usar a escala da direita
                            )
                        )
                        
                        # Cor negativa (vermelho para negativos)
                        apds.append(
                            mpf.make_addplot(
                                negative_data,
                                color=ind.get('negative_color', 'red'),  # Vermelho para negativos
                                panel=0,
                                type='bar',
                                width=ind.get('width', 1.5),
                                secondary_y=True  # Usar a escala da direita também
                            )
                        )
                    elif plot_type == 'heatmap':
                        # Heatmap plotting requires custom implementation below
                        # We'll store the heatmap information to add after creating the plot
                        pass
                else:
                    logger.warning(f"Indicator {ind['name']} not found in dataframe columns")
        
        # Define panel indices - panel 0 is price, panel 1 is volume
        # So indicators_below start at panel 2
        first_indicator_panel = 2
        
        # Add indicators to separate panels below
        if indicators_below:
            # Primeiro, vamos agrupar os indicadores por painel
            panel_groups = {}
            # Mapeamento de nomes de painel para índices numéricos
            panel_name_to_idx = {}
            panel_idx = first_indicator_panel  # Contador para painéis numéricos
            
            for ind in indicators_below:
                if ind['name'] in df.columns:
                    # Obter o nome/identificador do painel do indicador
                    panel_id = ind.get('panel', None)
                    
                    if panel_id is None:
                        # Se não tiver painel especificado, use o nome do indicador
                        panel_id = ind['name']
                    
                    # Garantir que temos um índice numérico para cada painel
                    if panel_id not in panel_name_to_idx:
                        panel_name_to_idx[panel_id] = panel_idx
                        panel_idx += 1
                    
                    # Adicionar ao grupo do painel usando o identificador original
                    if panel_id not in panel_groups:
                        panel_groups[panel_id] = []
                    
                    panel_groups[panel_id].append(ind)
                else:
                    logger.warning(f"Indicator {ind['name']} not found in dataframe columns")
            
            # Agora, adicionar os indicadores painel por painel
            for panel_id, indicators_in_panel in panel_groups.items():
                # Obter o índice numérico do painel
                panel_num = panel_name_to_idx[panel_id]
                
                # Usar o nome do painel como rótulo ou o identificador do painel se for uma string
                panel_label = panel_id if isinstance(panel_id, str) else indicators_in_panel[0]['name']
                
                for ind in indicators_in_panel:
                    # Verificar o tipo de plotagem
                    plot_type = ind.get('type', 'line')
                    
                    if plot_type == 'line':
                        # Plotagem de linha (padrão)
                        apds.append(
                            mpf.make_addplot(
                                df[ind['name']], 
                                color=ind['color'], 
                                panel=panel_num,  # Usar o número de painel mapeado
                                width=ind.get('width', 1.5),
                                ylabel=panel_label  # Usar o nome/identificador como rótulo
                            )
                        )
                    elif plot_type == 'bar':
                        # Para barras com cores diferentes por valor (positivo/negativo)
                        # Precisamos criar dois conjuntos de dados e plotar separadamente
                        positive_data = df[ind['name']].copy()
                        negative_data = df[ind['name']].copy()
                        
                        # Manter apenas os valores positivos/negativos
                        positive_data[positive_data <= 0] = float('nan')  # Remove negativos
                        negative_data[negative_data >= 0] = float('nan')  # Remove positivos
                        
                        # Cor positiva (verde ou a cor especificada)
                        apds.append(
                            mpf.make_addplot(
                                positive_data,
                                color=ind.get('positive_color', 'green') if 'positive_color' in ind else ind['color'],
                                panel=panel_num,
                                type='bar',
                                width=ind.get('width', 1.5),
                                ylabel=panel_label,
                                secondary_y=True  # Usar a escala da direita
                            )
                        )
                        
                        # Cor negativa (vermelho para negativos)
                        apds.append(
                            mpf.make_addplot(
                                negative_data,
                                color=ind.get('negative_color', 'red'),  # Vermelho para negativos
                                panel=panel_num,
                                type='bar',
                                width=ind.get('width', 1.5),
                                secondary_y=True  # Usar a escala da direita também
                            )
                        )
            
            # Calcular total de painéis únicos necessários
            if panel_name_to_idx:
                total_panels = max(panel_name_to_idx.values()) + 1
            else:
                total_panels = first_indicator_panel
        else:
            total_panels = first_indicator_panel
        
        # Create the plot
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=s,
            title=title,
            volume=True,  # Certifica que o volume é exibido
            panel_ratios=(6, 2, *[2] * (total_panels - 2)),  # Main:Volume:Indicators ratio
            addplot=apds if apds else None,
            returnfig=True,
            figsize=(12, 8 + 2 * (total_panels - 2)),  # Adjust height based on panels
            num_panels=total_panels
        )
        
        # Add subtitle if provided
        if subtitle:
            fig.suptitle(title, y=0.99, fontsize=16, weight='bold')
            fig.text(0.5, 0.93, subtitle, ha='center', fontsize=12)
        else:
            fig.suptitle(title, y=0.98, fontsize=16, weight='bold')
        
        # Set axis labels - o matplotlib usa axes[i] ao invés de update_yaxes
        # axes[0] é o painel principal (preço)
        # axes[1] é o painel de volume
        if len(axes) > 0:
            axes[0].set_ylabel('Price')
        if len(axes) > 1:
            axes[1].set_ylabel('Volume')
        
        # Add a legend for indicators
        if (indicators or indicators_below) and len(apds) > 0:
            main_panel = axes[0]
            legend_elements = []
            
            # Add main indicators to legend
            if indicators:
                for ind in indicators:
                    if ind['name'] in df.columns and ind.get('type', 'line') != 'heatmap' and 'color' in ind:
                        from matplotlib.lines import Line2D
                        legend_elements.append(
                            Line2D([0], [0], color=ind['color'], lw=ind.get('width', 1.5), label=ind['name'])
                        )
            
            # Add indicators_below to legend
            if indicators_below:
                for ind in indicators_below:
                    if ind['name'] in df.columns and ind.get('type', 'line') != 'heatmap' and 'color' in ind:
                        from matplotlib.lines import Line2D
                        legend_elements.append(
                            Line2D([0], [0], color=ind['color'], lw=ind.get('width', 1.5), label=ind['name'])
                        )
            
            if legend_elements:
                main_panel.legend(handles=legend_elements, loc='upper left')
            
        # Now add custom heatmap if requested
        if indicators:
            for ind in indicators:
                if ind.get('type') == 'heatmap' and ind['name'] in df.columns:
                    try:
                        # Get the panel to draw the heatmap on
                        panel_num = ind.get('panel', 0)
                        panel = axes[panel_num]
                        
                        # Get the axes limits
                        x_min, x_max = panel.get_xlim()
                        y_min, y_max = panel.get_ylim()
                        
                        # Process last valid row with heatmap data
                        last_idx = len(df) - 1
                        while last_idx >= 0:
                            if pd.notna(df[ind['name']].iloc[last_idx]):
                                break
                            last_idx -= 1
                        
                        if last_idx >= 0 and pd.notna(df[ind['name']].iloc[last_idx]):
                            # Parse the heatmap data
                            try:
                                heatmap_data = eval(df[ind['name']].iloc[last_idx])
                                if isinstance(heatmap_data, list) and len(heatmap_data) > 0:
                                    # Extract price levels and intensity values
                                    price_levels = [item[0] for item in heatmap_data]
                                    intensity_values = [item[1] for item in heatmap_data]
                                    
                                    # Calculate height of each rectangle
                                    if len(price_levels) >= 2:
                                        height = (max(price_levels) - min(price_levels)) / (len(price_levels) - 1)
                                    else:
                                        height = (y_max - y_min) * 0.02
                                    
                                    # Create a colormap (hot_r for intensity)
                                    cmap = plt.cm.get_cmap('hot_r')
                                    
                                    # Get x coordinates for creating horizontal bars across the entire chart
                                    xmin = x_min
                                    xmax = x_max
                                    width = xmax - xmin
                                    
                                    # Draw horizontal bars across the entire chart for each price level
                                    for i, (price, intensity) in enumerate(zip(price_levels, intensity_values)):
                                        # Calculate color based on intensity
                                        color = cmap(intensity)
                                        # Create rectangle with transparency (alpha) to see candles through it
                                        rect = plt.Rectangle(
                                            (xmin, price - height/2),  # Lower left corner position
                                            width,                     # Width (full chart width)
                                            height,                    # Height (based on price level spacing)
                                            color=color,
                                            alpha=0.3,                # Make it transparent
                                            linewidth=0,              # No outline
                                            zorder=0                  # Place behind candles
                                        )
                                        panel.add_patch(rect)
                                    
                                    # Add a small legend explaining heatmap colors in text only, no colorbar
                                    text_x = x_min + width * 0.02
                                    text_y = y_min + (y_max - y_min) * 0.03
                                    panel.text(
                                        text_x, text_y, 
                                        "Heat levels: Bright = High Volume/Price Activity", 
                                        fontsize=8, 
                                        alpha=0.7,
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                                    )
                            except Exception as e:
                                logger.error(f"Error parsing heatmap data: {e}")
                    except Exception as e:
                        logger.error(f"Error adding heatmap to plot: {e}")
        
        # Save the figure
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved candle plot to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error in mplfinance plotting: {e}", exc_info=True)
        return None

def _plot_with_plotly(df, pair, timeframe, filepath, indicators=None, indicators_below=None, save_html=False, volume_spacing='auto', title=None, subtitle=None):
    """
    Plot candles using plotly library for interactive charts
    """
    try:
        # Determine how many subplots we need for indicators_below
        panel_count = 0
        panel_groups = {}
        panel_name_to_idx = {}
        panel_titles = []

        if indicators_below:
            # Agrupar indicadores por painel
            panel_idx = 0  # Contador para painéis
            for ind in indicators_below:
                if ind['name'] in df.columns:
                    # Obter nome do painel ou usar nome do indicador
                    panel_id = ind.get('panel', ind['name'])
                    
                    # Manter mapeamento de nome para índice
                    if panel_id not in panel_name_to_idx:
                        panel_name_to_idx[panel_id] = panel_idx
                        # Armazenar o título/nome do painel
                        panel_titles.append(panel_id if isinstance(panel_id, str) else ind['name'])
                        panel_idx += 1
                    
                    # Adicionar ao grupo de painel
                    if panel_id not in panel_groups:
                        panel_groups[panel_id] = []
                    
                    panel_groups[panel_id].append(ind)
        
            # Número de painéis é o número de grupos únicos
            panel_count = len(panel_groups)

        # Create subplots - main chart for candles + volume, additional for indicators below
        fig = make_subplots(
            rows=panel_count + 2,  # +2 for price and volume
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,  # Slightly increase spacing between panels
            row_heights=[0.5, 0.2] + [0.2] * panel_count,  # Main chart, volume, and indicator panels
            subplot_titles=(
                "",  # Deixaremos o título principal vazio para usar o título personalizado
                "Volume",
                *panel_titles  # Usar os títulos dos painéis
            ),
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] + [[{"secondary_y": True}]] * panel_count  # Adicionar eixo secundário para o painel principal e indicadores
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Candles"
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add volume trace with colors based on price direction
        # Create color array - green for bullish candles, red for bearish
        colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in df.iterrows()]
        
        # Determinar a largura das barras com base no parâmetro volume_spacing
        bar_width = 0.8  # Padrão
        if volume_spacing == 'sparse' or (volume_spacing == 'auto' and len(df) <= 30):
            bar_width = 0.4  # Mais espaço entre barras
        elif volume_spacing == 'none':
            bar_width = 0.9  # Barras próximas
        
        # Adicionar borda nas barras para destacar cada uma individualmente
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name="Volume",
                marker=dict(
                    color=colors, 
                    opacity=0.8,
                    line=dict(
                        width=2.0,  # Borda mais grossa
                        color='rgba(255, 255, 255, 1.0)'  # Branco puro com opacidade total
                    )
                ),
                width=bar_width  # Largura ajustável das barras
            ),
            row=2, col=1
        )
        
        # Add main panel indicators
        if indicators:
            for ind in indicators:
                if ind['name'] in df.columns:
                    # Verificar o tipo de plotagem
                    plot_type = ind.get('type', 'line')
                    
                    if plot_type == 'line':
                        # Add to main panel (row=1)
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[ind['name']],
                                name=ind['name'],  # Use column name as legend label
                                line=dict(color=ind['color'], width=ind.get('width', 1.5))
                            ),
                            row=1, col=1, secondary_y=False
                        )
                    elif plot_type == 'bar':
                        # Para barras com cores diferentes por valor (positivo/negativo)
                        # Precisamos criar dois conjuntos de dados e plotar separadamente
                        positive_data = df[ind['name']].copy()
                        negative_data = df[ind['name']].copy()
                        
                        # Manter apenas os valores positivos/negativos
                        positive_data[positive_data <= 0] = float('nan')  # Remove negativos
                        negative_data[negative_data >= 0] = float('nan')  # Remove positivos
                        
                        # Cor positiva (verde ou a cor especificada)
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=positive_data,
                                name=ind['name'] + " (Positivo)",
                                marker=dict(
                                    color=ind.get('positive_color', 'green') if 'positive_color' in ind else ind['color'],
                                    line=dict(width=0.5, color='white')
                                )
                            ),
                            row=1, col=1, secondary_y=True
                        )
                        
                        # Cor negativa (vermelho para negativos)
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=negative_data,
                                name=ind['name'] + " (Negativo)",
                                marker=dict(
                                    color=ind.get('negative_color', 'red'),  # Vermelho para negativos
                                    line=dict(width=0.5, color='white')
                                )
                            ),
                            row=1, col=1, secondary_y=True
                        )
                    elif plot_type == 'heatmap':
                        # Get last row with valid heatmap data
                        last_idx = len(df) - 1
                        while last_idx >= 0:
                            if pd.notna(df[ind['name']].iloc[last_idx]):
                                break
                            last_idx -= 1
                        
                        if last_idx >= 0 and pd.notna(df[ind['name']].iloc[last_idx]):
                            try:
                                # Parse the heatmap data
                                heatmap_data = eval(df[ind['name']].iloc[last_idx])
                                if isinstance(heatmap_data, list) and len(heatmap_data) > 0:
                                    # Extract price levels and intensity values
                                    price_levels = [item[0] for item in heatmap_data]
                                    intensity_values = [item[1] for item in heatmap_data]
                                    
                                    # Add horizontal rectangles for each price level
                                    for price, intensity in zip(price_levels, intensity_values):
                                        # Skip if intensity is too low (reduces visual clutter)
                                        if intensity < 0.1:  
                                            continue
                                            
                                        # Calculate height of the bar
                                        if len(price_levels) >= 2:
                                            height = (max(price_levels) - min(price_levels)) / (len(price_levels) - 1)
                                        else:
                                            height = (price * 0.001)  # Default height if can't calculate
                                        
                                        # Calculate color from hot_r colormap
                                        # Convert intensity (0-1) to a color in the hot_r scale
                                        r = min(1.0, intensity * 2)
                                        g = max(0, min(1.0, intensity * 1.5 - 0.2))
                                        b = max(0, min(0.8, intensity - 0.6))
                                        
                                        # Add a rectangle shape for this price level
                                        fig.add_shape(
                                            type="rect",
                                            x0=df.index[0],              # Start from the left of the chart
                                            x1=df.index[-1],             # End at the right of the chart
                                            y0=price - height/2,         # Bottom of the bar
                                            y1=price + height/2,         # Top of the bar
                                            fillcolor=f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.3)",  # With transparency
                                            line=dict(width=0),
                                            layer="below",               # Place behind the candles
                                            row=1, col=1
                                        )
                                    
                                    # Add an annotation explaining the heatmap
                                    fig.add_annotation(
                                        x=df.index[int(len(df)/4)],  # Quarter way through the chart
                                        y=min(price_levels),          # Near the bottom of price range
                                        text="Heat levels: Bright = High Volume/Price Activity",
                                        showarrow=False,
                                        bgcolor="rgba(255,255,255,0.5)",
                                        font=dict(size=10),
                                        row=1, col=1
                                    )
                            except Exception as e:
                                logger.error(f"Error adding heatmap to plotly plot: {e}")
                else:
                    logger.warning(f"Indicator {ind['name']} not found in dataframe columns")
        
        # Add indicators below in separate panels
        if panel_groups:
            # Para cada grupo de painéis
            for panel_id, indicators_in_panel in panel_groups.items():
                # Calcular o índice da linha no plotly (painéis começam em 3 após preço e volume)
                panel_idx = panel_name_to_idx[panel_id]
                panel_row = panel_idx + 3
                
                # Usar o nome do painel como título
                panel_title = panel_id if isinstance(panel_id, str) else indicators_in_panel[0]['name']
                
                # Adicionar cada indicador ao mesmo painel
                for ind in indicators_in_panel:
                    # Verificar o tipo de plotagem
                    plot_type = ind.get('type', 'line')
                    
                    if plot_type == 'line':
                        # Plotagem de linha (padrão)
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df[ind['name']],
                                name=ind['name'],
                                line=dict(color=ind['color'], width=ind.get('width', 1.5))
                            ),
                            row=panel_row, col=1, secondary_y=False
                        )
                    elif plot_type == 'bar':
                        # Para barras com cores diferentes por valor (positivo/negativo)
                        # Precisamos criar dois conjuntos de dados e plotar separadamente
                        positive_data = df[ind['name']].copy()
                        negative_data = df[ind['name']].copy()
                        
                        # Manter apenas os valores positivos/negativos
                        positive_data[positive_data <= 0] = float('nan')  # Remove negativos
                        negative_data[negative_data >= 0] = float('nan')  # Remove positivos
                        
                        # Cor positiva (verde ou a cor especificada)
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=positive_data,
                                name=ind['name'] + " (Positivo)",
                                marker=dict(
                                    color=ind.get('positive_color', 'green') if 'positive_color' in ind else ind['color'],
                                    line=dict(width=0.5, color='white')
                                )
                            ),
                            row=panel_row, col=1, secondary_y=True
                        )
                        
                        # Cor negativa (vermelho para negativos)
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=negative_data,
                                name=ind['name'] + " (Negativo)",
                                marker=dict(
                                    color=ind.get('negative_color', 'red'),  # Vermelho para negativos
                                    line=dict(width=0.5, color='white')
                                )
                            ),
                            row=panel_row, col=1, secondary_y=True
                        )
                
                # Atualizar título do eixo y para este painel
                fig.update_yaxes(title_text=panel_title, row=panel_row, col=1)
                
                # Atualizar título da subplot
                if panel_row <= len(fig.layout.annotations):
                    fig.layout.annotations[panel_row-1].text = panel_title
        
        # Increase vertical spacing and add border between panels
        for i in range(1, panel_count + 2):
            fig.update_xaxes(showgrid=True, gridcolor='lightgray', row=i, col=1)
            
            if i == 1:  # Main panel (price + MACD)
                # Eixo Y primário (esquerda) - Preço
                fig.update_yaxes(
                    title_text='Price', 
                    showgrid=True, 
                    gridcolor='lightgray',
                    side='left',
                    row=i, col=1,
                    secondary_y=False
                )
                # Eixo Y secundário (direita) - MACD/Histograma
                fig.update_yaxes(
                    title_text='Indicator',
                    showgrid=False,
                    side='right', 
                    row=i, col=1,
                    secondary_y=True,
                    fixedrange=False  # Permitir zoom
                )
            else:
                # Configurar eixos Y para os outros painéis
                fig.update_yaxes(
                    showgrid=True, 
                    gridcolor='lightgray', 
                    row=i, col=1,
                    side='left',  # Garantir que todos os eixos Y estejam à esquerda
                    fixedrange=False  # Permitir zoom no eixo Y
                )
        
        # Update layout with more height for multiple panels
        chart_title = title or f"{pair} - {timeframe}"
        subtitle_text = subtitle or ""

        fig.update_layout(
            title={
                'text': chart_title + ("<br><span style='font-size:0.8em; font-weight:normal'>" + subtitle_text + "</span>" if subtitle_text else ""),
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_rangeslider_visible=False,
            height=800 + (panel_count * 200),  # Increase height based on number of indicators
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Save as static image first
        fig.write_image(filepath)
        logger.info(f"Saved static plot to {filepath}")
        
        # Optionally save as interactive HTML
        if save_html:
            html_filepath = filepath.replace(f".{filepath.split('.')[-1]}", ".html")
            fig.write_html(html_filepath)
            logger.info(f"Saved interactive plot to {html_filepath}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error in plotly plotting: {e}", exc_info=True)
        return None 