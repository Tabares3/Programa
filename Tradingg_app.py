import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import feedparser
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import streamlit.components.v1 as components

# Importar Prophet
from prophet import Prophet
from prophet.plot import plot_plotly # No se usa directamente pero es común importarlo

st.set_page_config(page_title="Asistente de Trading", layout="wide")
st.title("\U0001F4C8 Asistente de Trading - Señales de Compra/Venta + Noticias en Tiempo Real")

with st.expander("ℹ️ ¿Cómo funciona esta aplicación?"):
    st.markdown("""
    Esta aplicación realiza análisis técnico y fundamental automático:

    ### 🔍 Análisis Técnico:
    - Indicadores: RSI, SMA 20 y SMA 50
    - Señales:
        - 🟢 Comprar (RSI < 30 y SMA_20 > SMA_50)
        - 🔴 Vender (RSI > 70 y SMA_20 < SMA_50)
        - ❌ Mantener (otros casos)

    ### ✉️ Alertas automáticas por correo:
    - Se envía email a juaanda2313@gmail.com desde juansebastiantabaresmartinez@gmail.com con la señal detectada.

    ### 📜 Noticias Financieras
    - RSS de múltiples fuentes

    ### 📓 Análisis Fundamental
    - P/E, EPS, ROE, ROA, Deuda/Capital

    ### ✔️ Instrucciones:
    1. Escribe uno o varios símbolos (ej. AAPL, MSFT)
    2. Haz clic en "Analizar"
    """)

sector_options = {
    "Tecnología": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Energía": ["XOM", "CVX"],
    "Salud": ["JNJ", "PFE", "UNH"],
    "Consumo": ["KO", "MCD", "NKE"],
    "Industriales": ["CAT", "LMT"]
}

col1, col2 = st.columns([1, 2])

with col1:
    selected_sector = st.selectbox("📂 Selecciona un sector:", options=["(Ninguno)"] + list(sector_options.keys()))

with col2:
    default_symbols = ", ".join(sector_options[selected_sector]) if selected_sector != "(Ninguno)" else "AAPL"
    symbol_input = st.text_input("🔭 Símbolos a analizar (ej. AAPL, MSFT, BTC-USD):", value=default_symbols)

symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]

st.subheader("🗞 Noticias Relevantes del Mercado")

news_sources = {
    "Yahoo Finance - Top Stories": "https://finance.yahoo.com/rss/topfinstories",
    "Bloomberg ETF Report": "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "Investing.com - Noticias económicas": "https://www.investing.com/rss/news.rss",
    "Reuters - Business News": "http://feeds.reuters.com/reuters/businessNews"
}

selected_sources = st.multiselect(
    "Selecciona fuentes de noticias:",
    options=list(news_sources.keys()),
    default=["Yahoo Finance - Top Stories", "Investing.com - Noticias económicas"]
)

num_articles = st.slider("¿Cuántas noticias por fuente deseas ver?", min_value=1, max_value=10, value=3)

for fuente in selected_sources:
    url = news_sources[fuente]
    feed = feedparser.parse(url)
    st.markdown(f"### {fuente}")
    if not feed.entries:
        st.caption("⚠️ No hay noticias disponibles en este momento.")
    for entry in feed.entries[:num_articles]:
        st.markdown(f"**[{entry.title}]({entry.link})**")
        st.caption(entry.published)

def enviar_alerta_por_correo(symbol, fecha, signal):
    try:
        remitente = st.secrets["email"]["remitente"]
        destinatario = st.secrets["email"]["destinatario"]
        token = st.secrets["email"]["token"]
    except KeyError as e:
        st.error(f"Error: La configuración de email en secrets.toml está incompleta o falta la clave {e}. Por favor, verifica tu archivo secrets.toml.")
        return

    asunto = f"Alerta de Trading para {symbol}: {signal}"

    mensaje = MIMEMultipart()
    mensaje['From'] = remitente
    mensaje['To'] = destinatario
    mensaje['Subject'] = asunto
    cuerpo = f"Señal detectada para {symbol} el {fecha}: {signal}"
    mensaje.attach(MIMEText(cuerpo, 'plain'))

    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, token)
        servidor.sendmail(remitente, destinatario, mensaje.as_string())
        servidor.quit()
        st.success(f"✉️ Alerta enviada para {symbol} - {signal}")
    except Exception as e:
        st.error(f"Error al enviar correo: {e}. Asegúrate de que el remitente y el token de aplicación de Gmail son correctos.")

def mostrar_tradingview_widget(simbolo_tv, ancho=980, alto=610):
    codigo_html = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_{simbolo_tv}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": {ancho},
        "height": {alto},
        "symbol": "{simbolo_tv}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "es",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_{simbolo_tv}"
      }});
      </script>
    </div>
    """
    components.html(codigo_html, height=alto + 50)

# Función para identificar patrones de velas simples
def identify_candlestick_patterns(data, open_col, high_col, low_col, close_col):
    body_size_doji = abs(data[close_col] - data[open_col])
    total_range_doji = data[high_col] - data[low_col]
    data['Doji'] = (body_size_doji < total_range_doji * 0.1) & (total_range_doji > 0)

    body_size = abs(data[close_col] - data[open_col])
    total_range = data[high_col] - data[low_col]

    lower_shadow = data[[open_col, close_col]].min(axis=1) - data[low_col]
    upper_shadow = data[high_col] - data[[open_col, close_col]].max(axis=1)

    data['Hammer'] = (
        (body_size < total_range * 0.2) &
        (lower_shadow > (body_size * 2)) &
        (upper_shadow < body_size * 0.5) &
        (total_range > 0)
    )

    data['Engulfing_Bullish'] = (
        (data[close_col] > data[open_col]) &
        (data[close_col].shift(1) < data[open_col].shift(1)) &
        (data[open_col] < data[close_col].shift(1)) &
        (data[close_col] > data[open_col].shift(1))
    ).fillna(False)

    data['Engulfing_Bearish'] = (
        (data[close_col] < data[open_col]) &
        (data[close_col].shift(1) > data[open_col].shift(1)) &
        (data[open_col] > data[close_col].shift(1)) &
        (data[close_col] < data[open_col].shift(1))
    ).fillna(False)

    return data


if st.button("🔍 Analizar"):
    all_recommendations = [] # Para la fase 2

    for symbol in symbols:
        st.markdown(f"## 📊 Resultados para {symbol}")
        try:
            # Aumentar el período de descarga para Prophet
            data = yf.download(symbol, period="2y", interval="1d")
            if data.empty:
                st.warning(f"⚠️ No se encontraron datos para {symbol}")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]

            try:
                open_col = [col for col in data.columns if 'Open' in col][0]
                high_col = [col for col in data.columns if 'High' in col][0]
                low_col = [col for col in data.columns if 'Low' in col][0]
                close_col = [col for col in data.columns if 'Close' in col][0]
            except IndexError:
                st.warning(f"⚠️ No se encontraron columnas necesarias para {symbol}")
                continue

            close_prices = data[close_col]
            data['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
            data['SMA_20'] = ta.trend.SMAIndicator(close=close_prices, window=20).sma_indicator()
            data['SMA_50'] = ta.trend.SMAIndicator(close=close_prices, window=50).sma_indicator()

            data['Inside_Bar'] = (
                (data[high_col] < data[high_col].shift(1)) &
                (data[low_col] > data[low_col].shift(1))
            )

            data['BB_Upper'] = ta.volatility.BollingerBands(close=close_prices, window=20, window_dev=2).bollinger_hband()
            data['BB_Lower'] = ta.volatility.BollingerBands(close=close_prices, window=20, window_dev=2).bollinger_lband()
            data['BB_Middle'] = ta.volatility.BollingerBands(close=close_prices, window=20, window_dev=2).bollinger_mavg()

            data = identify_candlestick_patterns(data, open_col, high_col, low_col, close_col)

            data["Signal"] = "❌ Mantener"
            data.loc[(data['RSI'] < 30) & (data['SMA_20'] > data['SMA_50']), "Signal"] = "🟢 Comprar"
            data.loc[(data['RSI'] > 70) & (data['SMA_20'] < data['SMA_50']), "Signal"] = "🔴 Vender"

            output_cols = [close_col, 'RSI', 'SMA_20', 'SMA_50', 'Signal',
                           'Inside_Bar', 'Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish']
            
            # Asegurarse de que todas las columnas existen antes de intentar usarlas
            existing_output_cols = [col for col in output_cols if col in data.columns]
            output = data[existing_output_cols].dropna()

            output = output.rename(columns={close_col: 'Close'})

            ultimas = output.tail(3)
            if not ultimas.empty:
                for fecha, fila in ultimas.iterrows():
                    if fila['Signal'] != '❌ Mantener':
                        enviar_alerta_por_correo(symbol, fecha.date(), fila['Signal'])

            st.dataframe(output.tail(20))
            st.line_chart(output[['Close', 'SMA_20', 'SMA_50']])

            st.subheader(f"📉 Gráfico de Velas Japonesas - {symbol}")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data[open_col], high=data[high_col],
                low=data[low_col], close=data[close_col],
                name='Velas'))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_20'],
                line=dict(color='blue', width=1), name='SMA 20'))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_50'],
                line=dict(color='orange', width=1), name='SMA 50'))

            for pattern_col, color, symbol_marker, name in [
                ("Inside_Bar", 'purple', 'diamond', 'Inside Bar'),
                ("Doji", 'gray', 'star', 'Doji'),
                ("Hammer", 'green', 'triangle-up', 'Martillo (Hammer)'),
                ("Engulfing_Bullish", 'darkgreen', 'circle', 'Envolvente Alcista'),
                ("Engulfing_Bearish", 'darkred', 'circle', 'Envolvente Bajista')
            ]:
                if pattern_col in data.columns: # Asegurarse de que la columna existe
                    pattern_indices = data.index[data[pattern_col].fillna(False)]
                    if not pattern_indices.empty:
                        pattern_closes = data.loc[pattern_indices, close_col]
                        fig.add_trace(go.Scatter(
                            x=pattern_indices,
                            y=pattern_closes,
                            mode='markers',
                            marker=dict(color=color, size=8 if 'Doji' in name else 10, symbol=symbol_marker),
                            name=name
                        ))


            fig.update_layout(
                xaxis_rangeslider_visible=False, height=500,
                title=f"{symbol} - Velas + SMA",
                xaxis_title="Fecha", yaxis_title="Precio")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"📺 Gráfico Avanzado de {symbol} (TradingView)")
            tv_symbol_map = {
                "CL=F": "TVC:USOIL",
                "GC=F": "TVC:GOLD",
                "SI=F": "TVC:SILVER",
                "KC=F": "ICEUSA:KC1!",
                "BTC-USD": "BINANCE:BTCUSDT",
                "ETH-USD": "BINANCE:ETHUSDT"
            }
            tv_symbol = tv_symbol_map.get(symbol, symbol.replace("-", ""))
            mostrar_tradingview_widget(tv_symbol)

            st.subheader(f"🔮 Predicción de Precio para {symbol} (Próximos 125 días hábiles)")

            df_prophet = output.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                        changepoint_prior_scale=0.05,
                        seasonality_prior_scale=10.0
                       )
            m.fit(df_prophet)

            future = m.make_future_dataframe(periods=125)
            forecast = m.predict(future)

            # --- NUEVO: Cuadro con los posibles precios futuros de Prophet (por día) ---
            st.markdown("### 📈 Precios Futuros Predichos por Día (Prophet)")
            # Seleccionar solo las fechas futuras para la tabla
            future_forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(125).copy()
            future_forecast_table['ds'] = future_forecast_table['ds'].dt.strftime('%Y-%m-%d')
            future_forecast_table.columns = ['Fecha', 'Predicción (yhat)', 'Límite Inferior', 'Límite Superior']
            st.dataframe(future_forecast_table.set_index('Fecha').style.format("{:.2f}"))

            # --- Diagnóstico de la predicción de Prophet ---
            st.markdown("### 💡 Diagnóstico de la Predicción (Prophet)")
            # Calcular el cambio porcentual de la predicción
            initial_pred_price = forecast['yhat'].iloc[-125] # Precio al inicio del periodo de predicción
            final_pred_price = forecast['yhat'].iloc[-1]   # Precio al final del periodo de predicción
            
            if initial_pred_price != 0:
                predicted_change_percent = ((final_pred_price - initial_pred_price) / initial_pred_price) * 100
                st.info(f"**Cambio Predicho (Prophet):** El precio predicho de {symbol} cambiará un **{predicted_change_percent:.2f}%** en los próximos 125 días hábiles (aprox. 6 meses).")
            else:
                 st.info(f"**Cambio Predicho (Prophet):** No se pudo calcular el cambio porcentual para {symbol} debido a un precio inicial de predicción cero.")


            # Evaluar la señal actual para la recomendación
            current_signal_row = output.iloc[-1]
            current_signal = current_signal_row['Signal']

            recommendation_text = ""
            if current_signal == "🟢 Comprar":
                recommendation_text = "La señal de análisis técnico es **COMPRAR**."
                if predicted_change_percent > 5: # Umbral de ejemplo para "fuerte"
                    recommendation_text += " La predicción de Prophet sugiere un fuerte potencial alcista."
                    st.success(f"✔️ **RECOMENDACIÓN para {symbol}:** Fuerte indicio de compra. {recommendation_text}")
                elif predicted_change_percent > 0:
                    recommendation_text += " La predicción de Prophet sugiere una tendencia alcista."
                    st.success(f"✔️ **RECOMENDACIÓN para {symbol}:** Indicación de compra. {recommendation_text}")
                else:
                    recommendation_text += " La predicción de Prophet sugiere una posible lateralización o leve caída, a pesar de la señal de compra inicial."
                    st.warning(f"⚠️ **RECOMENDACIÓN para {symbol}:** Compra cautelosa. {recommendation_text}")
            elif current_signal == "🔴 Vender":
                recommendation_text = "La señal de análisis técnico es **VENDER**."
                if predicted_change_percent < -5: # Umbral de ejemplo para "fuerte"
                    recommendation_text += " La predicción de Prophet sugiere un fuerte potencial bajista."
                    st.error(f"❌ **RECOMENDACIÓN para {symbol}:** Fuerte indicio de venta. {recommendation_text}")
                elif predicted_change_percent < 0:
                    recommendation_text += " La predicción de Prophet sugiere una tendencia bajista."
                    st.error(f"❌ **RECOMENDACIÓN para {symbol}:** Indicación de venta. {recommendation_text}")
                else:
                    recommendation_text += " La predicción de Prophet sugiere una posible lateralización o leve subida, a pesar de la señal de venta inicial."
                    st.warning(f"⚠️ **RECOMENDACIÓN para {symbol}:** Venta cautelosa. {recommendation_text}")
            else:
                recommendation_text = "La señal de análisis técnico es **MANTENER**."
                if abs(predicted_change_percent) < 2: # Umbral de ejemplo para "lateral"
                    recommendation_text += " La predicción de Prophet sugiere una tendencia lateral."
                    st.info(f"ℹ️ **RECOMENDACIÓN para {symbol}:** Mantener. {recommendation_text}")
                elif predicted_change_percent > 0:
                    recommendation_text += " La predicción de Prophet sugiere una tendencia alcista, lo que podría implicar mantener o buscar un punto de entrada."
                    st.info(f"ℹ️ **RECOMENDACIÓN para {symbol}:** Mantener, con posible potencial alcista. {recommendation_text}")
                else:
                    recommendation_text += " La predicción de Prophet sugiere una tendencia bajista, lo que podría implicar mantener o considerar salir."
                    st.info(f"ℹ️ **RECOMENDACIÓN para {symbol}:** Mantener, con posible riesgo bajista. {recommendation_text}")

            # Almacenar para el análisis comparativo (Fase 2)
            all_recommendations.append({
                "Símbolo": symbol,
                "Señal Actual": current_signal,
                "Cambio Predicho (%)": f"{predicted_change_percent:.2f}%",
                "Predicción Final": f"{final_pred_price:.2f}",
                "Recomendación General": recommendation_text
            })


            fig_pred = go.Figure()

            fig_pred.add_trace(go.Scatter(x=output.index, y=output["Close"], mode='lines', name="Histórico"))

            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                mode='lines', name='Prophet - Predicción (Picos y Caídas)', line=dict(color='cyan', dash='solid')
            ))
            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'],
                mode='lines', line=dict(color='rgba(0,100,80,0.2)', width=0),
                showlegend=False
            ))
            fig_pred.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'],
                mode='lines',
                line=dict(color='rgba(0,100,80,0.2)', width=0),
                fill='tonexty', fillcolor='rgba(0,100,80,0.1)',
                name='Prophet - Intervalo de Confianza'
            ))

            close_data = output['Close'].values
            dias = np.arange(len(close_data)).reshape(-1, 1)
            modelo_lineal = LinearRegression()
            modelo_lineal.fit(dias, close_data)

            dias_futuros = np.arange(len(close_data), len(close_data) + 125).reshape(-1, 1)
            fechas_futuras = pd.date_range(start=output.index[-1], periods=125, freq='B')

            predicciones_lineal = modelo_lineal.predict(dias_futuros)
            fig_pred.add_trace(go.Scatter(x=fechas_futuras, y=predicciones_lineal.flatten(), mode='lines', name="Predicción Lineal", line=dict(color='blue', dash='dash')))

            model_g2 = make_pipeline(PolynomialFeatures(2), LinearRegression())
            model_g2.fit(dias, close_data)
            pred_g2 = model_g2.predict(dias_futuros)
            fig_pred.add_trace(go.Scatter(
                x=fechas_futuras, y=pred_g2.flatten(),
                mode='lines', name='Predicción Polinómica G2', line=dict(color='green', dash='dot')))

            fig_pred.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'],
                mode='lines', name='BB Superior Hist.', line=dict(color='lightcoral', width=1)))
            fig_pred.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'],
                mode='lines', name='BB Inferior Hist.', line=dict(color='lightskyblue', width=1)))
            fig_pred.add_trace(go.Scatter(
                x=data.index, y=data['BB_Middle'],
                mode='lines', name='BB Media Hist.', line=dict(color='darkgray', width=1)))


            fig_pred.update_layout(title=f"Predicción de Precio para {symbol} con Múltiples Modelos", xaxis_title="Fecha", yaxis_title="Precio")
            st.plotly_chart(fig_pred, use_container_width=True)

            st.markdown("""
            **Sobre las Predicciones:**
            - **Prophet (Picos y Caídas):** Modelo de series temporales diseñado para predecir series con fuertes efectos estacionales (diarios, semanales, anuales) y festivos. Intenta capturar mejor los patrones de subidas y bajadas. El área sombreada representa su intervalo de confianza.
            - **Predicción Lineal:** Muestra una tendencia general futura. No predice picos o caídas.
            - **Predicción Polinómica (G2):** Intenta capturar curvas y cambios de dirección. Un grado 2 puede ofrecer una curva más suave. Grados más altos pueden ajustarse demasiado a los datos históricos y generar predicciones poco realistas fuera del rango conocido.
            **Importante:** Las predicciones son solo estimaciones. El mercado puede ser muy volátil y no hay garantía de que los patrones históricos se repitan.
            """)

            # --- Información de Patrones de Velas Detectados (Mejorado) ---
            st.subheader(f"🕯️ Análisis de Patrones de Velas Japonesas Recientes para {symbol}")

            patterns_df = output[['Inside_Bar', 'Doji', 'Hammer', 'Engulfing_Bullish', 'Engulfing_Bearish']].tail(10).copy()
            patterns_df = patterns_df.replace({True: '✅', False: '❌'})
            patterns_df.index.name = "Fecha"
            st.dataframe(patterns_df)

            st.markdown("""
            **Resumen de Patrones Recientes Detectados:**
            """)
            found_patterns_summary = False
            if patterns_df['Inside_Bar'].str.contains('✅').any():
                st.info("🟡 **Inside Bar (Vela Interna)** detectada. Puede indicar consolidación o indecisión antes de un posible movimiento fuerte.")
                found_patterns_summary = True
            if patterns_df['Doji'].str.contains('✅').any():
                st.info("⚫ **Doji** detectada. Indica indecisión en el mercado, con la apertura y el cierre muy cercanos. Puede ser una señal de reversión.")
                found_patterns_summary = True
            if patterns_df['Hammer'].str.contains('✅').any():
                st.success("🔨 **Martillo (Hammer)** detectado. A menudo ocurre después de una tendencia bajista y sugiere un posible giro alcista.")
                found_patterns_summary = True
            if patterns_df['Engulfing_Bullish'].str.contains('✅').any():
                st.success("⬆️ **Envolvente Alcista (Bullish Engulfing)** detectada. Un patrón fuerte de reversión alcista.")
                found_patterns_summary = True
            if patterns_df['Engulfing_Bearish'].str.contains('✅').any():
                st.error("⬇️ **Envolvente Bajista (Bearish Engulfing)** detectada. Un patrón fuerte de reversión bajista.")
                found_patterns_summary = True

            if not found_patterns_summary:
                st.info("No se detectaron los patrones de velas principales (Doji, Martillo, Envolvente) en los últimos 10 días.")


            st.markdown("""
            **Interpretación de Patrones de Velas:**
            - **Inside Bar:** La vela actual está contenida dentro del rango (alto-bajo) de la vela anterior. Sugiere consolidación e indecisión, y a menudo precede a un movimiento significativo.
            - **Doji:** La apertura y el cierre están muy cerca uno del otro, formando un cuerpo pequeño o inexistente. Indica indecisión o un equilibrio entre compradores y vendedores. Puede ser una señal de reversión.
            - **Martillo (Hammer):** Pequeño cuerpo real en la parte superior del rango del día, con una larga sombra inferior y poca o ninguna sombra superior. Generalmente aparece en una tendencia bajista e indica una posible reversión alcista.
            - **Envolvente Alcista (Bullish Engulfing):** Una vela alcista grande que "envuelve" completamente el cuerpo real de una vela bajista anterior. Fuerte señal de reversión alcista.
            - **Envolvente Bajista (Bearish Engulfing):** Una vela bajista grande que "envuelve" completamente el cuerpo real de una vela alcista anterior. Fuerte señal de reversión bajista.
            """)

        except Exception as e:
            st.error(f"🚨 Error en análisis de {symbol}: {e}")

    st.subheader("📘 Comparación de Análisis Fundamental")
    fundamental_data = []
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info
            fundamental_data.append({
                "Símbolo": symbol,
                "🧲 P/E Ratio": round(info.get("trailingPE", 0), 2),
                "💰 EPS": round(info.get("trailingEps", 0), 2),
                "📈 ROE (%)": round(info.get("returnOnEquity", 0) * 100, 2),
                "📉 ROA (%)": round(info.get("returnOnAssets", 0) * 100, 2),
                "🏦 Deuda/Capital": round(info.get("debtToEquity", 0), 2),
            })
        except Exception as e:
            st.warning(f"⚠️ No se pudo obtener datos de {symbol}: {e}")

    if fundamental_data:
        st.dataframe(pd.DataFrame(fundamental_data).set_index("Símbolo"))

    # --- NUEVO: Recomendaciones Comparativas Finales (Fase 2 - Versión inicial) ---
    st.subheader("🏆 Resumen y Mejores Oportunidades")
    if all_recommendations:
        recommendations_df = pd.DataFrame(all_recommendations)
        st.dataframe(recommendations_df)

        # Lógica de "mejor empresa" - Puedes personalizar esto
        # Ejemplo: Buscar la señal de compra más fuerte con mayor cambio porcentual predicho
        buy_signals = recommendations_df[recommendations_df['Señal Actual'] == "🟢 Comprar"].copy()
        
        # Convertir 'Cambio Predicho (%)' a numérico para poder ordenar
        if not buy_signals.empty:
            buy_signals['Cambio Predicho Num'] = buy_signals['Cambio Predicho (%)'].str.replace('%', '').astype(float)
            best_buy = buy_signals.sort_values(by='Cambio Predicho Num', ascending=False).iloc[0]
            st.success(f"**Mejor Oportunidad de COMPRA:** {best_buy['Símbolo']} (Señal: {best_buy['Señal Actual']}, Predicción de Cambio: {best_buy['Cambio Predicho (%)']})")
            st.markdown(f"**Diagnóstico:** {best_buy['Recomendación General']}")
        else:
            st.info("No se encontraron oportunidades de compra claras en los símbolos analizados.")

        sell_signals = recommendations_df[recommendations_df['Señal Actual'] == "🔴 Vender"].copy()
        if not sell_signals.empty:
            sell_signals['Cambio Predicho Num'] = sell_signals['Cambio Predicho (%)'].str.replace('%', '').astype(float)
            best_sell = sell_signals.sort_values(by='Cambio Predicho Num', ascending=True).iloc[0] # Ascending para el más negativo
            st.error(f"**Mejor Oportunidad de VENTA:** {best_sell['Símbolo']} (Señal: {best_sell['Señal Actual']}, Predicción de Cambio: {best_sell['Cambio Predicho (%)']})")
            st.markdown(f"**Diagnóstico:** {best_sell['Recomendación General']}")
        else:
            st.info("No se encontraron oportunidades de venta claras en los símbolos analizados.")
    else:
        st.info("No hay símbolos analizados para generar recomendaciones comparativas.")