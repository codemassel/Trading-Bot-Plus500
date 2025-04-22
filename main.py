# pip install pandas numpy matplotlib

# Sharpe Ratio
# (Durchschnittliche Rendite des Portfolios − Risikofreier Zinssatz) / Standardabweichung der Portfolio-Rendite
# Wenn > 1: Attraktive Rendite im Vergleich zum Risiko 

# Max Drawdown
# ((Höchster Wert des Portfolios - Tiefster Punkt) /  Höchster Wert ) * 100
# Also Größter Rückgang vom Höchstwert des Portfolios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time

# --- Binance API Loader: Lade bis zu 1000 Daten pro Aufruf ---
def get_historical_binance_data(symbol="BTCUSDT", interval="1m", lookback_minutes=10080):
    limit = 1000  # max pro Anfrage
    df_all = pd.DataFrame()
    end_time = int(time.time() * 1000)
    loops = int(lookback_minutes / (limit)) + 1

    for _ in range(loops):
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": end_time
        }
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df_all = pd.concat([df, df_all], ignore_index=True)

        end_time = int(data[0][0]) - 1  # Neues Enddatum auf ersten Timestamp -1 setzen
        time.sleep(0.2)  # delay wegen Rate-Limit

    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    return df_all.sort_values('timestamp')

# --- Strategie 1: EMA Crossover ---
def strategy_ema_crossover(df):
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA21'] = df['close'].ewm(span=21).mean()
    df['ema_signal'] = np.where(df['EMA9'] > df['EMA21'], 1, -1)
    return df

# --- Strategie 2: RSI Reversal ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def strategy_rsi(df):
    df['RSI'] = compute_rsi(df['close'])
    df['rsi_signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    return df

# --- Strategie 3: Breakout High ---
def strategy_breakout(df, lookback=20):
    df['rolling_high'] = df['high'].rolling(window=lookback).max()
    df['breakout_signal'] = np.where(df['close'] > df['rolling_high'].shift(1), 1, 0)
    return df

# --- Strategie 4: VWAP Bounce ---
def strategy_vwap(df):
    df['cum_vol'] = df['volume'].cumsum()
    df['cum_vol_x_price'] = (df['volume'] * df['close']).cumsum()
    df['VWAP'] = df['cum_vol_x_price'] / df['cum_vol']
    df['vwap_signal'] = np.where((df['close'] < df['VWAP']) & (df['close'].shift(1) > df['VWAP']), 1, 0)
    return df

# --- Combine Signals ---
def combine_signals(df):
    df['combined_signal'] = df[['ema_signal', 'rsi_signal', 'breakout_signal', 'vwap_signal']].sum(axis=1)
    df['final_signal'] = np.where(df['combined_signal'] >= 2, 1, np.where(df['combined_signal'] <= -2, -1, 0))
    return df

# --- Backtesting Funktion ---
def backtest(df, initial_balance=10000):
    # Sicherstellen, dass 'final_signal' keine NaN-Werte enthält
    df['final_signal'].fillna(0, inplace=True)

    df['position'] = df['final_signal'].shift(1).fillna(0)
    df['market_return'] = df['close'].pct_change()
    
    # Fülle NaN-Werte in 'strategy_return' mit 0
    df['strategy_return'] = df['market_return'] * df['position']
    df['strategy_return'].fillna(0, inplace=True)
    
    # Berechnung der Equity Curve
    df['equity_curve'] = (1 + df['strategy_return']).cumprod() * initial_balance
    df['equity_curve'].fillna(method='ffill', inplace=True)
    
    # Berechnung der Total Return
    total_return = (df['equity_curve'].iloc[-1] - initial_balance) / initial_balance * 100
    print(f"Total Return: {total_return:.2f}%")
    
    # Berechnung des Sharpe Ratios
    sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)  # 252 Handelstage
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Berechnung des Max Drawdowns
    max_drawdown = (df['equity_curve'].min() - df['equity_curve'].max()) / df['equity_curve'].max() * 100
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Speichern des Logs
    df[['timestamp', 'equity_curve', 'final_signal']].to_csv('trading_log.csv', index=False)

    print(df[['timestamp', 'final_signal', 'equity_curve']].head())
    
    return df


# --- Visualisiere Equity + Entry/Exit Signale ---
def plot_equity(df, filename='equity_curve_entry_exit.png'):
    plt.figure(figsize=(14, 6))
    plt.plot(df['equity_curve'], label='Equity Curve', color='blue')
    buy_signals = df[df['final_signal'] == 1]
    sell_signals = df[df['final_signal'] == -1]
    plt.scatter(buy_signals.index, df.loc[buy_signals.index, 'equity_curve'], label='Buy', marker='^', color='green')
    plt.scatter(sell_signals.index, df.loc[sell_signals.index, 'equity_curve'], label='Sell', marker='v', color='red')
    plt.legend()
    plt.title('Equity Curve mit Entry/Exit Signalen')
    plt.grid(True)
    plt.savefig(filename)  


# --- Visualisiere Candlestick Chart mit Buy/Sell Signalen ---
def plot_signals(df, filename='equity_curve_buy_sell.png'):
    plt.figure(figsize=(14, 6))
    plt.plot(df['close'], label='Close Price', color='black', alpha=0.6)
    buy_signals = df[df['final_signal'] == 1]
    sell_signals = df[df['final_signal'] == -1]
    plt.scatter(buy_signals.index, df.loc[buy_signals.index, 'close'], label='Buy', marker='^', color='green')
    plt.scatter(sell_signals.index, df.loc[sell_signals.index, 'close'], label='Sell', marker='v', color='red')
    plt.legend()
    plt.title('Close Price mit Entry/Exit Signalen')
    plt.grid(True)
    plt.savefig(filename)  

# --- Berechne Performance-Metriken ---
def performance_metrics(df):
    total_return = df['equity_curve'].iloc[-1] / df['equity_curve'].iloc[0] - 1
    sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252 * 24 * 60)
    max_drawdown = (df['equity_curve'] / df['equity_curve'].cummax() - 1).min()
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

# --- Speichere als CSV ---
def save_to_csv(df, filename="trading_log.csv"):
    df.to_csv(filename)
    print(f"Gespeichert als {filename}")

# --- Hauptfunktion ---
def run_bot():
    df = get_historical_binance_data(lookback_minutes=43200)  # 30 Tage 1-Minuten-Kerzen
    df.set_index('timestamp', inplace=True, drop=False)

    df = strategy_ema_crossover(df)
    df = strategy_rsi(df)
    df = strategy_breakout(df)
    df = strategy_vwap(df)
    df = combine_signals(df)
    df = backtest(df)

    plot_equity(df, filename='equity_curve_entry_exit.png')
    plot_signals(df, filename='equity_curve_buy_sell.png')
    performance_metrics(df)
    save_to_csv(df)

    plt.show()

    return df

df_result = run_bot()

# TODO
# return noch nan -> Liegt an den strategies.. Übergang: NaN auf 10k gesetzt
# Buy sell signale noch bullshit -> Logik?
