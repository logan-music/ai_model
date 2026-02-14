import os
import pickle
import numpy as np
import pandas as pd
from binance.client import Client

UNIVERSAL_MODEL = None
UNIVERSAL_SCALER = None
UNIVERSAL_FEATURES = None

def load_universal_model(model_file='universal_scalp_model.pkl',
                         scaler_file='universal_scalp_scaler.pkl',
                         features_file='universal_features.pkl'):
    global UNIVERSAL_MODEL, UNIVERSAL_SCALER, UNIVERSAL_FEATURES
    try:
        with open(model_file, 'rb') as f:
            UNIVERSAL_MODEL = pickle.load(f)
        with open(scaler_file, 'rb') as f:
            UNIVERSAL_SCALER = pickle.load(f)
        with open(features_file, 'rb') as f:
            UNIVERSAL_FEATURES = pickle.load(f)
        print("Loaded universal model and features.")
        return UNIVERSAL_MODEL, UNIVERSAL_SCALER, UNIVERSAL_FEATURES
    except Exception as e:
        print("Error loading model:", e)
        return None, None, None

def calculate_universal_features(df):
    """
    Reproduce feature engineering exactly as training (per-coin).
    Expects df with columns: open, high, low, close, volume and timestamp index.
    """
    df = df.copy()
    # returns
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_15'] = df['close'].pct_change(15)
    # sma/ema ratios
    for period in [5, 10, 20, 50]:
        sma = df['close'].rolling(window=period, min_periods=1).mean()
        df[f'sma_{period}_ratio'] = df['close'] / (sma + 1e-10)
    for period in [3, 7, 10, 25, 50]:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_ratio'] = df['close'] / (ema + 1e-10)
    ema_3 = df['close'].ewm(span=3, adjust=False).mean()
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_7 = df['close'].ewm(span=7, adjust=False).mean()
    ema_25 = df['close'].ewm(span=25, adjust=False).mean()
    df['ema_3_10_ratio'] = ema_3 / (ema_10 + 1e-10)
    df['ema_7_25_ratio'] = ema_7 / (ema_25 + 1e-10)
    # rsi
    delta = df['close'].diff()
    gain = delta.mask(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    loss = (-delta).mask(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    # macd normalized
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_pct'] = macd / (df['close'] + 1e-10)
    df['macd_signal_pct'] = macd_signal / (df['close'] + 1e-10)
    df['macd_histogram_pct'] = (macd - macd_signal) / (df['close'] + 1e-10)
    df['macd_bullish'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)
    # bb
    bb_middle = df['close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    df['bb_bandwidth'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    df['bb_position'] = (df['close'] - bb_lower) / ((bb_upper - bb_lower) + 1e-10)
    # stochastic
    low_min = df['low'].rolling(window=14, min_periods=1).min()
    high_max = df['high'].rolling(window=14, min_periods=1).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
    # atr %
    high_low = (df['high'] - df['low']).abs()
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14, min_periods=1).mean()
    df['atr_pct'] = atr / (df['close'] + 1e-10)
    # volatility
    returns = df['close'].pct_change().fillna(0)
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = returns.rolling(window=period, min_periods=1).std()
    # volume ratios
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (volume_ma_20 + 1e-10)
    df['volume_roc'] = df['volume'].pct_change(5).fillna(0)
    # obv normalized and vwap (per coin)
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    obv_mean = obv.rolling(window=20, min_periods=1).mean()
    df['obv_normalized'] = obv / (obv_mean + 1e-10)
    cum_vp = (df['close'] * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    vwap = cum_vp / (cum_vol + 1e-10)
    df['price_vwap_ratio'] = df['close'] / (vwap + 1e-10)
    # price action
    df['body_pct'] = (df['close'] - df['open']).abs() / (df['open'] + 1e-10)
    df['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-10)
    df['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-10)
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['up_streak'] = (df['consecutive_up'].groupby((df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()).cumsum()).fillna(0)
    df['momentum_5_pct'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-10)
    df['momentum_10_pct'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)
    return df

def get_live_features_for_any_coin(client, symbol, interval='5m', limit=200):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df.set_index('timestamp', inplace=True)
        df_feat = calculate_universal_features(df)
        return df_feat
    except Exception as e:
        print(f"Error fetching live klines for {symbol}: {e}")
        return None

def get_universal_ai_prediction(client, symbol, min_confidence=0.70):
    global UNIVERSAL_MODEL, UNIVERSAL_SCALER, UNIVERSAL_FEATURES
    if UNIVERSAL_MODEL is None:
        UNIVERSAL_MODEL, UNIVERSAL_SCALER, UNIVERSAL_FEATURES = load_universal_model()
    if UNIVERSAL_MODEL is None:
        return 'HOLD', 0.5, 0.5
    try:
        features_df = get_live_features_for_any_coin(client, symbol)
        if features_df is None or len(features_df) == 0:
            return 'HOLD', 0.5, 0.5
        # take last row and select universal features (ensure same order)
        last = features_df.iloc[-1:]
        # fill missing features with 0
        missing = [f for f in UNIVERSAL_FEATURES if f not in last.columns]
        for m in missing:
            last[m] = 0.0
        X = last[UNIVERSAL_FEATURES].values
        Xs = UNIVERSAL_SCALER.transform(X)
        try:
            probs = UNIVERSAL_MODEL.predict_proba(Xs)[0]
            buy_prob = float(probs[1])
        except Exception:
            buy_prob = float(UNIVERSAL_MODEL.predict(Xs)[0])
            # fallback
        conf = abs(buy_prob - 0.5) * 2
        if buy_prob >= min_confidence:
            return 'BUY', buy_prob, conf
        elif buy_prob < 0.3:
            return 'SELL', buy_prob, conf
        else:
            return 'HOLD', buy_prob, conf
    except Exception as e:
        print(f"AI prediction error for {symbol}: {e}")
        return 'HOLD', 0.5, 0.5

if __name__ == "__main__":
    # quick test (requires BINANCE_API_KEY/SECRET set)
    client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
    UNIVERSAL_MODEL, UNIVERSAL_SCALER, UNIVERSAL_FEATURES = load_universal_model()
    if UNIVERSAL_MODEL is None:
        print("Model not found. Train first.")
    else:
        for s in ['BTCUSDT', 'ETHUSDT', 'SHIBUSDT']:
            try:
                df = get_live_features_for_any_coin(client, s)
                if df is not None and len(df) > 0:
                    last = df.iloc[-1:]
                    print(s, "-> features ready, sample:")
                    print(last[UNIVERSAL_FEATURES].iloc[0].to_dict())
                else:
                    print(s, "-> no features")
            except Exception as e:
                print("Error test:", e)