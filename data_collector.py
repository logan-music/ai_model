#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import warnings
warnings.filterwarnings('ignore')


# Default coins for universal training
DEFAULT_TRAINING_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOGEUSDT', 'SHIBUSDT', 'ADAUSDT',
    'LUNAUSDT', 'PEPEUSDT', 'SUIUSDT', 'UNIUSDT',
    'FILUSDT', 'NEARUSDT', 'ICPUSDT', 'APEUSDT',
    'RAYUSDT', 'RUNEUSDT', 'MORPHOUSDT', 'PENGUUSDT',
    'ASTERUSDT', 'TIAUSDT', 'ALPINEUSDT', 'PROVEUSDT',
    'FORTHUSDT', 
]

class UniversalDataCollector:
    def __init__(self, api_key=None, api_secret=None, use_testnet=False):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        if use_testnet:
            self.client = Client(self.api_key, self.api_secret, testnet=True)
        else:
            self.client = Client(self.api_key, self.api_secret)
        print(f"✅ Universal Data Collector initialized (Testnet: {use_testnet})")

    def get_historical_klines(self, symbol, interval, lookback_days=30):
        try:
            start_time = datetime.utcnow() - timedelta(days=lookback_days)
            start_str = start_time.strftime("%d %b %Y %H:%M:%S")

            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )

            if not klines:
                return None

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)
            # Keep full set for per-coin feature engineering
            return df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]

        except Exception as e:
            print(f"      ❌ Error fetching {symbol}: {e}")
            return None


class UniversalFeatureEngineer:
    """
    Feature engineering PER COIN to prevent cross-coin leakage.
    """

    @staticmethod
    def add_normalized_features_coinwise(coin_df):
        """
        Engineer features for a single coin DataFrame (index=timestamp).
        All rolling/ewm/vwap calculations are done only within this coin.
        """
        df = coin_df.copy()

        # PRICE RETURNS (percent)
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_15'] = df['close'].pct_change(15)

        # SMA / EMA ratios (ratios to avoid absolute scale)
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(window=period, min_periods=1).mean()
            df[f'sma_{period}_ratio'] = df['close'] / (sma + 1e-10)

        for period in [3, 7, 10, 25, 50]:
            ema = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_ratio'] = df['close'] / (ema + 1e-10)

        # EMA crossovers
        ema_3 = df['close'].ewm(span=3, adjust=False).mean()
        ema_10 = df['close'].ewm(span=10, adjust=False).mean()
        ema_7 = df['close'].ewm(span=7, adjust=False).mean()
        ema_25 = df['close'].ewm(span=25, adjust=False).mean()
        df['ema_3_10_ratio'] = ema_3 / (ema_10 + 1e-10)
        df['ema_7_25_ratio'] = ema_7 / (ema_25 + 1e-10)

        # RSI
        delta = df['close'].diff()
        gain = delta.mask(delta < 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta).mask(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        # MACD normalized
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_pct'] = macd / (df['close'] + 1e-10)
        df['macd_signal_pct'] = macd_signal / (df['close'] + 1e-10)
        df['macd_histogram_pct'] = (macd - macd_signal) / (df['close'] + 1e-10)
        df['macd_bullish'] = ((macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))).astype(int)

        # Bollinger bands position
        bb_middle = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std().fillna(0)
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_bandwidth'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        df['bb_position'] = (df['close'] - bb_lower) / ((bb_upper - bb_lower) + 1e-10)

        # Stochastic
        low_min = df['low'].rolling(window=14, min_periods=1).min()
        high_max = df['high'].rolling(window=14, min_periods=1).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()

        # ATR % of price
        high_low = (df['high'] - df['low']).abs()
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14, min_periods=1).mean()
        df['atr_pct'] = atr / (df['close'] + 1e-10)

        # Volatility
        returns = df['close'].pct_change().fillna(0)
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = returns.rolling(window=period, min_periods=1).std()

        # Volume ratios (per coin)
        volume_ma_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        volume_ma_20 = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (volume_ma_20 + 1e-10)
        df['volume_roc'] = df['volume'].pct_change(5).fillna(0)

        # OBV normalized per coin
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        obv_mean = obv.rolling(window=20, min_periods=1).mean()
        df['obv_normalized'] = obv / (obv_mean + 1e-10)

        # VWAP per coin
        cum_vp = (df['close'] * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        vwap = cum_vp / (cum_vol + 1e-10)
        df['price_vwap_ratio'] = df['close'] / (vwap + 1e-10)

        # Price action percentages
        df['body_pct'] = (df['close'] - df['open']).abs() / (df['open'] + 1e-10)
        df['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-10)
        df['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-10)
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Momentum / streaks (per coin)
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        # Group-based streak (resets correctly per coin index)
        df['up_streak'] = (df['consecutive_up'].groupby((df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()).cumsum()).fillna(0)
        df['momentum_5_pct'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-10)
        df['momentum_10_pct'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-10)

        return df

    @classmethod
    def engineer_all_universal_features_for_symbol(cls, df, symbol):
        """
        Engineer features for a single symbol and attach symbol column.
        """
        df_feat = cls.add_normalized_features_coinwise(df)
        df_feat = df_feat.dropna()
        df_feat['symbol'] = symbol
        return df_feat


def collect_universal_dataset(symbols=None, interval='5m', lookback_days=30,
                              save_to='universal_features.csv'):
    """
    Collect multi-coin data and engineer features per coin (no cross-coin leakage).
    """
    if symbols is None:
        symbols = DEFAULT_TRAINING_COINS

    collector = UniversalDataCollector(use_testnet=False)
    all_feature_frames = []
    successful = 0
    failed = 0

    print(f"\nCollecting for {len(symbols)} coins (per-coin feature engineering)...\n")
    for symbol in symbols:
        try:
            df_raw = collector.get_historical_klines(symbol, interval, lookback_days)
            if df_raw is None or len(df_raw) < 50:
                print(f"  {symbol}: insufficient data, skipping")
                failed += 1
                continue
            df_feat = UniversalFeatureEngineer.engineer_all_universal_features_for_symbol(df_raw, symbol)
            if df_feat is None or len(df_feat) == 0:
                print(f"  {symbol}: feature engineering returned empty, skipping")
                failed += 1
                continue
            all_feature_frames.append(df_feat)
            successful += 1
            print(f"  {symbol}: engineered {len(df_feat):,} rows")
            time.sleep(0.15)
        except Exception as e:
            print(f"  {symbol}: error {e}")
            failed += 1
            continue

    if not all_feature_frames:
        print("❌ No coin data collected/engineered.")
        return None

    # concat (safe because all features computed per-coin)
    combined = pd.concat(all_feature_frames, ignore_index=False)
    # keep index name as timestamp. Do not sort across coins (we will rely on per-coin time-aware operations later)
    # but for convenience, sort by index then symbol to keep deterministic order
    combined = combined.sort_index().reset_index().set_index('timestamp')

    print(f"\nCombined dataset: {len(combined):,} rows, coins: {combined['symbol'].nunique()}")
    if save_to:
        combined.to_csv(save_to)
        print(f"Saved dataset to {save_to} ({os.path.getsize(save_to)/1024/1024:.2f} MB)")

    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Universal Multi-coin Data Collector (leak-safe)')
    parser.add_argument('--coins', nargs='+', default=None, help='Coins to collect')
    parser.add_argument('--interval', type=str, default='5m')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--output', type=str, default='universal_features.csv')
    args = parser.parse_args()

    coins = None
    if args.coins:
        coins = [c.upper() + 'USDT' if not c.upper().endswith('USDT') else c.upper() for c in args.coins]

    df = collect_universal_dataset(symbols=coins, interval=args.interval, lookback_days=args.days, save_to=args.output)
    if df is not None:
        print("\n✅ Universal dataset collection complete!")
    else:
        print("\n❌ Dataset collection failed!")