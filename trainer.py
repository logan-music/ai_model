#!/usr/bin/env python3
import os
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# =========================================================
# HYPERPARAMETERS - Badilisha hapa tu
# =========================================================
DEFAULT_MIN_PROB      = 0.68   # Ongezwa kutoka 0.65 -> kuchagua zaidi
DEFAULT_PROB_EDGE     = 0.20   # Ongezwa kutoka 0.15
DEFAULT_TOP_PCT       = 0.15   # Punguza kutoka 0.20 -> signals bora zaidi
BASE_POSITION_PCT     = 0.005
MAX_POSITION_PCT      = 0.02
ATR_MEDIAN_WINDOW     = 20
VOL_BREAKOUT_MULT     = 1.2    # Ongezwa - volatility breakout ya kweli zaidi

# Train/Test split strategy
TRAIN_TEST_STRATEGY   = "time"   # "time" = temporal split, "coin" = separate coins
TEST_COINS            = ["ADAUSDT", "ALPINEUSDT", "APEUSDT", "ASTERUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "FILUSDT", "FORTHUSDT", "ICPUSDT", "LUNAUSDT", "MORPHOUSDT", "NEARUSDT", "PENGUUSDT", "PEPEUSDT", "PROVEUSDT", "RAYUSDT", "RUNEUSDT", "SHIBUSDT", "SOLUSDT", "SUIUSDT", "TIAUSDT", "UNIUSDT", "XRPUSDT"]       # Weka coins hapa kama unataka separate test coins
                                  # Mfano: ["PEPEUSDT", "SUIUSDT", "NEARUSDT"]
                                  # Kama tupu, itatumia temporal split
TRAIN_RATIO           = 0.75    # 75% ya data kwa training (temporal)

# Labeling
STRONG_SIGNAL_MULT    = 2.5     # Ongezwa kutoka 1.0 -> precision bora
MIN_NET_RETURN        = 0.004   # 0.4% minimum net return ili label=1
LOOKFORWARD_CANDLES   = 20      # Ongezwa kutoka 8 -> dakika 20

# Feature selection
MAX_FEATURES          = 40      # Punguza features nyingi (reduce overfitting)
MIN_MUTUAL_INFO       = 0.001   # Punguza features zenye mutual info ndogo
CORR_THRESHOLD        = 0.92    # Punguza features zenye correlation kubwa

# Threshold optimization
MIN_PRECISION_FOR_THR = 0.40    # Precision minimum kabla ya kuchagua threshold
MIN_TRADES_FOR_THR    = 15      # Minimum trades kwa threshold evaluation


# =========================================================
# FEATURE ENGINEERING ILIYOBORESHWA
# =========================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ongeza features bora zaidi kwa kila row.
    Inategemea: open, high, low, close, volume columns.
    Inafanya kazi per-coin ili kuepuka data leakage.
    """
    df = df.copy()

    # --- Candle body & wick features ---
    df['body_pct']       = (df['close'] - df['open']).abs() / (df['open'].replace(0, np.nan) + 1e-12)
    df['upper_wick_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['open'].replace(0, np.nan) + 1e-12)
    df['lower_wick_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['open'].replace(0, np.nan) + 1e-12)
    df['is_bullish']     = (df['close'] > df['open']).astype(int)
    df['body_ratio']     = df['body_pct'] / (df['body_pct'] + df['upper_wick_pct'] + df['lower_wick_pct'] + 1e-12)

    # --- Returns ---
    df['ret_1']  = df['close'].pct_change(1)
    df['ret_3']  = df['close'].pct_change(3)
    df['ret_5']  = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_15'] = df['close'].pct_change(15)

    # --- Momentum ---
    df['mom_5_10']  = df['ret_5'] - df['ret_10']
    df['mom_acc']   = df['ret_1'] - df['ret_3']   # Acceleration ya momentum
    df['ret_std_5'] = df['ret_1'].rolling(5, min_periods=2).std()
    df['ret_std_10']= df['ret_1'].rolling(10, min_periods=3).std()

    # --- Volume features ---
    vol_ma5  = df['volume'].rolling(5, min_periods=1).mean()
    vol_ma20 = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_5']  = df['volume'] / (vol_ma5 + 1e-12)
    df['vol_ratio_20'] = df['volume'] / (vol_ma20 + 1e-12)
    df['vol_spike']    = (df['vol_ratio_5'] > 2.0).astype(int)
    df['vol_trend']    = vol_ma5 / (vol_ma20 + 1e-12)

    # Volume-price relationship
    df['vol_price_corr'] = df['ret_1'] * df['vol_ratio_5']
    df['buy_pressure']   = df['vol_ratio_5'] * df['is_bullish']

    # --- Moving averages ---
    for w in [3, 5, 10, 20, 50]:
        ma = df['close'].rolling(w, min_periods=max(1, w//2)).mean()
        df[f'ma{w}_ratio'] = df['close'] / (ma + 1e-12)
        df[f'ma{w}_slope'] = ma.pct_change(1)

    # EMA crossovers
    ema5  = df['close'].ewm(span=5,  adjust=False).mean()
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    ema50 = df['close'].ewm(span=50, adjust=False).mean()
    df['ema5_10_cross']  = ema5 / (ema10 + 1e-12) - 1
    df['ema10_20_cross'] = ema10 / (ema20 + 1e-12) - 1
    df['ema20_50_cross'] = ema20 / (ema50 + 1e-12) - 1
    df['ema5_slope']     = ema5.pct_change(1)
    df['ema10_slope']    = ema10.pct_change(1)

    # --- RSI ---
    def _rsi(series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs    = gain / (loss + 1e-12)
        return 100 - (100 / (1 + rs))

    df['rsi_7']  = _rsi(df['close'], 7)
    df['rsi_14'] = _rsi(df['close'], 14)
    df['rsi_21'] = _rsi(df['close'], 21)
    df['rsi_slope'] = df['rsi_14'].diff(1)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_oversold']   = (df['rsi_14'] < 30).astype(int)

    # --- Bollinger Bands ---
    bb_mid = df['close'].rolling(20, min_periods=5).mean()
    bb_std = df['close'].rolling(20, min_periods=5).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-12)
    df['bb_width']    = (bb_upper - bb_lower) / (bb_mid + 1e-12)
    df['bb_squeeze']  = (df['bb_width'] < df['bb_width'].rolling(20, min_periods=5).quantile(0.2)).astype(int)

    # --- ATR ---
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14']    = tr.rolling(14, min_periods=5).mean()
    df['atr_pct']   = df['atr_14'] / (df['close'] + 1e-12)
    df['atr_ratio'] = df['atr_pct'] / (df['atr_pct'].rolling(20, min_periods=5).mean() + 1e-12)

    # --- Price position ---
    high_20 = df['high'].rolling(20, min_periods=5).max()
    low_20  = df['low'].rolling(20, min_periods=5).min()
    df['price_pos_20'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-12)

    high_5 = df['high'].rolling(5, min_periods=2).max()
    low_5  = df['low'].rolling(5, min_periods=2).min()
    df['price_pos_5']  = (df['close'] - low_5) / (high_5 - low_5 + 1e-12)

    # --- Consecutive candles ---
    df['consec_bull'] = df['is_bullish'].groupby(
        (df['is_bullish'] != df['is_bullish'].shift()).cumsum()
    ).cumcount() + 1
    df['consec_bull'] = df['consec_bull'] * df['is_bullish']

    # --- MACD ---
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df['macd']          = macd / (df['close'] + 1e-12)
    df['macd_signal']   = signal_line / (df['close'] + 1e-12)
    df['macd_hist']     = (macd - signal_line) / (df['close'] + 1e-12)
    df['macd_cross_up'] = ((macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))).astype(int)

    # --- Stochastic ---
    low_14  = df['low'].rolling(14, min_periods=5).min()
    high_14 = df['high'].rolling(14, min_periods=5).max()
    df['stoch_k'] = (df['close'] - low_14) / (high_14 - low_14 + 1e-12) * 100
    df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean()

    return df


# =========================================================
# LABELING ILIYOBORESHWA - Precision-focused
# =========================================================
def label_coin_precision_focused(coin_df: pd.DataFrame,
                                  trig_mult: float,
                                  trail_mult: float,
                                  fee_rate: float,
                                  target_profit: float,
                                  stop_loss: float,
                                  slippage: float,
                                  lookforward: int,
                                  strong_signal_mult: float = STRONG_SIGNAL_MULT,
                                  min_net_return: float = MIN_NET_RETURN) -> pd.DataFrame:
    """
    Labeling iliyoboreshwa:
    - label=1 tu kama net_return > target_profit * strong_signal_mult
    - na pia net_return > min_net_return (absolute minimum)
    - Hii inapunguza BUY labels lakini zinakuwa za ubora zaidi (precision bora)
    """
    coin_df = coin_df.copy().sort_index()
    n = len(coin_df)

    highs  = coin_df['high'].values
    lows   = coin_df['low'].values
    closes = coin_df['close'].values
    atrs   = coin_df['atr_pct'].values if 'atr_pct' in coin_df.columns else np.zeros(n)

    labels          = np.zeros(n, dtype=int)
    trailing_returns= np.full(n, np.nan)

    strong_threshold = target_profit * strong_signal_mult

    for i in range(n - 1):
        entry_price = closes[i]
        atr_pct     = float(atrs[i]) if not np.isnan(atrs[i]) else 0.0

        # Dynamic TP/SL
        tp_pct = max(target_profit, atr_pct * 3.0)
        sl_pct = max(stop_loss,     atr_pct * 2.0)
        tp_price = entry_price * (1 + tp_pct + fee_rate + slippage)
        sl_price = entry_price * (1 - sl_pct - slippage)

        # Trailing simulation
        atr_abs = atr_pct * entry_price
        trigger_price   = entry_price * (1 + trig_mult * atr_pct) if atr_pct > 0 else entry_price * (1 + trig_mult * 0.001)
        trailing_active = False
        trailing_stop   = None
        best_high       = entry_price
        net_return      = np.nan
        hit_sl          = False

        for j in range(i + 1, min(n, i + 1 + lookforward)):
            h = highs[j]
            l = lows[j]

            # Stop loss check first
            if l <= sl_price:
                exit_rev   = sl_price * (1 - fee_rate - slippage)
                cost       = entry_price * (1 + fee_rate + slippage)
                net_return = (exit_rev - cost) / (cost + 1e-12)
                hit_sl     = True
                break

            if h > best_high:
                best_high = h

            if not trailing_active:
                if h >= trigger_price:
                    trailing_active = True
                    trailing_stop   = max(
                        entry_price * (1 - slippage - fee_rate),
                        best_high - trail_mult * atr_abs
                    )
            else:
                candidate = best_high - trail_mult * atr_abs
                if candidate > trailing_stop:
                    trailing_stop = candidate
                if l <= trailing_stop:
                    exit_rev   = trailing_stop * (1 - fee_rate - slippage)
                    cost       = entry_price * (1 + fee_rate + slippage)
                    net_return = (exit_rev - cost) / (cost + 1e-12)
                    break

        # Kama haijafika trailing/SL, tumia best_high
        if np.isnan(net_return) and not hit_sl:
            exit_rev   = best_high * (1 - fee_rate - slippage)
            cost       = entry_price * (1 + fee_rate + slippage)
            net_return = (exit_rev - cost) / (cost + 1e-12)

        trailing_returns[i] = net_return

        # LABELING ILIYOBORESHWA: criteria mbili lazima zikutane
        if np.isfinite(net_return) and not hit_sl:
            if net_return >= strong_threshold and net_return >= min_net_return:
                labels[i] = 1

    coin_df['trailing_return'] = trailing_returns
    coin_df['target']          = labels
    return coin_df


# =========================================================
# TRAILING OPTIMIZER
# =========================================================
def optimize_trailing_per_coin(coin_df: pd.DataFrame,
                                fee_rate: float,
                                target_profit: float,
                                stop_loss: float,
                                slippage: float,
                                lookforward: int) -> tuple:
    """Grid search kwa best trailing params - inatumia train data tu."""
    trigger_grid = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    trail_grid   = [0.5, 0.8, 1.0, 1.2, 1.6, 2.0]

    highs  = coin_df['high'].values
    lows   = coin_df['low'].values
    closes = coin_df['close'].values
    atrs   = coin_df['atr_pct'].values if 'atr_pct' in coin_df.columns else np.zeros(len(closes))
    n      = len(closes)

    idxs = range(0, n, max(1, n // 1500))
    best_score  = -np.inf
    best_params = (1.0, 1.2)

    for trig in trigger_grid:
        for trail in trail_grid:
            returns = []
            for i in idxs:
                ep      = closes[i]
                atr_pct = float(atrs[i]) if not np.isnan(atrs[i]) else 0.0
                atr_abs = atr_pct * ep
                trigger = ep * (1 + trig * atr_pct) if atr_pct > 0 else ep * (1 + trig * 0.001)
                sl_p    = ep * (1 - max(stop_loss, atr_pct * 2.0) - slippage)
                trailing_active = False
                trailing_stop   = None
                best_high       = ep
                nr = np.nan

                for j in range(i + 1, min(n, i + 1 + lookforward)):
                    h = highs[j]; l = lows[j]
                    if l <= sl_p:
                        exit_r = sl_p * (1 - fee_rate - slippage)
                        nr = (exit_r - ep * (1 + fee_rate + slippage)) / (ep * (1 + fee_rate + slippage) + 1e-12)
                        break
                    if h > best_high: best_high = h
                    if not trailing_active:
                        if h >= trigger:
                            trailing_active = True
                            trailing_stop = max(ep * (1 - slippage - fee_rate), best_high - trail * atr_abs)
                    else:
                        c = best_high - trail * atr_abs
                        if c > trailing_stop: trailing_stop = c
                        if l <= trailing_stop:
                            exit_r = trailing_stop * (1 - fee_rate - slippage)
                            nr = (exit_r - ep * (1 + fee_rate + slippage)) / (ep * (1 + fee_rate + slippage) + 1e-12)
                            break

                if np.isnan(nr):
                    exit_r = best_high * (1 - fee_rate - slippage)
                    nr = (exit_r - ep * (1 + fee_rate + slippage)) / (ep * (1 + fee_rate + slippage) + 1e-12)
                if np.isfinite(nr): returns.append(nr)

            score = np.mean(returns) if returns else -np.inf
            if score > best_score:
                best_score  = score
                best_params = (trig, trail)

    return best_params, best_score


# =========================================================
# FEATURE SELECTION ILIYOBORESHWA
# =========================================================
def select_features_smart(df: pd.DataFrame,
                           target_col: str = 'target',
                           max_features: int = MAX_FEATURES,
                           min_mi: float = MIN_MUTUAL_INFO,
                           corr_thresh: float = CORR_THRESHOLD) -> list:
    """
    Feature selection kwa hatua 3:
    1. Punguza columns za lookahead au target-related
    2. Mutual information filter
    3. Correlation pruning (punguza redundant features)
    """
    exclude = {
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'trades', 'future_return', 'trailing_return', 'target',
        'symbol', 'signal_volume_ok', 'pred_prob'
    }
    candidates = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, float, int]
        and df[c].notna().sum() > len(df) * 0.6
    ]

    print(f"  Candidate features: {len(candidates)}")

    # Punguza infinities
    X = df[candidates].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    y = df[target_col].fillna(0).astype(int)

    if len(y.unique()) < 2:
        print("  ⚠️ Only one class in target — using all candidates")
        return candidates[:max_features]

    # Hatua 1: Mutual information
    try:
        mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        mi_series = pd.Series(mi, index=candidates)
        good_mi   = mi_series[mi_series >= min_mi].sort_values(ascending=False)
        print(f"  After MI filter (>={min_mi}): {len(good_mi)} features")
        candidates_mi = good_mi.index.tolist()
    except Exception as e:
        print(f"  MI filter failed: {e}, using all candidates")
        candidates_mi = candidates

    if not candidates_mi:
        return candidates[:max_features]

    # Hatua 2: Correlation pruning
    X_mi   = X[candidates_mi]
    corr_m = X_mi.corr().abs()
    upper  = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop: continue
        high_corr = upper[col][upper[col] >= corr_thresh].index.tolist()
        to_drop.update(high_corr)

    final_features = [c for c in candidates_mi if c not in to_drop]
    print(f"  After correlation pruning (<{corr_thresh}): {len(final_features)} features")

    # Hatua 3: Chukua top max_features
    if hasattr(good_mi, 'loc'):
        try:
            final_features = sorted(
                final_features,
                key=lambda c: good_mi.get(c, 0),
                reverse=True
            )
        except Exception:
            pass

    final_features = final_features[:max_features]
    print(f"  Final selected features: {len(final_features)}")
    return final_features


# =========================================================
# TRAIN/TEST SPLIT ILIYOBORESHWA
# =========================================================
def make_train_test_split(df: pd.DataFrame,
                           strategy: str = "time",
                           test_coins: list = None,
                           train_ratio: float = TRAIN_RATIO):
    """
    Strategies:
    - "time": Gawanya kwa wakati (train=mapema, test=hivi karibuni) per coin
    - "coin": Gawanya kwa coins (train/test coins tofauti kabisa)
    """
    if strategy == "coin" and test_coins:
        test_coins_set = set(test_coins)
        train_df = df[~df['symbol'].isin(test_coins_set)].copy()
        test_df  = df[df['symbol'].isin(test_coins_set)].copy()
        print(f"  COIN SPLIT:")
        print(f"    Train coins: {sorted(train_df['symbol'].unique())}")
        print(f"    Test coins:  {sorted(test_df['symbol'].unique())}")
        return train_df, test_df

    # TIME SPLIT (bora zaidi - hakuna data leakage)
    print(f"  TIME SPLIT (train={train_ratio*100:.0f}%, test={100-train_ratio*100:.0f}%):")
    train_parts = []
    test_parts  = []
    for sym in df['symbol'].unique():
        coin_df  = df[df['symbol'] == sym].sort_index()
        n        = len(coin_df)
        if n < 30:
            continue
        split    = int(n * train_ratio)
        train_parts.append(coin_df.iloc[:split])
        test_parts.append(coin_df.iloc[split:])

    train_df = pd.concat(train_parts) if train_parts else pd.DataFrame()
    test_df  = pd.concat(test_parts)  if test_parts  else pd.DataFrame()
    print(f"    Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


# =========================================================
# MODEL TRAINING ILIYOBORESHWA
# =========================================================
def train_precision_ensemble(X_train, y_train, calibrate: bool = False):
    """
    Ensemble iliyoboreshwa:
    - RF na class_weight='balanced'
    - GB na scale_pos_weight sahihi (si VotingClassifier - inachanganya probabilities)
    - Tumia RF peke yake kwa precision au ensemble ya calibrated
    """
    n_pos  = int(y_train.sum())
    n_neg  = int(len(y_train) - n_pos)
    ratio  = n_neg / max(n_pos, 1)

    print(f"  Class distribution: BUY={n_pos}, NO_BUY={n_neg}, ratio={ratio:.1f}x")

    # RF iliyoboreshwa - precision-focused
    rf = RandomForestClassifier(
        n_estimators       = 300,
        max_depth          = 10,         # Punguza kutoka 12 -> less overfitting
        min_samples_split  = 50,         # Ongeza -> conservative
        min_samples_leaf   = 25,         # Ongeza -> conservative
        max_features       = 'sqrt',
        class_weight       = 'balanced',
        random_state       = 42,
        n_jobs             = -1,
        max_samples        = 0.8,        # Bagging -> generalization bora
    )

    # GB iliyoboreshwa na class weight sahihi
    # subsample wa chini -> less overfitting
    gb = GradientBoostingClassifier(
        n_estimators  = 200,
        max_depth     = 4,               # Punguza kutoka 6
        learning_rate = 0.03,            # Punguza -> conservative
        subsample     = 0.7,
        min_samples_split = 40,
        min_samples_leaf  = 20,
        random_state  = 42,
    )

    print("  Training RF...")
    rf.fit(X_train, y_train)

    # Resample kwa GB (manual class weight kwa GB)
    if n_pos > 0 and ratio > 1:
        # Oversample BUY class kwa GB
        pos_idx  = np.where(y_train == 1)[0]
        neg_idx  = np.where(y_train == 0)[0]
        n_sample = min(n_pos * 3, n_neg)  # Max 3x oversampling
        pos_sample = np.random.choice(pos_idx, size=min(n_pos * 3, len(neg_idx)), replace=True)
        neg_sample = np.random.choice(neg_idx, size=min(n_pos * 3, len(neg_idx)), replace=False)
        idx_bal  = np.concatenate([pos_sample, neg_sample])
        np.random.shuffle(idx_bal)
        X_bal = X_train[idx_bal]
        y_bal = y_train[idx_bal]
    else:
        X_bal, y_bal = X_train, y_train

    print("  Training GB (balanced)...")
    gb.fit(X_bal, y_bal)

    # Calibrate kama imetakiwa
    if calibrate:
        print("  Calibrating RF...")
        rf = CalibratedClassifierCV(rf, method='isotonic', cv=3)
        rf.fit(X_train, y_train)

    # Ensemble kwa weighted average (RF ina uzito zaidi - precision bora)
    class WeightedEnsemble:
        def __init__(self, rf, gb, rf_weight=0.65, gb_weight=0.35):
            self.rf = rf
            self.gb = gb
            self.rf_weight = rf_weight
            self.gb_weight = gb_weight
            self.classes_  = np.array([0, 1])

        def predict_proba(self, X):
            p_rf = self.rf.predict_proba(X)
            p_gb = self.gb.predict_proba(X)
            return self.rf_weight * p_rf + self.gb_weight * p_gb

        def predict(self, X):
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

        def get_feature_importances(self):
            try:
                return self.rf.feature_importances_
            except Exception:
                return None

    return WeightedEnsemble(rf, gb)


# =========================================================
# THRESHOLD OPTIMIZATION ILIYOBORESHWA
# =========================================================
def optimize_threshold_precision_focused(model, scaler, val_df: pd.DataFrame,
                                          feature_cols: list,
                                          min_precision: float = MIN_PRECISION_FOR_THR,
                                          min_trades: int = MIN_TRADES_FOR_THR) -> tuple:
    """
    Chagua threshold inayoleta:
    1. precision >= min_precision (lazima)
    2. Maximize: precision * sqrt(trades) * avg_return
    """
    X_val  = val_df[feature_cols].fillna(0).clip(-1e6, 1e6).values
    X_vals = scaler.transform(X_val)
    probs  = model.predict_proba(X_vals)[:, 1]

    val_df = val_df.copy()
    val_df['pred_prob'] = probs

    if 'trailing_return' not in val_df.columns:
        print("  ⚠️ No trailing_return in validation — using default threshold 0.65")
        return 0.65, {}

    best_thr   = DEFAULT_MIN_PROB
    best_score = -np.inf
    best_stats = {}

    # Test thresholds
    thresholds = np.linspace(0.50, 0.95, 46)
    print(f"\n  Threshold optimization (min_precision={min_precision}, min_trades={min_trades}):")

    results_table = []
    for thr in thresholds:
        sel = val_df[val_df['pred_prob'] >= thr]
        if len(sel) < min_trades:
            continue

        y_true = sel['target'].values
        y_pred = np.ones(len(sel), dtype=int)
        prec   = precision_score(y_true, y_pred, zero_division=0)
        rec    = recall_score(y_true, y_pred, zero_division=0)

        if prec < min_precision:
            continue

        total_ret = np.nansum(sel['trailing_return'].values)
        avg_ret   = total_ret / len(sel)
        score     = prec * np.sqrt(len(sel)) * max(0, avg_ret)

        results_table.append({
            'threshold': thr, 'precision': prec, 'recall': rec,
            'trades': len(sel), 'avg_ret': avg_ret, 'score': score
        })

        if score > best_score:
            best_score = score
            best_thr   = float(thr)
            best_stats = {
                'threshold': best_thr, 'precision': float(prec),
                'recall': float(rec), 'trades': int(len(sel)),
                'avg_return': float(avg_ret), 'score': float(score)
            }

    # Print top 5 results
    if results_table:
        results_df = pd.DataFrame(results_table).sort_values('score', ascending=False)
        print("  Top 5 threshold candidates:")
        for _, row in results_df.head(5).iterrows():
            print(f"    thr={row['threshold']:.2f} | prec={row['precision']:.3f} | "
                  f"trades={row['trades']:3d} | avg_ret={row['avg_ret']:.4f} | score={row['score']:.4f}")

    if best_score == -np.inf:
        print(f"  ⚠️ No threshold met precision>={min_precision} with {min_trades}+ trades")
        print(f"  → Falling back to {DEFAULT_MIN_PROB}")
        best_thr   = DEFAULT_MIN_PROB
        best_stats = {'threshold': best_thr, 'note': 'fallback_no_min_precision'}

    print(f"  ✅ Chosen threshold: {best_thr:.3f} | Stats: {best_stats}")
    return best_thr, best_stats


# =========================================================
# WALK-FORWARD VALIDATION
# =========================================================
def walk_forward_validation(df: pd.DataFrame, feature_cols: list,
                              n_splits: int = 4,
                              fee_rate: float = 0.001,
                              target_profit: float = 0.006,
                              stop_loss: float = 0.01,
                              slippage: float = 0.0005) -> dict:
    """
    Walk-forward validation ya kweli:
    - Gawanya data kwa n_splits
    - Kwa kila split: train kwa data ya awali, test kwa data inayofuata
    - Ripoti precision, recall, AUC kwa kila fold
    """
    print(f"\n{'='*55}")
    print(f"WALK-FORWARD VALIDATION ({n_splits} folds)")
    print(f"{'='*55}")

    all_coins  = df['symbol'].unique()
    fold_results = []

    # Gawanya kwa wakati (global timeline)
    all_indices = df.index.sort_values().unique()
    fold_size   = len(all_indices) // (n_splits + 1)

    for fold in range(n_splits):
        train_end   = all_indices[fold_size * (fold + 1)]
        test_start  = train_end
        test_end_idx= min(fold_size * (fold + 2), len(all_indices) - 1)
        test_end    = all_indices[test_end_idx]

        train_f = df[df.index <= train_end].copy()
        test_f  = df[(df.index > test_start) & (df.index <= test_end)].copy()

        if len(train_f) < 200 or len(test_f) < 50:
            continue

        train_f = train_f.dropna(subset=feature_cols + ['target'])
        test_f  = test_f.dropna(subset=feature_cols + ['target'])

        if train_f['target'].sum() < 10:
            continue

        X_tr = train_f[feature_cols].fillna(0).clip(-1e6, 1e6).values
        y_tr = train_f['target'].values
        X_te = test_f[feature_cols].fillna(0).clip(-1e6, 1e6).values
        y_te = test_f['target'].values

        sc   = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        m = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        m.fit(X_tr_s, y_tr)

        probs  = m.predict_proba(X_te_s)[:, 1]
        preds  = (probs >= DEFAULT_MIN_PROB).astype(int)
        prec   = precision_score(y_te, preds, zero_division=0)
        rec    = recall_score(y_te, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = 0.5

        n_signals = int(preds.sum())
        print(f"  Fold {fold+1}: train={len(train_f):,} test={len(test_f):,} | "
              f"precision={prec:.3f} recall={rec:.3f} AUC={auc:.3f} signals={n_signals}")
        fold_results.append({'fold': fold+1, 'precision': prec, 'recall': rec, 'auc': auc, 'signals': n_signals})

    if fold_results:
        avg_prec = np.mean([r['precision'] for r in fold_results])
        avg_auc  = np.mean([r['auc'] for r in fold_results])
        print(f"\n  Average: precision={avg_prec:.3f} AUC={avg_auc:.3f}")
        print(f"  {'✅ Good generalization!' if avg_prec >= 0.35 else '⚠️ Low precision - consider more features'}")

    return {'folds': fold_results}


# =========================================================
# MAIN TRAINER CLASS
# =========================================================
class UniversalModelTrainerV2:
    def __init__(self,
                 fee_rate: float          = 0.001,
                 target_profit_pct: float = 0.6,
                 lookforward_candles: int = LOOKFORWARD_CANDLES,
                 stop_loss_pct: float     = 1.0,
                 slippage_pct: float      = 0.05,
                 calibrate: bool          = False,
                 optimize_trailing: bool  = True,
                 strong_signal_mult: float= STRONG_SIGNAL_MULT,
                 min_net_return: float    = MIN_NET_RETURN):

        self.fee_rate          = fee_rate
        self.target_profit     = target_profit_pct / 100.0
        self.lookforward       = lookforward_candles
        self.stop_loss         = stop_loss_pct / 100.0
        self.slippage          = slippage_pct / 100.0
        self.calibrate         = calibrate
        self.optimize_trailing = optimize_trailing
        self.strong_signal_mult= strong_signal_mult
        self.min_net_return    = min_net_return

        self.scaler             = StandardScaler()
        self.model              = None
        self.coin_trailing_params = {}
        self.feature_cols       = []
        self.threshold          = DEFAULT_MIN_PROB
        self.base_rate          = None

        print(f"✅ Trainer V2 init:")
        print(f"   fee={fee_rate*100:.2f}% | target={target_profit_pct}% | lookforward={lookforward_candles}")
        print(f"   stop_loss={stop_loss_pct}% | slippage={slippage_pct}%")
        print(f"   strong_signal_mult={strong_signal_mult} | min_net_return={min_net_return*100:.2f}%")

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features per coin na label kwa usahihi."""
        print("\n📊 Step 1: Feature Engineering per coin...")
        required = {'open', 'high', 'low', 'close', 'volume'}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        parts = []
        for sym in df['symbol'].unique():
            cd = df[df['symbol'] == sym].sort_index().copy()
            if len(cd) < 50:
                print(f"  ⚠️ {sym}: too few rows ({len(cd)}), skipping")
                continue
            try:
                cd = engineer_features(cd)
                parts.append(cd)
            except Exception as e:
                print(f"  ⚠️ {sym}: feature engineering failed: {e}")
        if not parts:
            raise RuntimeError("No coins with enough data after feature engineering")
        return pd.concat(parts).sort_index()

    def optimize_trailing_params(self, df: pd.DataFrame, train_only: bool = True) -> dict:
        """Optimize trailing params per coin (train data tu)."""
        print("\n⚙️  Step 2: Optimizing trailing params per coin (train-only, no leakage)...")
        params = {}
        for sym in df['symbol'].unique():
            cd = df[df['symbol'] == sym].sort_index()
            n  = len(cd)
            if n < 50:
                params[sym] = (1.0, 1.2)
                continue
            if not self.optimize_trailing:
                params[sym] = (1.0, 1.2)
                continue
            if train_only:
                split = int(n * TRAIN_RATIO)
                cd    = cd.iloc[:split]
            if len(cd) < 30:
                params[sym] = (1.0, 1.2)
                continue
            p, score = optimize_trailing_per_coin(
                cd, self.fee_rate, self.target_profit,
                self.stop_loss, self.slippage, self.lookforward
            )
            params[sym] = p
            print(f"  {sym}: trigger={p[0]} trail={p[1]} (avg_net={score:.5f})")
        self.coin_trailing_params = params
        return params

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weka labels per coin kwa trailing simulation."""
        print("\n🏷️  Step 3: Creating precision-focused labels...")
        parts = []
        total_buy = 0; total_rows = 0

        for sym in df['symbol'].unique():
            cd = df[df['symbol'] == sym].sort_index().copy()
            if len(cd) < 30:
                continue
            trig, trail = self.coin_trailing_params.get(sym, (1.0, 1.2))
            cd_labeled  = label_coin_precision_focused(
                cd, trig, trail,
                self.fee_rate, self.target_profit, self.stop_loss,
                self.slippage, self.lookforward,
                self.strong_signal_mult, self.min_net_return
            )
            buys = int(cd_labeled['target'].sum())
            n    = len(cd_labeled)
            total_buy  += buys
            total_rows += n
            print(f"  {sym}: {buys}/{n} BUY ({buys/n*100:.2f}%)")
            parts.append(cd_labeled)

        df_out = pd.concat(parts).sort_index()
        pct    = total_buy / max(total_rows, 1) * 100
        print(f"\n  TOTAL: {total_buy}/{total_rows} BUY ({pct:.2f}%)")
        if pct > 15:
            print("  ⚠️ BUY rate ni juu sana (>15%) - fikiria kuongeza strong_signal_mult")
        elif pct < 1:
            print("  ⚠️ BUY rate ni chini sana (<1%) - fikiria kupunguza strong_signal_mult")
        else:
            print("  ✅ BUY rate iko sawa")
        return df_out

    def select_features(self, df: pd.DataFrame) -> list:
        """Feature selection iliyoboreshwa."""
        print("\n🔍 Step 4: Smart feature selection...")
        # Tumia train data tu kwa feature selection
        train_df, _ = make_train_test_split(
            df, strategy=TRAIN_TEST_STRATEGY,
            test_coins=TEST_COINS, train_ratio=TRAIN_RATIO
        )
        cols = select_features_smart(
            train_df.dropna(subset=['target']),
            target_col   = 'target',
            max_features = MAX_FEATURES,
            min_mi       = MIN_MUTUAL_INFO,
            corr_thresh  = CORR_THRESHOLD
        )
        self.feature_cols = cols
        return cols

    def train(self, df: pd.DataFrame) -> tuple:
        """Main training pipeline."""
        print("\n🚀 Step 5: Training model...")
        train_df, test_df = make_train_test_split(
            df, strategy=TRAIN_TEST_STRATEGY,
            test_coins=TEST_COINS, train_ratio=TRAIN_RATIO
        )

        req_cols = self.feature_cols + ['target']
        train_df = train_df.dropna(subset=req_cols)
        test_df  = test_df.dropna(subset=req_cols)

        if len(train_df) < 100:
            raise RuntimeError(f"Not enough training data: {len(train_df)} rows")

        X_train = train_df[self.feature_cols].fillna(0).clip(-1e6, 1e6).values
        y_train = train_df['target'].values
        X_test  = test_df[self.feature_cols].fillna(0).clip(-1e6, 1e6).values
        y_test  = test_df['target'].values

        self.scaler = StandardScaler()
        X_train_s   = self.scaler.fit_transform(X_train)
        X_test_s    = self.scaler.transform(X_test)

        self.model = train_precision_ensemble(X_train_s, y_train, self.calibrate)

        # Test performance
        print("\n📈 TEST PERFORMANCE:")
        probs  = self.model.predict_proba(X_test_s)[:, 1]

        # Ripoti kwa thresholds tofauti
        print(f"\n  {'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | "
              f"{'Signals':>8} | {'AUC':>6}")
        print("  " + "-" * 55)

        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = 0.5

        for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            preds  = (probs >= thr).astype(int)
            prec   = precision_score(y_test, preds, zero_division=0)
            rec    = recall_score(y_test, preds, zero_division=0)
            nsig   = int(preds.sum())
            print(f"  {thr:>10.2f} | {prec:>10.3f} | {rec:>8.3f} | "
                  f"{nsig:>8d} | {auc:>6.3f}")

        # Confusion matrix kwa threshold ya default
        preds_default = (probs >= DEFAULT_MIN_PROB).astype(int)
        cm = confusion_matrix(y_test, preds_default)
        print(f"\n  Confusion Matrix (threshold={DEFAULT_MIN_PROB}):")
        print(f"  {cm}")
        print(f"\n  ROC-AUC: {auc:.3f}")
        print(f"\n  Precision@TopK:")
        for k_pct in [0.05, 0.10, 0.15, 0.20]:
            k    = max(1, int(len(probs) * k_pct))
            topk = np.argsort(probs)[-k:]
            prec_k = y_test[topk].mean()
            print(f"    Top {k_pct*100:.0f}% ({k} signals): precision={prec_k:.3f}")

        # Per-coin accuracy kwenye test
        if 'symbol' in test_df.columns:
            print("\n  Per-coin test precision:")
            for sym in test_df['symbol'].unique():
                cm_coin = test_df[test_df['symbol'] == sym]
                if len(cm_coin) < 5: continue
                Xc = cm_coin[self.feature_cols].fillna(0).clip(-1e6, 1e6).values
                Xc_s = self.scaler.transform(Xc)
                pr = self.model.predict_proba(Xc_s)[:, 1]
                pd_c = (pr >= DEFAULT_MIN_PROB).astype(int)
                prec_c = precision_score(cm_coin['target'].values, pd_c, zero_division=0)
                print(f"    {sym}: precision={prec_c:.3f} ({len(cm_coin)} samples)")

        self.base_rate = float(train_df['target'].mean())
        return self.model, self.scaler

    def optimize_threshold(self, df: pd.DataFrame) -> float:
        """Optimize threshold kwa validation set."""
        print("\n🎯 Step 6: Threshold optimization...")
        _, test_df = make_train_test_split(
            df, strategy=TRAIN_TEST_STRATEGY,
            test_coins=TEST_COINS, train_ratio=TRAIN_RATIO
        )
        test_df = test_df.dropna(subset=self.feature_cols + ['target', 'trailing_return'])
        n = len(test_df)
        if n < 50:
            print("  ⚠️ Not enough validation data, using default threshold")
            return DEFAULT_MIN_PROB

        # Gawanya test: 50% validation, 50% holdout
        val_df  = test_df.iloc[:n // 2].copy()
        hold_df = test_df.iloc[n // 2:].copy()

        thr, stats = optimize_threshold_precision_focused(
            self.model, self.scaler, val_df,
            self.feature_cols,
            min_precision = MIN_PRECISION_FOR_THR,
            min_trades    = MIN_TRADES_FOR_THR
        )

        # Verify kwenye holdout
        if len(hold_df) >= 20:
            X_h  = hold_df[self.feature_cols].fillna(0).clip(-1e6, 1e6).values
            X_hs = self.scaler.transform(X_h)
            pr_h = self.model.predict_proba(X_hs)[:, 1]
            pd_h = (pr_h >= thr).astype(int)
            prec_h = precision_score(hold_df['target'].values, pd_h, zero_division=0)
            nsig_h = int(pd_h.sum())
            print(f"  Holdout verification: precision={prec_h:.3f}, signals={nsig_h}")

        self.threshold = thr
        return thr

    def save(self, prefix: str = 'universal_scalp') -> bool:
        """Hifadhi model, scaler, na metadata."""
        if self.model is None:
            print("❌ No model to save")
            return False
        try:
            model_file    = f"{prefix}_model.pkl"
            scaler_file   = f"{prefix}_scaler.pkl"
            trailing_file = f"{prefix}_trailing_params.pkl"
            threshold_file= f"{prefix}_threshold.pkl"
            features_file = "universal_features.pkl"

            joblib.dump(self.model,  model_file)
            joblib.dump(self.scaler, scaler_file)

            with open(trailing_file, 'wb') as f:
                pickle.dump(self.coin_trailing_params, f)

            meta = {
                'threshold' : float(self.threshold),
                'min_prob'  : float(self.threshold),
                'prob_edge' : float(DEFAULT_PROB_EDGE),
                'top_pct'   : float(DEFAULT_TOP_PCT),
                'base_rate' : float(self.base_rate) if self.base_rate else None,
            }
            with open(threshold_file, 'wb') as f:
                pickle.dump(meta, f)

            with open(features_file, 'wb') as f:
                pickle.dump(self.feature_cols, f)

            print(f"\n✅ Saved:")
            print(f"   {model_file}")
            print(f"   {scaler_file}")
            print(f"   {trailing_file}")
            print(f"   {threshold_file}")
            print(f"   {features_file}")

            # Copy kwa universal names (compatibility na bot)
            import shutil
            for src, dst in [
                (model_file,    'universal_model.pkl'),
                (scaler_file,   'universal_scalp_scaler.pkl'),
                (trailing_file, 'universal_trailing_params.pkl'),
                (threshold_file,'universal_threshold.pkl'),
            ]:
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"   Copied {src} -> {dst}")

            return True
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False


# =========================================================
# MAIN PIPELINE
# =========================================================
def train_pipeline(data_file: str          = 'universal_features.csv',
                   fee_rate: float         = 0.001,
                   target_profit: float    = 0.6,
                   lookforward: int        = LOOKFORWARD_CANDLES,
                   stop_loss: float        = 1.0,
                   slippage: float         = 0.05,
                   calibrate: bool         = False,
                   optimize_trailing: bool = True,
                   strong_signal_mult:float= STRONG_SIGNAL_MULT,
                   min_net_return: float   = MIN_NET_RETURN,
                   run_walk_forward: bool  = True):

    print("=" * 58)
    print("UNIVERSAL SCALPING MODEL TRAINER V2.0")
    print("=" * 58)

    # Load data
    print(f"\n📂 Loading: {data_file}")
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"   Loaded {len(df):,} rows | Coins: {df['symbol'].nunique()}")
        print(f"   Columns: {list(df.columns)[:10]}{'...' if len(df.columns)>10 else ''}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None, None

    # Init trainer
    trainer = UniversalModelTrainerV2(
        fee_rate          = fee_rate,
        target_profit_pct = target_profit,
        lookforward_candles = lookforward,
        stop_loss_pct     = stop_loss,
        slippage_pct      = slippage,
        calibrate         = calibrate,
        optimize_trailing = optimize_trailing,
        strong_signal_mult= strong_signal_mult,
        min_net_return    = min_net_return,
    )

    # Pipeline
    df = trainer.prepare_data(df)
    trainer.optimize_trailing_params(df, train_only=True)
    df = trainer.create_labels(df)
    feature_cols = trainer.select_features(df)

    if not feature_cols:
        print("❌ No features selected, aborting")
        return None, None

    # Clean data
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df     = df.dropna(subset=feature_cols + ['target'])
    print(f"\n🧹 Cleaned: removed {before-len(df)} rows. Remaining: {len(df):,}")

    if len(df) < 200:
        print("❌ Not enough data after cleaning")
        return None, None

    # Walk-forward validation (kabla ya final training)
    if run_walk_forward:
        walk_forward_validation(
            df, feature_cols,
            n_splits     = 4,
            fee_rate     = fee_rate,
            target_profit= target_profit / 100,
            stop_loss    = stop_loss / 100,
            slippage     = slippage / 100,
        )

    # Train final model
    model, scaler = trainer.train(df)

    if model is not None:
        # Optimize threshold
        trainer.optimize_threshold(df)

        # Save
        trainer.save('universal_scalp')

        print(f"\n{'='*58}")
        print("MWISHO - MATOKEO YA MUHIMU")
        print(f"{'='*58}")
        print(f"  Threshold ya kuchagua: {trainer.threshold:.3f}")
        print(f"  Base rate (BUY%):      {trainer.base_rate*100:.2f}%")
        print(f"  Features zilizotumika: {len(feature_cols)}")
        print(f"\n  ✅ Model imehifadhiwa na iko tayari kwa bot!")
        print(f"\n  JINSI YA KUTUMIA KWENYE BOT:")
        print(f"  Badilisha kwenye bot script:")
        print(f"    AI_ALLOW_BUY_MIN = {trainer.threshold:.2f}")
        print(f"    AI_MODEL_PATH    = 'universal_scalp_model.pkl'")
        print(f"    AI_SCALER_PATH   = 'universal_scalp_scaler.pkl'")

    return model, scaler


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Universal Scalping Trainer V2.0 - Precision-focused'
    )
    parser.add_argument('--data',              type=str,   default='universal_features.csv')
    parser.add_argument('--fee',               type=float, default=0.1,   help='fee %% (e.g. 0.1)')
    parser.add_argument('--profit',            type=float, default=0.6,   help='target profit %% (e.g. 0.6)')
    parser.add_argument('--lookforward',       type=int,   default=LOOKFORWARD_CANDLES)
    parser.add_argument('--stoploss',          type=float, default=1.0)
    parser.add_argument('--slippage',          type=float, default=0.05)
    parser.add_argument('--calibrate',         action='store_true')
    parser.add_argument('--no-trailing',       action='store_true')
    parser.add_argument('--strong-mult',       type=float, default=STRONG_SIGNAL_MULT,
                        help='Strong signal multiplier (default 2.5)')
    parser.add_argument('--min-net-return',    type=float, default=MIN_NET_RETURN * 100,
                        help='Min net return %% for BUY label (default 0.4)')
    parser.add_argument('--no-walk-forward',   action='store_true')
    parser.add_argument('--test-coins',        type=str,   default='',
                        help='Comma-separated test coins e.g. PEPEUSDT,SUIUSDT')
    parser.add_argument('--train-ratio',       type=float, default=TRAIN_RATIO)
    args = parser.parse_args()

    # Override globals kama zimetolewa
    if args.test_coins:
        TEST_COINS[:] = [c.strip().upper() for c in args.test_coins.split(',') if c.strip()]
        TRAIN_TEST_STRATEGY = "coin"
        print(f"Using COIN split. Test coins: {TEST_COINS}")
    TRAIN_RATIO = args.train_ratio

    model, scaler = train_pipeline(
        data_file         = args.data,
        fee_rate          = args.fee / 100.0,
        target_profit     = args.profit,
        lookforward       = args.lookforward,
        stop_loss         = args.stoploss,
        slippage          = args.slippage,
        calibrate         = args.calibrate,
        optimize_trailing = not args.no_trailing,
        strong_signal_mult= args.strong_mult,
        min_net_return    = args.min_net_return / 100.0,
        run_walk_forward  = not args.no_walk_forward,
    )

    if model:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed.")