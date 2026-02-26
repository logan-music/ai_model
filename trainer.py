#!/usr/bin/env python3
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# DEFAULT INFERENCE / RISK HYPERPARAMS (tunable)
# -------------------------
DEFAULT_MIN_PROB = 0.65        # require prediction probability >= this
DEFAULT_PROB_EDGE = 0.15       # require prob - base_rate >= this (base_rate ≈ prevalence of BUY)
DEFAULT_TOP_PCT = 0.20         # only keep top X% of signals by probability (after other filters)
BASE_POSITION_PCT = 0.005      # base position sizing (0.5% capital)
MAX_POSITION_PCT = 0.02        # absolute cap per trade (2% capital)
ATR_MEDIAN_WINDOW = 20         # window to compute median ATR for volatility filter
VOL_BREAKOUT_MULT = 1.0        # multiplier for ATR median check (1.0 means >= median)

class UniversalModelTrainer:
    def __init__(self, fee_rate=0.001, target_profit_pct=0.6, lookforward_candles=8,
                 stop_loss_pct=1.0, slippage_pct=0.05, calibrate=False,
                 tp_atr_mul=3.0, sl_atr_mul=2.0,
                 optimize_trailing=True, use_strong_signal=True, strong_signal_mult=1.0):
        self.fee_rate = fee_rate
        self.target_profit = target_profit_pct / 100.0
        self.lookforward = lookforward_candles
        self.stop_loss = stop_loss_pct / 100.0
        self.slippage = slippage_pct / 100.0
        self.calibrate = calibrate

        self.tp_atr_mul = tp_atr_mul
        self.sl_atr_mul = sl_atr_mul
        self.optimize_trailing = optimize_trailing

        self.use_strong_signal = use_strong_signal
        self.strong_signal_mult = float(strong_signal_mult)

        self.scaler = StandardScaler()
        self.model = None
        self.coin_trailing_params = {}

        print(f"✅ Trainer init: fee={fee_rate*100:.2f}% target={target_profit_pct}% "
              f"lookforward={lookforward_candles} stop_loss={stop_loss_pct}% slippage={slippage_pct}% "
              f"tp_atr_mul={tp_atr_mul} sl_atr_mul={sl_atr_mul} optimize_trailing={optimize_trailing} "
              f"use_strong_signal={use_strong_signal} strong_mult={self.strong_signal_mult}")

    # -------------------------
    # Helper: robust align iterable -> Series (not used by main path but kept)
    # -------------------------
    def _align_iterable_to_index(self, values, index, name='value', dtype=None):
        try:
            arr = np.asarray(values)
        except Exception:
            arr = np.array(list(values), dtype=object)

        desired = len(index)
        actual = arr.shape[0] if arr.ndim == 1 else arr.shape[0]

        if actual == desired:
            s = pd.Series(arr, index=index)
        elif actual < desired:
            pad_len = desired - actual
            pad = np.full(pad_len, np.nan, dtype=arr.dtype if np.issubdtype(arr.dtype, np.number) else object)
            new = np.concatenate([arr, pad])
            print(f"⚠️ Align: padding {name} from {actual} -> {desired} (added {pad_len} NaNs)")
            s = pd.Series(new, index=index)
        else:
            new = arr[:desired]
            print(f"⚠️ Align: truncating {name} from {actual} -> {desired} (extra values dropped)")
            s = pd.Series(new, index=index)

        if dtype is not None:
            try:
                s = s.astype(dtype)
            except Exception:
                pass
        return s

    # -------------------------
    # Dynamic TP/SL helper
    # -------------------------
    def _dynamic_tp_sl_for_row(self, entry_price, atr_pct):
        atr_pct = float(atr_pct) if not np.isnan(atr_pct) else 0.0
        tp_pct = max(self.target_profit, atr_pct * self.tp_atr_mul)
        sl_pct = max(self.stop_loss, atr_pct * self.sl_atr_mul)
        tp_price = entry_price * (1 + tp_pct + self.fee_rate + self.slippage)
        sl_price = entry_price * (1 - sl_pct - self.slippage)
        return tp_price, sl_price, tp_pct, sl_pct

    # -------------------------
    # Trailing-stop simulator for an entry
    # -------------------------
    def simulate_trailing_for_entry(self, highs, lows, entry_idx, entry_price,
                                    atr_pct, lookforward, trail_trigger_atr_mult,
                                    trail_atr_mult):
        n = len(highs)
        atr_price = atr_pct * entry_price
        trigger_price = entry_price * (1 + trail_trigger_atr_mult * atr_pct) if atr_pct > 0 else entry_price * (1 + trail_trigger_atr_mult * 1e-6)

        trailing_active = False
        trailing_stop = None
        best_high = entry_price

        for j in range(entry_idx + 1, min(n, entry_idx + 1 + lookforward)):
            h = highs[j]; l = lows[j]
            if h > best_high:
                best_high = h
            if not trailing_active:
                if h >= trigger_price:
                    trailing_active = True
                    trailing_stop = max(entry_price * (1 - self.slippage - self.fee_rate), best_high - trail_atr_mult * atr_price)
            else:
                candidate = best_high - trail_atr_mult * atr_price
                if candidate > trailing_stop:
                    trailing_stop = candidate
                if l <= trailing_stop:
                    exit_revenue = trailing_stop * (1 - self.fee_rate - self.slippage)
                    net_return = (exit_revenue - entry_price * (1 + self.fee_rate + self.slippage)) / (entry_price * (1 + self.fee_rate + self.slippage) + 1e-12)
                    return j, trailing_stop, net_return, 'trail'

        exit_revenue = best_high * (1 - self.fee_rate - self.slippage)
        net_return = (exit_revenue - entry_price * (1 + self.fee_rate + self.slippage)) / (entry_price * (1 + self.fee_rate + self.slippage) + 1e-12)
        return None, best_high, net_return, 'none'

    # -------------------------
    # Per-coin trailing optimizer (expanded grid)
    # -------------------------
    def optimize_trailing_per_coin(self, coin_df, lookback=None):
        trigger_grid = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
        trail_grid = [0.5, 0.8, 1.0, 1.2, 1.6, 2.0]

        highs = coin_df['high'].values
        lows = coin_df['low'].values
        closes = coin_df['close'].values
        atrs = coin_df.get('atr_pct', pd.Series(0, index=coin_df.index)).values
        n = len(closes)
        lookforward = self.lookforward

        best_score = -np.inf
        best_params = (trigger_grid[0], trail_grid[0])

        idxs = range(0, n)
        if n > 2000:
            step = max(1, n // 1500)
            idxs = range(0, n, step)

        for trig in trigger_grid:
            for trail in trail_grid:
                cum = 0.0; count = 0
                for i in idxs:
                    entry_price = closes[i]
                    atr_pct = atrs[i] if i < len(atrs) else 0.0
                    if np.isnan(atr_pct):
                        atr_pct = 0.0
                    _, _, net_return, _ = self.simulate_trailing_for_entry(
                        highs, lows, i, entry_price, atr_pct, lookforward, trig, trail
                    )
                    if np.isfinite(net_return):
                        cum += net_return
                        count += 1
                score = (cum / count) if count else -np.inf
                if score > best_score:
                    best_score = score
                    best_params = (trig, trail)

        return best_params, best_score

    # -------------------------
    # Labeling helper for a single coin (bulletproof)
    # -------------------------
    def _label_single_coin(self, coin_df, trig_mult, trail_mult):
        """
        Build a coin-level DataFrame with columns:
         - future_return, trailing_return, target, signal_volume_ok
        Returns coin_df_copy with those columns set (index preserved).
        Robust to small datasets / missing ATR / duplicate indices handled upstream.
        """
        coin_df = coin_df.copy().sort_index()
        n = len(coin_df)
        # initialize arrays
        future_returns = [np.nan] * n
        trailing_returns = [np.nan] * n
        labels = [0] * n

        highs = coin_df['high'].values
        lows = coin_df['low'].values
        closes = coin_df['close'].values
        atrs = coin_df.get('atr_pct', pd.Series(0, index=coin_df.index)).values

        vol_series = coin_df.get('volume_ratio', pd.Series(0, index=coin_df.index))
        if not isinstance(vol_series, pd.Series):
            vol_series = pd.Series(vol_series, index=coin_df.index)
        vol_ok = (vol_series.reindex(coin_df.index).fillna(0) > 0.8).astype(int)

        for i in range(n):
            entry_price = closes[i]
            atr_pct = float(atrs[i]) if i < len(atrs) and not np.isnan(atrs[i]) else 0.0

            tp_price, sl_price, tp_pct, sl_pct = self._dynamic_tp_sl_for_row(entry_price, atr_pct)

            # static TP/SL conservative check
            hit_tp = False; hit_sl = False
            net_return_static = np.nan
            for j in range(i+1, min(n, i+1+self.lookforward)):
                if highs[j] >= tp_price and not hit_sl:
                    exit_revenue = tp_price * (1 - self.fee_rate - self.slippage)
                    net_return_static = (exit_revenue - entry_price * (1 + self.fee_rate + self.slippage)) / (entry_price * (1 + self.fee_rate + self.slippage) + 1e-12)
                    hit_tp = True
                    break
                if lows[j] <= sl_price:
                    exit_revenue = sl_price * (1 - self.fee_rate - self.slippage)
                    net_return_static = (exit_revenue - entry_price * (1 + self.fee_rate + self.slippage)) / (entry_price * (1 + self.fee_rate + self.slippage) + 1e-12)
                    hit_sl = True
                    break
            if not hit_tp and not hit_sl:
                start = i+1
                end = min(n, i+1+self.lookforward)
                if start < end:
                    future_window_high = np.max(highs[start:end])
                else:
                    future_window_high = highs[i]
                exit_revenue = future_window_high * (1 - self.fee_rate - self.slippage)
                net_return_static = (exit_revenue - entry_price * (1 + self.fee_rate + self.slippage)) / (entry_price * (1 + self.fee_rate + self.slippage) + 1e-12)

            future_returns[i] = net_return_static

            # trailing sim
            _, _, net_return_trail, _ = self.simulate_trailing_for_entry(
                highs, lows, i, entry_price, atr_pct, self.lookforward, trig_mult, trail_mult
            )
            trailing_returns[i] = net_return_trail

            # strong-signal thresholding (labels) - this is OK because labels are derived from simulated future returns
            if np.isfinite(net_return_trail):
                if self.use_strong_signal:
                    threshold = self.target_profit * float(self.strong_signal_mult)
                    labels[i] = 1 if net_return_trail > threshold else 0
                else:
                    labels[i] = 1 if net_return_trail > 0 else 0
            else:
                labels[i] = 0

        # assign to a coin-level DataFrame (keeps index)
        out = coin_df.copy()
        try:
            out['future_return'] = pd.Series(future_returns, index=out.index).astype(float)
            out['trailing_return'] = pd.Series(trailing_returns, index=out.index).astype(float)
            out['target'] = pd.Series(labels, index=out.index).astype(int)
            out['signal_volume_ok'] = vol_ok.reindex(out.index).fillna(0).astype(int)
        except Exception as e:
            # last-resort alignment: use helper to pad/truncate
            print(f"⚠️ Alignment fallback for coin {out.index.name or 'coin'}: {e}")
            out['future_return'] = self._align_iterable_to_index(future_returns, out.index, name='future_returns', dtype=float)
            out['trailing_return'] = self._align_iterable_to_index(trailing_returns, out.index, name='trailing_returns', dtype=float)
            out['target'] = self._align_iterable_to_index(labels, out.index, name='labels', dtype=int)
            out['signal_volume_ok'] = vol_ok.reindex(out.index).fillna(0).astype(int)

        return out

    # -------------------------
    # Main label creation (bulletproof)
    # -------------------------
    def create_universal_labels(self, df, use_trailing=True, coin_trailing_params=None):
        """
        Create labels per coin, using train-derived coin_trailing_params if provided to avoid leakage.
        Uses _label_single_coin to produce safe per-coin DataFrames and then df.update() to assign.
        """
        print("🔖 Creating labels per coin (TP/SL + trailing simulation, robust mode)")
        df = df.copy()

        # ensure columns exist so df.update will work
        df['future_return'] = np.nan
        df['trailing_return'] = np.nan
        df['target'] = 0
        df['signal_volume_ok'] = 0

        idx_name = df.index.name if df.index.name else 'timestamp'
        df = df.sort_values(['symbol', idx_name])

        symbols = df['symbol'].unique()

        # decide trailing params per coin:
        if coin_trailing_params:
            # use provided (from training splits)
            self.coin_trailing_params = dict(coin_trailing_params)
        else:
            # If no params provided, optionally run search (warn)
            if use_trailing and self.optimize_trailing:
                print("⚠️ Warning: optimizing trailing params on full dataset (not recommended).")
                for sym in symbols:
                    coin_df_tmp = df[df['symbol'] == sym].sort_index()
                    if len(coin_df_tmp) < 50:
                        self.coin_trailing_params[sym] = (1.0, 1.2)
                        continue
                    params, score = self.optimize_trailing_per_coin(coin_df_tmp)
                    self.coin_trailing_params[sym] = params
                    print(f"  {sym}: chosen trailing params trigger={params[0]} trail={params[1]} (avg net={score:.6f})")
            else:
                for sym in symbols:
                    self.coin_trailing_params[sym] = (1.0, 1.2)

        # iterate coins and build labelled coin_df safely
        total_buys = 0
        total_rows = 0
        for sym in symbols:
            coin_df = df[df['symbol'] == sym].copy().sort_index()
            if coin_df.empty:
                continue

            # detect duplicate indices and warn + drop duplicates (keep first)
            if coin_df.index.duplicated().any():
                dup_count = coin_df.index.duplicated().sum()
                print(f"⚠️ {sym}: found {dup_count} duplicate index rows — dropping duplicates (keep='first') to avoid alignment errors.")
                coin_df = coin_df[~coin_df.index.duplicated(keep='first')].copy()

            n = len(coin_df)
            if n == 0:
                continue

            trig_mult, trail_mult = self.coin_trailing_params.get(sym, (1.0, 1.2))
            labeled_coin_df = self._label_single_coin(coin_df, trig_mult, trail_mult)

            # Ensure labeled_coin_df has required columns
            for col in ['future_return', 'trailing_return', 'target', 'signal_volume_ok']:
                if col not in labeled_coin_df.columns:
                    # add safe defaults
                    if col == 'target' or col == 'signal_volume_ok':
                        labeled_coin_df[col] = 0
                    else:
                        labeled_coin_df[col] = np.nan

            # Update the global df using pandas .update (aligns by index)
            try:
                df.update(labeled_coin_df[['future_return', 'trailing_return', 'target', 'signal_volume_ok']])
            except Exception as e:
                print(f"⚠️ Failed to update main df for {sym}: {e}")
                # fallback: assign by loc with alignment (safe)
                idx = labeled_coin_df.index
                df.loc[idx, 'future_return'] = labeled_coin_df['future_return']
                df.loc[idx, 'trailing_return'] = labeled_coin_df['trailing_return']
                df.loc[idx, 'target'] = labeled_coin_df['target']
                df.loc[idx, 'signal_volume_ok'] = labeled_coin_df['signal_volume_ok']

            buys = int(np.nansum(labeled_coin_df['target'].fillna(0).astype(float).values))
            total_buys += buys
            total_rows += n
            print(f"  {sym}: labeled {buys} / {n} as BUY ({(buys / n * 100):.2f}%) using trailing params trigger={trig_mult}, trail={trail_mult}")

        # overall
        total = int(df['target'].dropna().shape[0])
        buys = int(df['target'].sum()) if total else 0
        pct = (buys / total * 100) if total else 0.0
        print(f"\nTOTAL: {buys} BUY signals out of {total} samples ({pct:.2f}%)")
        return df

    # -------------------------
    # Feature selection unchanged (FIXED: exclude trailing_return to prevent leakage)
    # -------------------------
    def select_universal_features(self, df):
        # IMPORTANT: exclude any columns that encode future info / labels.
        exclude = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
            'future_return', 'trailing_return', 'target', 'symbol', 'signal_volume_ok'
        ]
        feature_cols = [col for col in df.columns if col not in exclude and df[col].dtype in [float, int]]
        feature_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.7]
        print(f"Selected {len(feature_cols)} features")
        return feature_cols

    # -------------------------
    # Cross-coin validation unchanged
    # -------------------------
    def validate_across_coins(self, df, feature_cols):
        print("🔬 Cross-coin validation (time-aware)")
        results = []
        unique_coins = list(df['symbol'].unique())
        if len(unique_coins) < 3:
            print("Not enough coins for cross-coin validation.")
            return None

        for test_coin in unique_coins[:3]:
            test_df = df[df['symbol'] == test_coin].dropna()
            if len(test_df) < 50:
                print(f"Skipping {test_coin} (too small test set)")
                continue
            test_start = test_df.index.min()
            train_df = df[(df['symbol'] != test_coin) & (df.index < test_start)].dropna()
            if len(train_df) < 100:
                train_df = df[df['symbol'] != test_coin].dropna()
            X_train = train_df[feature_cols].values
            y_train = train_df['target'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['target'].values
            if len(np.unique(y_train)) < 2:
                print(f"Insufficient class diversity in training for {test_coin}, skipping.")
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
            m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            print(f"{test_coin}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f}")
            results.append({'test_coin': test_coin, 'acc': acc, 'prec': prec, 'rec': rec})

        return results

    # -------------------------
    # Ensemble training unchanged
    # -------------------------
    def train_universal_ensemble(self, X_train, y_train):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=30,
            min_samples_leaf=15, max_features='sqrt', random_state=42,
            n_jobs=-1, class_weight='balanced'
        )
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42
        )
        ensemble = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)
        if self.calibrate:
            print("Calibrating classifier probabilities (isotonic)...")
            calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
            calibrated.fit(X_train, y_train)
            print("Calibration complete.")
            return calibrated
        return ensemble

    # -------------------------
    # Final model training unchanged (time-safe)
    # -------------------------
    def train_final_universal_model(self, df, feature_cols, test_size=0.2):
        print("⏳ Preparing time-safe train/test split per coin...")
        train_parts = []; test_parts = []
        for sym in df['symbol'].unique():
            coin_df = df[df['symbol'] == sym].sort_index()
            n = len(coin_df)
            if n < 20:
                continue
            split_idx = int(n * (1 - test_size))
            train_parts.append(coin_df.iloc[:split_idx])
            test_parts.append(coin_df.iloc[split_idx:])

        if not train_parts or not test_parts:
            print("Not enough data to split per coin.")
            return None, None

        train_df = pd.concat(train_parts)
        test_df = pd.concat(test_parts)

        print(f"Train samples: {len(train_df):,}, Test samples: {len(test_df):,}")
        print(f"Train coins: {', '.join(train_df['symbol'].unique())}")
        print(f"Test coins: {', '.join(test_df['symbol'].unique())}")

        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = self.train_universal_ensemble(X_train_s, y_train)

        print("\nTEST PERFORMANCE:")
        y_pred = self.model.predict(X_test_s)
        try:
            y_prob = self.model.predict_proba(X_test_s)[:, 1]
        except Exception:
            y_prob = np.zeros_like(y_pred, dtype=float)

        print(classification_report(y_test, y_pred, target_names=['NO_BUY', 'BUY']))
        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC: {auc:.3f}")
        except Exception:
            auc = None

        cm2 = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm2)

        for sym in test_df['symbol'].unique():
            coin_test = test_df[test_df['symbol'] == sym]
            if len(coin_test) < 5:
                continue
            Xc = coin_test[feature_cols].values
            Xc_s = self.scaler.transform(Xc)
            y_c = coin_test['target'].values
            y_c_pred = self.model.predict(Xc_s)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_c, y_c_pred)
            print(f"{sym}: accuracy={acc:.3f} ({len(y_c)} samples)")

        return self.model, self.scaler

    # -------------------------
    # Save model + trailing params + threshold
    # -------------------------
    def save_universal_model(self, model_prefix='universal_scalp', threshold=0.5,
                             min_prob=DEFAULT_MIN_PROB, prob_edge=DEFAULT_PROB_EDGE, top_pct=DEFAULT_TOP_PCT,
                             base_rate=None):
        """
        Save model, scaler, trailing params and threshold metadata.
        threshold: the raw decision threshold (used by legacy code)
        min_prob/prob_edge/top_pct/base_rate: metadata for inference filters
        """
        if self.model is None:
            print("No model to save.")
            return False
        model_file = f"{model_prefix}_model.pkl"
        scaler_file = f"{model_prefix}_scaler.pkl"
        trailing_file = f"{model_prefix}_trailing_params.pkl"
        threshold_file = f"{model_prefix}_threshold.pkl"
        try:
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            with open(trailing_file, 'wb') as f:
                pickle.dump(self.coin_trailing_params, f)
            # Save structured metadata for safe inference later
            meta = {
                'threshold': float(threshold),
                'min_prob': float(min_prob),
                'prob_edge': float(prob_edge),
                'top_pct': float(top_pct),
                'base_rate': float(base_rate) if base_rate is not None else None
            }
            with open(threshold_file, 'wb') as f:
                pickle.dump(meta, f)
            print(f"Saved {model_file}, {scaler_file}, {trailing_file}, {threshold_file}")
            return True
        except Exception as e:
            print(f"Error saving model/scaler/trailing/threshold: {e}")
            return False

# ---------------------------
# Utility: decision threshold optimizer (validation/holdout)
# ---------------------------
def optimize_decision_threshold(model, scaler, df, feature_cols, test_size=0.2, min_trades=20, holdout_frac=0.5):
    """
    Splits per-coin test parts into validation & holdout:
      - validation: earlier part of test (used to pick threshold)
      - holdout: last part (used for reporting, untouched during selection)
    Metric used for selection: score = total_return * sqrt(count)
    Returns: best_threshold, stats_dict { 'validation': {...}, 'holdout': {...}, 'base_rate': ... }
    """
    # build per-coin test parts
    val_parts = []
    hold_parts = []
    for sym in df['symbol'].unique():
        coin_df = df[df['symbol'] == sym].sort_index()
        n = len(coin_df)
        if n < 5:
            continue
        split_idx = int(n * (1 - test_size))
        test_part = coin_df.iloc[split_idx:]
        if len(test_part) == 0:
            continue
        hold_len = max(1, int(len(test_part) * holdout_frac))
        if hold_len >= len(test_part):
            hold_len = max(1, len(test_part) - 1)
        val_part = test_part.iloc[:-hold_len].copy()
        hold_part = test_part.iloc[-hold_len:].copy()
        if len(val_part) > 0:
            val_parts.append(val_part)
        if len(hold_part) > 0:
            hold_parts.append(hold_part)

    if not val_parts:
        print("No validation parts available for threshold optimization.")
        return 0.5, {}

    val_df = pd.concat(val_parts)
    hold_df = pd.concat(hold_parts) if hold_parts else pd.DataFrame(columns=val_df.columns)

    # Build feature matrices and predicted probabilities
    X_val = val_df[feature_cols].values
    try:
        X_val_s = scaler.transform(X_val)
    except Exception:
        X_val_s = scaler.fit_transform(X_val)

    try:
        probs_val = model.predict_proba(X_val_s)[:, 1]
    except Exception:
        probs_val = model.predict(X_val_s)
        probs_val = np.array([float(p) for p in probs_val])

    val_df = val_df.copy()
    val_df['pred_prob'] = probs_val

    # Prepare holdout predictions (for final reporting)
    if not hold_df.empty:
        X_hold = hold_df[feature_cols].values
        try:
            X_hold_s = scaler.transform(X_hold)
        except Exception:
            X_hold_s = scaler.fit_transform(X_hold)
        try:
            probs_hold = model.predict_proba(X_hold_s)[:, 1]
        except Exception:
            probs_hold = model.predict(X_hold_s)
            probs_hold = np.array([float(p) for p in probs_hold])
        hold_df = hold_df.copy()
        hold_df['pred_prob'] = probs_hold

    # trailing_return must exist in validation
    if 'trailing_return' not in val_df.columns:
        print("No trailing_return in validation set for threshold optimization.")
        return 0.5, {}

    best_thr = 0.5
    best_score = -np.inf
    best_stats = {}

    thrs = np.linspace(0.5, 0.99, 50)
    for thr in thrs:
        sel = val_df[val_df['pred_prob'] >= thr]
        count = len(sel)
        if count < min_trades:
            continue
        total_return = np.nansum(sel['trailing_return'].values)
        avg_return = total_return / count if count else 0.0
        score = total_return * np.sqrt(count)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_stats = {'threshold': best_thr, 'total_return': float(total_return), 'avg_return': float(avg_return), 'count': int(count), 'score': float(score)}

    if best_score == -np.inf:
        fallback_best = -np.inf
        for thr in thrs:
            sel = val_df[val_df['pred_prob'] >= thr]
            count = len(sel)
            if count == 0:
                continue
            total_return = np.nansum(sel['trailing_return'].values)
            avg_return = total_return / count
            score2 = avg_return * count
            if score2 > fallback_best:
                fallback_best = score2
                best_thr = float(thr)
                best_stats = {'threshold': best_thr, 'total_return': float(total_return), 'avg_return': float(avg_return), 'count': int(count), 'score': float(score2)}
        if fallback_best == -np.inf:
            best_thr = 0.8
            best_stats = {}

    # compute holdout stats
    hold_stats = {}
    if not hold_df.empty:
        sel_hold = hold_df[hold_df['pred_prob'] >= best_thr]
        count_h = len(sel_hold)
        total_return_h = np.nansum(sel_hold['trailing_return'].values) if count_h else 0.0
        avg_return_h = (total_return_h / count_h) if count_h else 0.0
        hold_stats = {'threshold': float(best_thr), 'total_return': float(total_return_h), 'avg_return': float(avg_return_h), 'count': int(count_h)}
    else:
        hold_stats = {'note': 'no_holdout_data'}

    base_rate = float(val_df['target'].mean()) if 'target' in val_df.columns else None
    print("Threshold optimization (validation -> holdout) result:")
    print("  validation:", best_stats)
    print("  holdout:", hold_stats)
    print(f"  base_rate (validation BUY prevalence): {base_rate}")

    return float(best_thr), {'validation': best_stats, 'holdout': hold_stats, 'base_rate': base_rate}

# ---------------------------
# NEW: Inference / filtering helper
# ---------------------------
def generate_inference_signals(model, scaler, df, feature_cols,
                               min_prob=DEFAULT_MIN_PROB,
                               prob_edge=DEFAULT_PROB_EDGE,
                               top_pct=DEFAULT_TOP_PCT,
                               base_rate=None,
                               atr_window=ATR_MEDIAN_WINDOW,
                               atr_mult=VOL_BREAKOUT_MULT,
                               base_position_pct=BASE_POSITION_PCT,
                               max_position_pct=MAX_POSITION_PCT):
    """
    Given trained model+scaler and a dataframe (index aligned with features),
    return a DataFrame with:
      - pred_prob: model probability
      - signal_raw: bool (prob >= min_prob)
      - signal_edge: bool (prob - base_rate >= prob_edge)
      - vol_ok: bool (atr >= median_atr*atr_mult OR volatility breakout)
      - signal: final boolean after applying filters
      - position_pct: position size (fraction of capital) computed from probability
    Notes:
      - base_rate: if None, infer from df['target'] when available, else default to 0.13 approx.
      - This function is intended for both backtesting and live execution.
    """
    X = df[feature_cols].values
    Xs = scaler.transform(X)
    try:
        probs = model.predict_proba(Xs)[:, 1]
    except Exception:
        probs = model.predict(Xs).astype(float)
        # if predict returns 0/1 labels, convert to float (not ideal)
    out = pd.DataFrame(index=df.index)
    out['pred_prob'] = probs
    out['signal_raw'] = out['pred_prob'] >= float(min_prob)

    if base_rate is None:
        if 'target' in df.columns:
            base_rate = float(df['target'].mean())
        else:
            base_rate = 0.13
    out['prob_edge_val'] = out['pred_prob'] - float(base_rate)
    out['signal_edge'] = out['prob_edge_val'] >= float(prob_edge)

    # Volatility filter: ATR above median (per-coin would be better; here global per-index)
    if 'atr_pct' in df.columns:
        atr_med = df['atr_pct'].rolling(window=atr_window, min_periods=1).median()
        out['vol_ok'] = df['atr_pct'] >= (atr_med * float(atr_mult))
    else:
        out['vol_ok'] = True

    # combine
    out['signal'] = out['signal_raw'] & out['signal_edge'] & out['vol_ok']

    # If we want "top X% of signals" (by probability) — apply per-symbol if present, else global
    if 'symbol' in df.columns:
        # keep top pct per symbol among currently-signaled
        out['signal_top'] = False
        syms = df['symbol'].unique()
        for s in syms:
            mask = (df['symbol'] == s) & (out['signal'])
            if mask.sum() == 0:
                continue
            probs_masked = out.loc[mask, 'pred_prob']
            cutoff = np.quantile(probs_masked, 1 - float(top_pct)) if len(probs_masked) > 1 else probs_masked.max()
            out.loc[mask & (out['pred_prob'] >= cutoff), 'signal_top'] = True
    else:
        # global cutoff
        mask = out['signal']
        if mask.sum() > 0:
            probs_masked = out.loc[mask, 'pred_prob']
            cutoff = np.quantile(probs_masked, 1 - float(top_pct)) if len(probs_masked) > 1 else probs_masked.max()
            out['signal_top'] = (out['signal']) & (out['pred_prob'] >= cutoff)
        else:
            out['signal_top'] = False

    # final signal uses top filter
    out['signal_final'] = out['signal'] & out['signal_top']

    # Position sizing: scale from base_position_pct by probability ratio, and cap
    # size = base_position_pct * (prob / min_prob)  -> so prob==min_prob => base size
    # enforce caps
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = out['pred_prob'] / (float(min_prob) + 1e-12)
        ratio = np.clip(ratio, 0, 2)   # cap at 2x size

    out['position_pct'] = base_position_pct * ratio
    out['position_pct'] = out['position_pct'].clip(0.0, float(max_position_pct))

    # Only keep size where signal exists
    out.loc[~out['signal_final'], 'position_pct'] = 0.0

    # Debug info
    out['base_rate'] = float(base_rate)
    out['min_prob'] = float(min_prob)
    out['prob_edge'] = float(prob_edge)
    out['top_pct'] = float(top_pct)
    
    return out

# ---------------------------
# Main pipeline (leakage-free)
# ---------------------------
def train_universal_pipeline(data_file='universal_features.csv',
                             fee_rate=0.001, target_profit=0.6, lookforward=8,
                             stop_loss=1.0, slippage=0.05, calibrate=False,
                             tp_atr_mul=3.0, sl_atr_mul=2.0, optimize_trailing=True,
                             use_strong_signal=True, strong_signal_mult=1.0, test_size=0.2):
    print("Loading dataset:", data_file)
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        print(f"Loaded {len(df):,} rows")
    except Exception as e:
        print("Error loading data:", e)
        return None, None

    trainer = UniversalModelTrainer(
        fee_rate=fee_rate,
        target_profit_pct=target_profit,
        lookforward_candles=lookforward,
        stop_loss_pct=stop_loss,
        slippage_pct=slippage,
        calibrate=calibrate,
        tp_atr_mul=tp_atr_mul,
        sl_atr_mul=sl_atr_mul,
        optimize_trailing=optimize_trailing,
        use_strong_signal=use_strong_signal,
        strong_signal_mult=strong_signal_mult
    )

    # -------------------------
    # Create train/test splits per coin FIRST (time-aware) and compute trailing params on train only
    # -------------------------
    print("⏱ Step 1: computing per-coin training slices and optimizing trailing on train-only data (no leakage).")
    coin_trailing_params = {}
    for sym in df['symbol'].unique():
        coin_df = df[df['symbol'] == sym].sort_index()
        n = len(coin_df)
        if n < 50:
            coin_trailing_params[sym] = (1.0, 1.2)
            continue
        split_idx = int(n * (1 - test_size))
        train_coin_df = coin_df.iloc[:split_idx]
        if len(train_coin_df) < 50 or not optimize_trailing:
            coin_trailing_params[sym] = (1.0, 1.2)
            continue
        params, score = trainer.optimize_trailing_per_coin(train_coin_df)
        coin_trailing_params[sym] = params
        print(f"  {sym}: train-only chosen trailing params trigger={params[0]} trail={params[1]} (avg net={score:.6f})")

    # -------------------------
    # Build labels using training-derived trailing params (no further trailing optimization)
    # -------------------------
    df = trainer.create_universal_labels(df, use_trailing=True, coin_trailing_params=coin_trailing_params)

    # feature selection and cleaning
    feature_cols = trainer.select_universal_features(df)

    if feature_cols:
        print("🧹 Checking for infinities / extreme values in selected features...")
        try:
            inf_counts = np.isinf(df[feature_cols]).sum()
            large_counts = (np.abs(df[feature_cols]) > 1e6).sum()
            bad_inf = inf_counts[inf_counts > 0]
            bad_large = large_counts[large_counts > 0]
            if len(bad_inf):
                print("Found infinity counts in features (col:count):")
                for c, cnt in bad_inf.items():
                    print(f"  - {c}: {int(cnt)}")
            if len(bad_large):
                print("Found extremely large values in features (>|1e6|) (col:count):")
                for c, cnt in bad_large.items():
                    print(f"  - {c}: {int(cnt)}")
        except Exception as e:
            print("  Warning: failed to compute inf/large counts:", e)

        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        try:
            df[feature_cols] = df[feature_cols].clip(-1e6, 1e6)
        except Exception:
            for c in feature_cols:
                df[c] = np.clip(df[c].astype(float), -1e6, 1e6)

        before = len(df)
        required_for_model = feature_cols + ['target']
        df = df.dropna(subset=required_for_model)
        after = len(df)
        print(f"🧾 Data cleaning: removed {before - after} rows with NaN/inf in features/target. Remaining rows: {after:,}")

        if after < 50:
            print("❌ Not enough data after cleaning to train. Aborting.")
            return None, None
    else:
        print("❗ No feature columns selected; aborting training.")
        return None, None

    # informative cross-coin validation
    trainer.validate_across_coins(df, feature_cols)

    # train final model (this function will do time-safe per-coin split internally)
    model, scaler = trainer.train_final_universal_model(df, feature_cols, test_size=test_size)
    if model is not None:
        # optimize decision threshold on validation (part of test) to maximize realized trailing return
        print("🔎 Optimizing decision threshold (validation -> holdout, leakage-free)...")
        thr, stats = optimize_decision_threshold(model, scaler, df, feature_cols, test_size=test_size, min_trades=20, holdout_frac=0.5)
        base_rate = stats.get('base_rate', None) if isinstance(stats, dict) else None
        # If optimize_decision_threshold returned a base_rate include it, else compute directly
        if base_rate is None:
            base_rate = float(df['target'].mean()) if 'target' in df.columns else None
        print(f"Chosen decision threshold: {thr} (stats: {stats})")
        print(f"Detected base_rate (BUY prevalence): {base_rate}")

        # Save model + trailing params + threshold metadata for safer inference
        trainer.coin_trailing_params = coin_trailing_params
        # Save with suggested inference defaults (we pick conservative defaults)
        trainer.save_universal_model('universal_scalp',
                                    threshold=thr,
                                    min_prob=DEFAULT_MIN_PROB,
                                    prob_edge=DEFAULT_PROB_EDGE,
                                    top_pct=DEFAULT_TOP_PCT,
                                    base_rate=base_rate)

        # Save feature list and trailing params + threshold metadata (same as before)
        with open('universal_features.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        with open('universal_trailing_params.pkl', 'wb') as f:
            pickle.dump(coin_trailing_params, f)
        print("Saved feature list, trailing params, and threshold (with stats).")
    return model, scaler

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Universal trainer (leak-safe, trailing-optimized)')
    parser.add_argument('--data', type=str, default='universal_features.csv')
    parser.add_argument('--fee', type=float, default=0.1, help='fee %% (e.g. 0.1)')
    parser.add_argument('--profit', type=float, default=0.6, help='target profit %% (e.g. 0.6)')
    parser.add_argument('--lookforward', type=int, default=8)
    parser.add_argument('--stoploss', type=float, default=1.0)
    parser.add_argument('--slippage', type=float, default=0.05)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--tp-atr-mul', type=float, default=3.0)
    parser.add_argument('--sl-atr-mul', type=float, default=2.0)
    parser.add_argument('--no-optimize-trailing', action='store_true',
                        help='Disable per-coin trailing param search (faster)')
    parser.add_argument('--no-strong-signal', action='store_true',
                        help='Disable strong-signal labeling (fall back to trailing_return > 0)')
    parser.add_argument('--strong-signal-mult', type=float, default=1.0,
                        help='Multiplier of target profit for strong-signal threshold (e.g. 1.5)')
    args = parser.parse_args()

    model, scaler = train_universal_pipeline(
        data_file=args.data,
        fee_rate=args.fee/100.0,
        target_profit=args.profit,
        lookforward=args.lookforward,
        stop_loss=args.stoploss,
        slippage=args.slippage,
        calibrate=args.calibrate,
        tp_atr_mul=args.tp_atr_mul,
        sl_atr_mul=args.sl_atr_mul,
        optimize_trailing=not args.no_optimize_trailing,
        use_strong_signal=(not args.no_strong_signal),
        strong_signal_mult=args.strong_signal_mult
    )
    if model:
        print("Training completed successfully.")
    else:
        print("Training failed.")