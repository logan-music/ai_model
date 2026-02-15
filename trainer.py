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


class UniversalModelTrainer:
    def __init__(self, fee_rate=0.001, target_profit_pct=0.6, lookforward_candles=6,
                 stop_loss_pct=1.0, slippage_pct=0.05, calibrate=False):
        """
        Defaults tuned for slow-grind continuation:
          - target_profit_pct = 0.6  (0.6%)
          - lookforward_candles = 6
          - stop_loss_pct = 1.0
        """
        self.fee_rate = fee_rate
        self.target_profit = target_profit_pct / 100.0
        self.lookforward = lookforward_candles
        self.stop_loss = stop_loss_pct / 100.0
        self.slippage = slippage_pct / 100.0
        self.calibrate = calibrate

        self.scaler = StandardScaler()
        self.model = None
        print(f"✅ Trainer init: fee={fee_rate*100:.2f}% target={target_profit_pct}% lookforward={lookforward_candles} stop_loss={stop_loss_pct}% slippage={slippage_pct}% calibrate={calibrate}")

    def create_universal_labels(self, df):
        """
        Create labels per coin. Uses TP-before-SL logic:
        For each entry (row i) compute if within next lookforward candles:
            - price reaches TP_level = entry_price * (1 + target + fee + slippage)
              BEFORE any price hits SL_level = entry_price * (1 - stop_loss - slippage)
        If TP hit first -> label 1 (BUY), else 0.
        Adds 'signal_volume_ok' column (optional filter).
        """
        print("🔖 Creating labels per coin (TP-before-SL logic)")
        df = df.copy()
        df['future_return'] = np.nan
        df['target'] = 0
        df['signal_volume_ok'] = 0

        # Ensure index is datetime
        df = df.sort_index()

        symbols = df['symbol'].unique()
        for sym in symbols:
            coin_df = df[df['symbol'] == sym].copy()
            n = len(coin_df)
            if n == 0:
                continue

            closes = coin_df['close'].values
            highs = coin_df['high'].values
            lows = coin_df['low'].values

            future_returns = [np.nan] * n
            labels = [0] * n

            # compute volume_ratio based signal (optional)
            vol_ok = (coin_df.get('volume_ratio', pd.Series(0, index=coin_df.index)) > 0.8).astype(int)

            for i in range(n):
                entry_price = closes[i]
                # realistic effective entry (fees taken on buy)
                entry_cost = entry_price * (1 + self.fee_rate + self.slippage)

                tp_price = entry_price * (1 + self.target_profit + self.fee_rate + self.slippage)
                sl_price = entry_price * (1 - self.stop_loss - self.slippage)

                hit_tp = False
                hit_sl = False
                # search forward within this coin only
                for j in range(i+1, min(n, i+1+self.lookforward)):
                    # use the candle high/low as possible fill window (conservative)
                    if highs[j] >= tp_price and not hit_sl:
                        # TP hit at or before SL
                        hit_tp = True
                        # net return approximated as tp minus entry cost over entry cost
                        exit_revenue = tp_price * (1 - self.fee_rate - self.slippage)
                        net_return = (exit_revenue - entry_cost) / (entry_cost + 1e-12)
                        future_returns[i] = net_return
                        labels[i] = 1
                        break
                    if lows[j] <= sl_price:
                        # SL hit first - loss
                        hit_sl = True
                        exit_revenue = sl_price * (1 - self.fee_rate - self.slippage)
                        net_return = (exit_revenue - entry_cost) / (entry_cost + 1e-12)
                        future_returns[i] = net_return
                        labels[i] = 0
                        break
                if not hit_tp and not hit_sl:
                    # neither hit within window: compute best achievable net_return (conservative: max high seen)
                    future_window_high = highs[i+1:min(n, i+1+self.lookforward)].max() if i+1 < min(n, i+1+self.lookforward) else highs[i]
                    exit_revenue = future_window_high * (1 - self.fee_rate - self.slippage)
                    net_return = (exit_revenue - entry_cost) / (entry_cost + 1e-12)
                    future_returns[i] = net_return
                    labels[i] = 1 if net_return > self.target_profit else 0

            # assign back to df
            df.loc[coin_df.index, 'future_return'] = future_returns
            df.loc[coin_df.index, 'target'] = labels
            df.loc[coin_df.index, 'signal_volume_ok'] = vol_ok.values

            print(f"  {sym}: labeled {sum(labels)} / {n} as BUY ({(sum(labels)/n*100):.2f}%)")

        # Overall stats
        total = df['target'].dropna().shape[0]
        buys = int(df['target'].sum())
        print(f"\nTOTAL: {buys} BUY signals out of {total} samples ({(buys/total*100) if total else 0:.2f}%)")
        return df

    def select_universal_features(self, df):
        exclude = [
            'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
            'future_return', 'target', 'symbol'
        ]
        feature_cols = [col for col in df.columns if col not in exclude and df[col].dtype in [float, int]]
        # drop near-constant or NaN-heavy features
        feature_cols = [c for c in feature_cols if df[c].notna().sum() > len(df) * 0.7]
        print(f"Selected {len(feature_cols)} features")
        return feature_cols

    def validate_across_coins(self, df, feature_cols):
        """
        Time-aware cross-coin validation:
        For each test coin, train on other coins but USE ONLY data that is older than test coin's test window start.
        This avoids 'seeing the future' across coins.
        """
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
            # train on other coins BUT only data strictly before test_start
            train_df = df[(df['symbol'] != test_coin) & (df.index < test_start)].dropna()
            if len(train_df) < 100:
                # fallback: train on all other coins (no perfect time split available)
                train_df = df[df['symbol'] != test_coin].dropna()
            X_train = train_df[feature_cols].values
            y_train = train_df['target'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['target'].values
            if len(np.unique(y_train)) < 2:
                print(f"Insufficient class diversity in training for {test_coin}, skipping.")
                continue

            # scale
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
            # calibrate probabilities (slow but recommended)
            print("Calibrating classifier probabilities (isotonic)...")
            calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
            calibrated.fit(X_train, y_train)
            print("Calibration complete.")
            return calibrated
        return ensemble

    def train_final_universal_model(self, df, feature_cols, test_size=0.2):
        """
        Time-safe train/test split per coin.
        For each coin we split by time and then concatenate.
        """
        print("⏳ Preparing time-safe train/test split per coin...")
        train_parts = []
        test_parts = []
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

        # scale (trees don't require scaling but we may keep for other models)
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # train ensemble (optionally calibrated)
        self.model = self.train_universal_ensemble(X_train_s, y_train)

        # Evaluate
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

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix:\n", cm)

        # Per-coin performance
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

    def save_universal_model(self, model_prefix='universal_scalp'):
        if self.model is None:
            print("No model to save.")
            return False
        model_file = f"{model_prefix}_model.pkl"
        scaler_file = f"{model_prefix}_scaler.pkl"
        try:
            # Use joblib for robust serialization of sklearn ensembles
            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            print(f"Saved {model_file}, {scaler_file}")
            return True
        except Exception as e:
            print(f"Error saving model/scaler: {e}")
            return False


def train_universal_pipeline(data_file='universal_features.csv',
                             fee_rate=0.001, target_profit=0.6, lookforward=6,
                             stop_loss=1.0, slippage=0.05, calibrate=False):
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
        calibrate=calibrate
    )

    df = trainer.create_universal_labels(df)
    feature_cols = trainer.select_universal_features(df)
    trainer.validate_across_coins(df, feature_cols)
    model, scaler = trainer.train_final_universal_model(df, feature_cols)
    if model is not None:
        trainer.save_universal_model('universal_scalp')
        # Save feature list for inference
        with open('universal_features.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        print("Saved feature list: universal_features.pkl")
    return model, scaler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Universal trainer (leak-safe)')
    parser.add_argument('--data', type=str, default='universal_features.csv')
    parser.add_argument('--fee', type=float, default=0.1)
    parser.add_argument('--profit', type=float, default=0.6)
    parser.add_argument('--lookforward', type=int, default=6)
    parser.add_argument('--stoploss', type=float, default=1.0)
    parser.add_argument('--slippage', type=float, default=0.05)
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()

    model, scaler = train_universal_pipeline(
        data_file=args.data,
        fee_rate=args.fee/100.0,
        target_profit=args.profit,
        lookforward=args.lookforward,
        stop_loss=args.stoploss,
        slippage=args.slippage,
        calibrate=args.calibrate
    )
    if model:
        print("Training completed successfully.")
    else:
        print("Training failed.")