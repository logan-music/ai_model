import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║          UNIVERSAL MULTI-COIN SCALPING MODEL TRAINER     ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_complete_pipeline(skip_collection=False, coins=None, interval='5m',
                          lookback_days=30, fee=0.1, profit=0.5, lookforward=10,
                          stoploss=1.5, slippage=0.05, calibrate=False):
    print_banner()
    print("Configuration:")
    print("  coins:", coins if coins else "default")
    print("  interval:", interval, "days:", lookback_days)
    print("  fee:", fee, "profit:", profit, "lookforward:", lookforward)
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return False

    # STEP 1: collect
    if not skip_collection:
        try:
            from universal_data_collector import collect_universal_dataset
            df = collect_universal_dataset(symbols=coins, interval=interval,
                                           lookback_days=lookback_days, save_to='universal_features.csv')
            if df is None:
                print("Data collection failed.")
                return False
            print("Data collected.")
        except Exception as e:
            print("Collection error:", e)
            return False
    else:
        if not os.path.exists('universal_features.csv'):
            print("universal_features.csv not found.")
            return False
        print("Using existing universal_features.csv")

    # STEP 2: train
    try:
        from universal_trainer import train_universal_pipeline
        model, scaler = train_universal_pipeline(
            data_file='universal_features.csv',
            fee_rate=fee/100.0,
            target_profit=profit,
            lookforward=lookforward,
            stop_loss=stoploss,
            slippage=slippage,
            calibrate=calibrate
        )
        if model is None:
            print("Training failed.")
            return False
        print("Model trained.")
    except Exception as e:
        print("Training error:", e)
        return False

    print("\nPipeline complete. Files produced:")
    for f in ['universal_features.csv', 'universal_scalp_model.pkl', 'universal_scalp_scaler.pkl', 'universal_features.pkl']:
        if os.path.exists(f):
            print("  -", f)
    return True

def quick_train():
    return run_complete_pipeline(skip_collection=False, coins=None, interval='5m', lookback_days=30,
                                 fee=0.1, profit=0.5, lookforward=10, stoploss=1.5, slippage=0.05, calibrate=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Universal pipeline updated')
    parser.add_argument('--skip-collection', action='store_true')
    parser.add_argument('--coins', nargs='+', default=None)
    parser.add_argument('--interval', default='5m')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--fee', type=float, default=0.1)
    parser.add_argument('--profit', type=float, default=0.6)
    parser.add_argument('--lookforward', type=int, default=6)
    parser.add_argument('--stoploss', type=float, default=1.0)
    parser.add_argument('--slippage', type=float, default=0.05)
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()

    coins = None
    if args.coins:
        coins = [c.upper() + 'USDT' if not c.upper().endswith('USDT') else c.upper() for c in args.coins]

    success = run_complete_pipeline(skip_collection=args.skip_collection,
                                    coins=coins,
                                    interval=args.interval,
                                    lookback_days=args.days,
                                    fee=args.fee,
                                    profit=args.profit,
                                    lookforward=args.lookforward,
                                    stoploss=args.stoploss,
                                    slippage=args.slippage,
                                    calibrate=args.calibrate)
    sys.exit(0 if success else 1)