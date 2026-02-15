#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import warnings
import threading
import time
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

warnings.filterwarnings('ignore')


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║          UNIVERSAL MULTI-COIN SCALPING MODEL TRAINER     ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def push_model_to_github(model_filename="universal_model.pkl"):
    """
    Push model file to repo using token from GIT_TOKEN or GITHUB_TOKEN env var.
    - Expects origin remote configured.
    - Will temporarily set origin URL to include token, push current branch, then restore original URL.
    """
    token = os.getenv('GIT_TOKEN') or os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GIT_TOKEN not set in env. Set GIT_TOKEN to a personal access token with 'repo' scope.")
        return False

    git_user_name = os.getenv('GIT_USER_NAME', 'Render Trainer')
    git_user_email = os.getenv('GIT_USER_EMAIL', 'render@example.com')

    original_origin = None
    try:
        # Git config name/email
        subprocess.run(["git", "config", "--global", "user.email", git_user_email], check=False)
        subprocess.run(["git", "config", "--global", "user.name", git_user_name], check=False)

        # get origin url
        origin = subprocess.check_output(["git", "remote", "get-url", "origin"]).decode().strip()
        original_origin = origin

        # convert SSH -> https if needed
        if origin.startswith("git@"):
            origin_https = origin.replace(":", "/").replace("git@", "https://")
        else:
            origin_https = origin

        # prepare tokenized url
        if origin_https.startswith("https://"):
            token_url = origin_https.replace("https://", f"https://{token}@")
        else:
            token_url = f"https://{token}@{origin_https}"

        # set remote to tokenized URL temporarily
        subprocess.run(["git", "remote", "set-url", "origin", token_url], check=True)

        # add, commit, push current branch
        subprocess.run(["git", "add", model_filename], check=False)
        try:
            subprocess.run(["git", "commit", "-m", "Auto: Add trained universal model"], check=True)
        except subprocess.CalledProcessError:
            # nothing to commit or commit failed (e.g., identical content), continue to push anyway
            pass

        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        except Exception:
            branch = "main"

        print(f"📤 Pushing {model_filename} to origin/{branch} ...")
        try:
            subprocess.run(["git", "push", "origin", branch], check=True)
            print("✅ Model pushed to GitHub successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Git push failed: {e}; try pushing manually.")
            return False

    except Exception as e:
        print(f"❌ push_model_to_github error: {e}")
        return False
    finally:
        # restore original origin if we changed it
        if original_origin:
            try:
                subprocess.run(["git", "remote", "set-url", "origin", original_origin], check=False)
            except Exception:
                pass

    return True


def run_complete_pipeline(skip_collection=False, coins=None, interval='5m',
                          lookback_days=120, fee=0.1, profit=0.6, lookforward=6,
                          stoploss=1.0, slippage=0.05, calibrate=False):
    """
    Orchestrates collection + training.
    Returns True on success, False on failure.
    Non-interactive: requires AUTO_APPROVE="true" (default) in env to run in CI.
    """
    print_banner()
    print("Configuration:")
    print("  coins:", coins if coins else "default")
    print("  interval:", interval, "days:", lookback_days)
    print("  fee:", fee, "profit:", profit, "lookforward:", lookforward)

    # Non-interactive guard for CI/cloud
    if os.getenv("AUTO_APPROVE", "true").lower() != "true":
        print("AUTO_APPROVE not enabled. Set AUTO_APPROVE=true to run in non-interactive environments.")
        return False

    # STEP 1: collect
    if not skip_collection:
        try:
            from data_collector import collect_universal_dataset
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
        from trainer import train_universal_pipeline
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

    return True


# ---------------------------
# Dummy HTTP server for Render
# ---------------------------
class DummyHandler(BaseHTTPRequestHandler):
    server_start_time = time.time()

    def _set_json(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def _set_html(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            # minimal health check
            self._set_json(200)
            payload = {
                "status": "ok",
                "uptime_seconds": int(time.time() - self.server_start_time)
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif self.path == "/":
            self._set_html(200)
            body = f"""
                <html>
                <head><title>Universal Trainer</title></head>
                <body>
                  <h1>Universal Trainer</h1>
                  <p>status: running</p>
                  <p>uptime_seconds: {int(time.time() - self.server_start_time)}</p>
                  <p>env PORT: {os.getenv('PORT')}</p>
                </body>
                </html>
            """
            self.wfile.write(body.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        # reduce noisy logs; print minimal
        sys.stdout.write("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))


def start_dummy_server_in_thread(port: int):
    """
    Starts a simple HTTP server in a background thread.
    Returns (server, thread) so caller can join/shutdown if desired.
    """
    try:
        server = HTTPServer(("0.0.0.0", port), DummyHandler)
    except OSError as e:
        print(f"⚠️ Failed to start dummy server on port {port}: {e}")
        return None, None

    thread = threading.Thread(target=server.serve_forever, name="DummyHTTPServer", daemon=False)
    thread.start()
    print(f"🔌 Dummy HTTP server started on 0.0.0.0:{port} (thread {thread.name})")
    return server, thread


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Universal pipeline runner (saves & optionally pushes model)')
    parser.add_argument('--skip-collection', action='store_true')
    parser.add_argument('--coins', nargs='+', default=None)
    parser.add_argument('--interval', default='5m')
    parser.add_argument('--days', type=int, default=120)
    parser.add_argument('--fee', type=float, default=0.1)
    parser.add_argument('--profit', type=float, default=0.6)
    parser.add_argument('--lookforward', type=int, default=6)
    parser.add_argument('--stoploss', type=float, default=1.0)
    parser.add_argument('--slippage', type=float, default=0.05)
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--push', action='store_true', help='Push model file to repo using GIT_TOKEN env var')
    args = parser.parse_args()

    # Convert coins to standard symbols
    coins = None
    if args.coins:
        coins = [c.upper() + 'USDT' if not c.upper().endswith('USDT') else c.upper() for c in args.coins]

    # Start dummy server early so render health checks can see it during run.
    port_env = os.getenv("PORT")
    try:
        port = int(port_env) if port_env else 10000
    except ValueError:
        port = 10000

    dummy_server, dummy_thread = start_dummy_server_in_thread(port)

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
    if not success:
        print("❌ Pipeline failed or cancelled.")
        # optional: keep server up to allow debug access; we'll exit with non-zero code
        # shutdown server cleanly if running
        if dummy_server:
            try:
                dummy_server.shutdown()
                dummy_server.server_close()
            except Exception:
                pass
        sys.exit(1)

    # Trainer should have saved universal_scalp_model.pkl; copy to universal_model.pkl for convenience
    scalp_model = "universal_scalp_model.pkl"
    public_model = "universal_model.pkl"
    if os.path.exists(scalp_model):
        try:
            shutil.copy2(scalp_model, public_model)
            print(f"✅ Copied {scalp_model} -> {public_model}")
        except Exception as e:
            print(f"⚠️ Failed to copy model file: {e}")
    else:
        # If trainer didn't save with that name, attempt to find a saved model file
        candidates = [f for f in os.listdir('.') if f.endswith('_model.pkl') or f.endswith('model.pkl')]
        if candidates:
            chosen = candidates[0]
            try:
                shutil.copy2(chosen, public_model)
                print(f"✅ Copied {chosen} -> {public_model}")
            except Exception as e:
                print(f"⚠️ Failed to copy discovered model file {chosen}: {e}")
        else:
            print("⚠️ No trainer-saved model file found (universal_scalp_model.pkl). You may want to save the model manually.")
            # Not fatal for exit; but we warn.

    # Optional: push model to GitHub (requires GIT_TOKEN/GITHUB_TOKEN)
    if args.push:
        if not os.path.exists(public_model):
            print(f"⚠️ {public_model} not found; nothing to push.")
            # shutdown dummy server before exit
            if dummy_server:
                try:
                    dummy_server.shutdown()
                    dummy_server.server_close()
                except Exception:
                    pass
            sys.exit(1)
        pushed = push_model_to_github(public_model)
        if not pushed:
            print("⚠️ Model not pushed. You can push manually from CI or locally.")
            if dummy_server:
                try:
                    dummy_server.shutdown()
                    dummy_server.server_close()
                except Exception:
                    pass
            sys.exit(1)
        else:
            print("✅ Model push complete.")

    print("✅ Pipeline completed successfully.")

    # If the environment requests the process to keep alive (useful for Render),
    # set KEEP_ALIVE=true in env. Otherwise script exits (and Render will stop the service).
    keep_alive = os.getenv("KEEP_ALIVE", "false").lower() == "true"
    if keep_alive and dummy_thread and dummy_thread.is_alive():
        try:
            print("🛑 KEEP_ALIVE=true — keeping process alive (press Ctrl+C to stop).")
            # Block here until interrupted; server runs in dummy_thread
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("🔌 Shutdown requested, stopping dummy server...")
        finally:
            try:
                dummy_server.shutdown()
                dummy_server.server_close()
            except Exception:
                pass
            print("✅ Dummy server stopped. Exiting.")
            sys.exit(0)
    else:
        # shutdown server cleanly if it was started (avoid lingering resources)
        if dummy_server:
            try:
                dummy_server.shutdown()
                dummy_server.server_close()
            except Exception:
                pass
        sys.exit(0)
