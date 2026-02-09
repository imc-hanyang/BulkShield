import subprocess
import time
import os
import sys
from datetime import datetime

# Configurations
PYTHON_BIN = "/home/srt/venvs/predict_models/bin/python"
SCRIPT_PATH = "/home/srt/workspaces/predict_models/transformer_without_added_labeling_main.py"
LOG_DIR = "/home/srt/workspaces/predict_models/logs/transformer_without_added_labeling"
NUM_RUNS = 20

def run_loop():
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Starting {NUM_RUNS} runs of {SCRIPT_PATH}...")
    print(f"Python: {PYTHON_BIN}")
    print(f"Logs will be saved to: {LOG_DIR}")
    print("-" * 50)

    for i in range(1, NUM_RUNS + 1):
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"transformer_main_{timestamp}.log"
        log_path = os.path.join(LOG_DIR, log_filename)

        print(f"[{i}/{NUM_RUNS}] Starting run at {timestamp}...")
        print(f"  -> Logging to {log_path}")

        # Construct command
        # We use shell=True to easily handle voltage redirection '>' 
        # But for better safety/control, we opens the file and passes as stdout.
        
        try:
            with open(log_path, "w") as log_file:
                # Execute the script
                process = subprocess.Popen(
                    [PYTHON_BIN, SCRIPT_PATH],
                    stdout=log_file,
                    stderr=subprocess.STDOUT, # Redirect stderr to stdout (log file)
                )
                
                # Wait for completion
                return_code = process.wait()
                
            if return_code == 0:
                print(f"  -> Run {i} completed successfully.")
            else:
                print(f"  -> Run {i} failed with return code {return_code}. Check log for details.")

        except Exception as e:
            print(f"  -> Error occurred during run {i}: {e}")

        # Optional: slight pause between runs
        time.sleep(1)
        print("-" * 50)

    print("All runs completed.")

if __name__ == "__main__":
    run_loop()
