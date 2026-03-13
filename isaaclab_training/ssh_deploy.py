import paramiko
import time
import sys

HOST = "fe91fae6a6756695.natapp.cc"
PORT = 12346
USER = "bsrl"
PASS = os.environ.get("SSH_PASS", "")

LOCAL_FILE = r"D:\brainstem-master\nova-robotarm\isaaclab_training\envs\nova_pick_place_env.py"
REMOTE_FILE = "/home/bsrl/hongsenpang/nova_training/envs/nova_pick_place_env.py"

SMOKE_CMD = (
    "nohup bash -c 'export CUDA_VISIBLE_DEVICES=6 && "
    "export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2 && "
    "export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab && "
    "bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/train.py "
    "--headless --num_envs 64 --max_iterations 30' "
    "> /home/bsrl/hongsenpang/nova_training/smoke_v5.log 2>&1 &"
)

FULL_TRAIN_CMD = (
    "nohup bash -c 'export CUDA_VISIBLE_DEVICES=4,5,6 && "
    "export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2 && "
    "export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab && "
    "bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/train.py "
    "--headless --num_envs 1024 --max_iterations 3000' "
    "> /home/bsrl/hongsenpang/nova_training/step3v5_train.log 2>&1 &"
)

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    return client

def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    return out, err

def main():
    print("=== Step 1: Connecting and uploading file ===")
    client = connect()
    print("Connected.")

    # Upload file
    sftp = client.open_sftp()
    # Ensure remote dir exists
    try:
        sftp.stat("/home/bsrl/hongsenpang/nova_training/envs")
    except FileNotFoundError:
        sftp.mkdir("/home/bsrl/hongsenpang/nova_training/envs")
    sftp.put(LOCAL_FILE, REMOTE_FILE)
    sftp.close()
    print(f"Uploaded: {LOCAL_FILE} -> {REMOTE_FILE}")

    # Verify upload
    out, err = run_cmd(client, f"ls -lh {REMOTE_FILE}")
    print(f"Remote file: {out.strip()}")

    print("\n=== Step 2: Launching smoke test (64 envs, 30 iterations, GPU 6) ===")
    out, err = run_cmd(client, SMOKE_CMD)
    print(f"Smoke cmd stdout: {out!r}")
    print(f"Smoke cmd stderr: {err!r}")

    # Poll every 30s up to 5 minutes
    print("\nPolling smoke_v5.log every 30s (max 5 min)...")
    smoke_done = False
    log_tail = ""
    for attempt in range(10):  # 10 * 30s = 5 min
        time.sleep(30)
        out, err = run_cmd(client, "tail -30 /home/bsrl/hongsenpang/nova_training/smoke_v5.log 2>/dev/null || echo 'LOG_NOT_FOUND'")
        log_tail = out
        print(f"\n--- Poll {attempt+1} (t={30*(attempt+1)}s) ---")
        print(log_tail)

        # Check if training finished or errored
        if any(kw in out for kw in ["Traceback", "Error", "error", "exception"]) and "Mean reward" not in out:
            print("[POLL] Detected error in log before any reward — stopping early.")
            smoke_done = True
            break
        # Check completion indicators
        if any(kw in out for kw in ["Training complete", "Finished", "iteration 30", "Iteration: 30", "[30/30]", "step=30"]):
            print("[POLL] Smoke test appears complete.")
            smoke_done = True
            break
        # Check if process is still running
        chk_out, _ = run_cmd(client, "pgrep -f 'train.py.*num_envs 64' | wc -l")
        if chk_out.strip() == "0" and attempt >= 1:
            print("[POLL] Process not running anymore — smoke test done.")
            smoke_done = True
            break

    print("\n=== Step 3: Final smoke_v5.log (last 30 lines) ===")
    out, err = run_cmd(client, "tail -30 /home/bsrl/hongsenpang/nova_training/smoke_v5.log 2>/dev/null")
    smoke_log = out
    print(smoke_log)

    # Get all reward lines
    out_reward, _ = run_cmd(client, "grep -i 'reward\\|Mean reward' /home/bsrl/hongsenpang/nova_training/smoke_v5.log 2>/dev/null | tail -30")
    print("\n--- Reward lines ---")
    print(out_reward)

    # Check for errors
    out_err, _ = run_cmd(client, "grep -i 'traceback\\|Error\\|exception' /home/bsrl/hongsenpang/nova_training/smoke_v5.log 2>/dev/null | head -20")
    print("\n--- Error lines ---")
    print(out_err if out_err else "(none found)")

    # Evaluate smoke test
    has_error = bool(out_err.strip())
    # Check for positive rewards
    import re
    reward_vals = re.findall(r'[Mm]ean\s+reward[:\s]+([+-]?\d+\.?\d*)', smoke_log + out_reward)
    reward_vals += re.findall(r'reward[:\s=]+([+-]?\d+\.?\d*)', smoke_log + out_reward)
    print(f"\nExtracted reward values: {reward_vals}")

    positive_reward = False
    if reward_vals:
        try:
            floats = [float(v) for v in reward_vals]
            print(f"Reward floats: {floats}")
            positive_reward = any(f > 0 for f in floats)
        except:
            pass

    print(f"\nSmoke test assessment: has_error={has_error}, positive_reward={positive_reward}")

    if has_error:
        print("\n[ABORT] Errors detected in smoke test. Skipping full training.")
        client.close()
        return

    if not positive_reward:
        print("\n[ABORT] No positive rewards detected. Skipping full training.")
        print("(Rewards seen:", reward_vals, ")")
        client.close()
        return

    print("\n=== Step 4: Launching full training (GPUs 4,5,6, 1024 envs, 3000 iter) ===")
    out, err = run_cmd(client, FULL_TRAIN_CMD)
    print(f"Full train stdout: {out!r}")
    print(f"Full train stderr: {err!r}")

    # Get PID
    time.sleep(3)
    pid_out, _ = run_cmd(client, "pgrep -f 'train.py.*num_envs 1024' | head -5")
    print(f"Full training PID(s): {pid_out.strip()}")

    print("\n=== Step 5: Waiting 20 minutes for full training metrics ===")
    print("Waiting 1200 seconds...")
    time.sleep(1200)

    print("\n--- Last 30 lines of step3v5_train.log ---")
    out, _ = run_cmd(client, "tail -30 /home/bsrl/hongsenpang/nova_training/step3v5_train.log 2>/dev/null")
    print(out)

    print("\n--- All 'Mean reward' lines ---")
    out, _ = run_cmd(client, "grep 'Mean reward' /home/bsrl/hongsenpang/nova_training/step3v5_train.log 2>/dev/null")
    print(out)

    print("\n--- Last 20 'action noise' lines ---")
    out, _ = run_cmd(client, "grep 'action noise' /home/bsrl/hongsenpang/nova_training/step3v5_train.log 2>/dev/null | tail -20")
    print(out)

    client.close()
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
