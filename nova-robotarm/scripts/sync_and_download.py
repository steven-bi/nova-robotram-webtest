"""
1. Download relevant RL + robot manipulation papers from arxiv
2. Sync server arm_grasp code to local
"""
import os
import time
import urllib.request
import paramiko
import stat

# ── Config ──────────────────────────────────────────────────────────────
SSH_HOST = 'connect.westd.seetacloud.com'
SSH_PORT = 14918
SSH_USER = 'root'
SSH_PASS = 'UvUnT2x1jsaa'
REMOTE_BASE = '/root/autodl-tmp/arm_grasp'
LOCAL_PAPERS = r'D:\inovxio\products\robotarm_ws\ref\papers'
LOCAL_CODE   = r'D:\inovxio\products\robotarm_ws\server_code'

# ── Papers to download (arxiv ID → filename) ────────────────────────────
PAPERS = [
    # Core pick-and-place / reward design
    ("2404.16779", "DrS_dense_reward_multistage_ICLR2024.pdf"),
    ("2503.11012", "sim2real_pick_place_long_horizon_ICRA2025.pdf"),
    ("2502.08643", "real2sim2real_VLM_keypoint_reward_2025.pdf"),
    ("2503.01837", "multistage_manipulation_demo_augmented_reward_2025.pdf"),
    # Reward shaping + contact force
    ("2410.13124", "just_add_force_contact_rich_policies_2024.pdf"),
    ("2502.20396", "sim2real_dexterous_manipulation_humanoids_2025.pdf"),
    ("2502.15442", "long_horizon_manipulation_privileged_action_2025.pdf"),
    # VLA / foundation models for manipulation
    ("2505.18719", "VLA_RL_general_robotic_manipulation_2025.pdf"),
    ("2512.01801", "GR_RL_dexterous_long_horizon_2024.pdf"),
    # Isaac Lab framework
    ("2511.04831", "isaac_lab_GPU_sim_framework_2025.pdf"),
]


def download_papers():
    os.makedirs(LOCAL_PAPERS, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Downloading Papers")
    print(f"{'='*60}")
    for arxiv_id, filename in PAPERS:
        dst = os.path.join(LOCAL_PAPERS, filename)
        if os.path.exists(dst) and os.path.getsize(dst) > 10000:
            print(f"  [skip] {filename} (already exists)")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"  Downloading {arxiv_id} → {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dst, 'wb') as f:
                data = resp.read()
                f.write(data)
            size_kb = os.path.getsize(dst) // 1024
            print(f"    OK  ({size_kb} KB)")
        except Exception as e:
            print(f"    FAIL: {e}")
        time.sleep(1)  # be polite to arxiv


def sftp_download_dir(sftp, remote_dir, local_dir):
    """Recursively download remote_dir to local_dir."""
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        rpath = remote_dir + '/' + entry.filename
        lpath = os.path.join(local_dir, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            sftp_download_dir(sftp, rpath, lpath)
        else:
            sftp.get(rpath, lpath)


def sync_server_code():
    print(f"\n{'='*60}")
    print("  Syncing Server Code")
    print(f"{'='*60}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=15)
    sftp = client.open_sftp()

    # List remote files first
    _, stdout, _ = client.exec_command(f'find {REMOTE_BASE} -name "*.py" -o -name "*.yaml" -o -name "*.urdf" | sort')
    remote_files = stdout.read().decode().strip().split('\n')
    print(f"  Found {len(remote_files)} files on server")

    # Download each file preserving directory structure
    os.makedirs(LOCAL_CODE, exist_ok=True)
    ok = 0
    fail = 0
    for rpath in remote_files:
        if not rpath.strip():
            continue
        rel = rpath[len(REMOTE_BASE):].lstrip('/')
        lpath = os.path.join(LOCAL_CODE, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(lpath), exist_ok=True)
        try:
            sftp.get(rpath, lpath)
            ok += 1
        except Exception as e:
            print(f"  FAIL {rel}: {e}")
            fail += 1

    # Also download logs directory structure (just model list, not weights)
    _, stdout, _ = client.exec_command(
        f'ls /root/autodl-tmp/arm_grasp/logs/rsl_rl/ 2>/dev/null'
    )
    exps = stdout.read().decode().strip()
    print(f"\n  Experiments on server:\n{exps}")

    sftp.close()
    client.close()
    print(f"\n  Sync complete: {ok} files OK, {fail} failed")
    print(f"  Local path: {LOCAL_CODE}")


if __name__ == '__main__':
    download_papers()
    sync_server_code()
    print("\nAll done!")
    print(f"Papers: {LOCAL_PAPERS}")
    print(f"Code:   {LOCAL_CODE}")
