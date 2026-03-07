"""Deploy v9.0 pick-place environment to SeetaCloud and start training."""
import time
import paramiko

# Read local files
with open('server_code/arm_grasp/envs/pick_place_cfg.py', encoding='utf-8') as f:
    cfg_content = f.read()
with open('server_code/arm_grasp/agents/rsl_rl_ppo_cfg.py', encoding='utf-8') as f:
    ppo_content = f.read()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.westd.seetacloud.com', port=14918,
               username='root', password='UvUnT2x1jsaa', timeout=15)
print("Connected.")

# Kill any running training
client.exec_command('pkill -9 -f "train_pick_place" 2>/dev/null', timeout=5)
time.sleep(2)

# Upload files
sftp = client.open_sftp()
base = '/root/autodl-tmp/arm_grasp/arm_grasp'

with sftp.file(f'{base}/envs/pick_place_cfg.py', 'w') as f:
    f.write(cfg_content)
print("Uploaded pick_place_cfg.py")

with sftp.file(f'{base}/agents/rsl_rl_ppo_cfg.py', 'w') as f:
    f.write(ppo_content)
print("Uploaded rsl_rl_ppo_cfg.py")

sftp.close()

# Syntax check only (no Isaac Sim needed)
syntax_cmd = (
    '/root/autodl-tmp/conda_envs/thunder2/bin/python -m py_compile '
    '/root/autodl-tmp/arm_grasp/arm_grasp/envs/pick_place_cfg.py '
    '/root/autodl-tmp/arm_grasp/arm_grasp/agents/rsl_rl_ppo_cfg.py && echo OK'
)
stdin, stdout, stderr = client.exec_command(syntax_cmd, timeout=10)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(f"Syntax check: {out or 'no output'}")
if err:
    print(f"Syntax errors: {err}")
    client.close()
    exit(1)
if 'OK' not in out:
    print("Syntax check failed")
    client.close()
    exit(1)

# Launch training
train_cmd = (
    'cd /root/autodl-tmp/arm_grasp && '
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
    'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u '
    'scripts/train_pick_place.py --num_envs 2048 --headless '
    '> /root/autodl-tmp/arm_grasp/train_pick_place.log 2>&1 &'
)
client.exec_command(train_cmd, timeout=10)
time.sleep(5)

# Verify started
stdin, stdout, stderr = client.exec_command(
    'ps aux | grep train_pick_place | grep -v grep | head -1', timeout=5
)
ps = stdout.read().decode().strip()
if ps:
    pid = ps.split()[1]
    print(f"Training started! PID={pid}")
    print(f"Experiment: arm_pick_place_v9_0, max_iters=18000")
else:
    print("WARNING: training process not found!")
    stdin, stdout, stderr = client.exec_command(
        'tail -20 /root/autodl-tmp/arm_grasp/train_pick_place.log', timeout=5
    )
    print(stdout.read().decode())

client.close()
