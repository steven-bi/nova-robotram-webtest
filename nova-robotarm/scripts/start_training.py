"""Start PPO training on server."""
import paramiko
import time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.westd.seetacloud.com', port=14918, username='root', password='UvUnT2x1jsaa', timeout=15)

# Clean old logs
client.exec_command('rm -rf /root/autodl-tmp/arm_grasp/logs/', timeout=5)
time.sleep(1)

# Start training
print("Starting PPO training (final config: fixed scene + gripper + grasp reward)...")
cmd = (
    'cd /root/autodl-tmp/arm_grasp && '
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
    'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u scripts/train.py '
    '--num_envs 2048 --max_iterations 5000 --headless '
    '> /root/autodl-tmp/arm_grasp/train_output.log 2>&1 &'
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=10)
time.sleep(5)

stdin, stdout, stderr = client.exec_command('pgrep -f "train.py" | head -1', timeout=5)
pid = stdout.read().decode().strip()
print(f"Training PID: {pid}")

# Wait for first iterations
time.sleep(60)
stdin, stdout, stderr = client.exec_command('tail -40 /root/autodl-tmp/arm_grasp/train_output.log 2>/dev/null', timeout=10)
output = stdout.read().decode()
print(output[-2500:])

client.close()
