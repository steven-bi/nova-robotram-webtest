"""Record Isaac Sim rendered video of trained policy."""
import paramiko
import os

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.westd.seetacloud.com', port=14918, username='root', password='UvUnT2x1jsaa', timeout=15)

sftp = client.open_sftp()

# Upload play_record.py to server
local_script = os.path.join(os.path.dirname(__file__), 'play_record_sim.py')

# Write the script content locally first, then upload
script_content = open(local_script, 'r').read()
with sftp.file('/root/autodl-tmp/arm_grasp/scripts/play_record.py', 'w') as f:
    f.write(script_content)
sftp.close()

print("Running video recording (this takes a few minutes)...")
cmd = (
    'cd /root/autodl-tmp/arm_grasp && '
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
    '/root/autodl-tmp/conda_envs/thunder2/bin/python -u scripts/play_record.py 2>&1 | tail -40'
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=600)
exit_code = stdout.channel.recv_exit_status()
output = stdout.read().decode()
print(output[-4000:])
print("Exit:", exit_code)

# Download video if it exists
if exit_code == 0:
    sftp = client.open_sftp()
    try:
        sftp.stat('/root/autodl-tmp/arm_grasp/policy_video.mp4')
        sftp.get('/root/autodl-tmp/arm_grasp/policy_video.mp4',
                 'D:/inovxio/products/robotarm_ws/policy_video.mp4')
        print("Video downloaded to policy_video.mp4")
    except FileNotFoundError:
        print("Video file not found on server")
    sftp.close()

client.close()
