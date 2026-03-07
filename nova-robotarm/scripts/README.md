# Scripts

## 常用脚本

| 脚本 | 说明 |
|------|------|
| `start_training.py` | 启动 PPO 训练（部署到 AutoDL 服务器） |
| `deploy.py` | 部署最新训练代码到服务器（v8.10，当前版本）|
| `play_script.py` | 本地仿真回放已训练策略 |
| `play_record_sim.py` | 仿真回放并录制视频 |
| `record_video.py` | 录制训练过程视频 |
| `record_sim_video.py` | 录制仿真视频 |
| `run_validation.py` | 验证训练结果 |
| `sync_and_download.py` | 从服务器同步下载训练结果 |
| `visualize_trajectory.py` | 可视化末端执行器轨迹 |

## archive/

历史迭代部署脚本，包含每个版本的奖励函数改动记录和失败原因分析。

| 版本 | 关键改动 |
|------|---------|
| v8.5 | 强制俯视抓取（EE height reward），解决侧抓局部最优 |
| v8.6 | 修复奖励冲突导致的振荡 |
| v8.7 | 修复"悬停+闭合夹爪"局部最优 |
| v8.8 | 基于论文的奖励函数根本性重设计 |
| v8.9 | 修复夹爪从不打开（一直保持闭合）|
| v8.10 | 修复 drop_penalty 阈值 bug + grasp_threshold 过大 bug（当前） |
