"""HER (Hindsight Experience Replay) — Episode Buffer + Goal Relabeling

支持 PPO-HER 的 episode 级别轨迹存储和 future 策略重标签。

用法：
    buf = HerEpisodeBuffer(num_envs, max_ep_len, her_k=4, device=device)
    # 每步调用：
    buf.add_step(env_step, obs, achieved_goal, desired_goal, action, reward, done)
    # episode 结束时：
    her_transitions = buf.flush_done_envs(done_mask, reward_fn)
    # her_transitions 可直接追加到 PPO storage
"""
from __future__ import annotations
import torch


class HerEpisodeBuffer:
    """按 env 并行存储 episode 轨迹，episode 结束后生成 HER 重标签样本。

    参数
    ----
    num_envs    : 并行环境数
    max_ep_len  : 最大 episode 长度（超过则强制 flush）
    her_k       : 每条真实 transition 生成的 HER 样本数（Future 策略）
    device      : torch device
    """

    def __init__(
        self,
        num_envs: int,
        max_ep_len: int,
        her_k: int = 4,
        device: str = "cuda",
    ):
        self.num_envs  = num_envs
        self.max_ep_len = max_ep_len
        self.her_k     = her_k
        self.device    = device

        # 每个 env 独立维护一条 episode buffer（list of dicts）
        self._eps: list[list[dict]] = [[] for _ in range(num_envs)]

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def add_step(
        self,
        step_idx: int,
        obs: torch.Tensor,           # (N, obs_dim)
        achieved_goal: torch.Tensor, # (N, goal_dim)
        desired_goal: torch.Tensor,  # (N, goal_dim)
        action: torch.Tensor,        # (N, act_dim)
        reward: torch.Tensor,        # (N,)
        done: torch.Tensor,          # (N,) bool
    ):
        """向每个 env 的 episode buffer 追加当前步数据。"""
        for i in range(self.num_envs):
            self._eps[i].append({
                "obs":            obs[i].clone(),
                "achieved_goal":  achieved_goal[i].clone(),
                "desired_goal":   desired_goal[i].clone(),
                "action":         action[i].clone(),
                "reward":         reward[i].clone(),
                "done":           done[i].item(),
            })

    def flush_done_envs(
        self,
        done_mask: torch.Tensor,   # (N,) bool — 哪些 env 完成了 episode
        reward_fn,                  # callable(achieved_goal, desired_goal) -> reward tensor
    ) -> list[dict] | None:
        """
        对 done_mask 中为 True 的 env，生成 HER 重标签样本并清空 buffer。

        返回
        ----
        list of dicts, 每个 dict 包含:
          obs, action, reward, done (均为 (T,) 或 scalar tensor)
        如果没有 env 完成，返回 None。
        """
        her_samples: list[dict] = []

        for i in range(self.num_envs):
            if not done_mask[i].item():
                continue
            ep = self._eps[i]
            if len(ep) < 2:
                self._eps[i] = []
                continue

            # 生成 HER 重标签样本（Future 策略）
            her_samples.extend(self._relabel_episode(ep, reward_fn))
            self._eps[i] = []

        return her_samples if her_samples else None

    def clear_all(self):
        """清空所有 env 的 buffer（训练结束时调用）。"""
        self._eps = [[] for _ in range(self.num_envs)]

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _relabel_episode(
        self,
        ep: list[dict],
        reward_fn,
    ) -> list[dict]:
        """
        Future 策略 HER：
        对 episode 中每个 step t，随机采样 her_k 个 t' > t 的 future achieved_goal
        作为新的 desired_goal，重新计算奖励。
        """
        T = len(ep)
        relabeled: list[dict] = []

        for t in range(T):
            # 随机采样 her_k 个 future 时刻
            future_indices = torch.randint(t, T, (self.her_k,))
            for f_idx in future_indices:
                new_goal = ep[f_idx.item()]["achieved_goal"]  # (goal_dim,)
                new_rew  = reward_fn(
                    ep[t]["achieved_goal"].unsqueeze(0),
                    new_goal.unsqueeze(0),
                ).squeeze(0)

                relabeled.append({
                    "obs":    ep[t]["obs"],
                    "action": ep[t]["action"],
                    "reward": new_rew,
                    "done":   torch.tensor(ep[t]["done"], dtype=torch.bool),
                })

        return relabeled


# ── 独立辅助：goal-conditioned 奖励函数 ──────────────────────────────────────

def her_reward_fn(
    achieved_goal: torch.Tensor,  # (..., 3)
    desired_goal: torch.Tensor,   # (..., 3)
    success_thresh: float = 0.05,
) -> torch.Tensor:
    """
    稠密 HER 奖励：
        r = tanh-shaped(-1 ~ 0)，物体到达目标时为 0，远离时趋近 -1
    保持与原始奖励量纲一致（不影响 PPO value 估计）。
    """
    dist = torch.norm(achieved_goal - desired_goal, dim=-1)
    # 范围 [-1, 0]：到达目标时 = 0，远离时 = -1
    return torch.tanh(dist / success_thresh) - 1.0
