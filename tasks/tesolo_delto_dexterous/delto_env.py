"""Thin DirectRLEnv wrapper for the Delto lifting task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv

from .delto_env_cfg import DeltoEnvCfg
from .utils.delto_task import (
    allocate_buffers,
    apply_actions,
    build_observations,
    compute_observation_dim,
    compute_state_dim,
    compute_rewards,
    compute_terminations,
    process_actions,
    reset_envs,
    setup_scene,
)

__all__ = ["DeltoEnv", "DeltoEnvCfg"]


class DeltoEnv(DirectRLEnv):
    """Direct RL environment for grasping and lifting an object with Delto."""

    cfg: DeltoEnvCfg

    def __init__(
        self, cfg: DeltoEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        cfg.observation_space = compute_observation_dim(cfg)
        cfg.state_space = compute_state_dim(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        allocate_buffers(self)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        process_actions(self, actions)

    def _apply_action(self) -> None:
        apply_actions(self)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    def _get_rewards(self) -> torch.Tensor:
        return compute_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_terminations(self)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        reset_envs(self, env_ids)
