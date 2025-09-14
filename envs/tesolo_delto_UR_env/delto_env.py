
from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor

# from isaaclab.utils.math import quat_to_rot_mats
from isaaclab.utils.math import quat_apply

##
# Configuration
##

from ur_delto_cfg import DELTO_CFG


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

@configclass
class DeltoEnvCfg(DirectRLEnvCfg):

    # ======================================================================= env params
    decimation = 2
    episode_length_s = 5.0
    action_space = 20
    observation_space = 157  # (full)
    state_space = 0
    action_scale = 1
    asymmetric_obs = False
    obs_type = "full"
    num_env = 1

    # ======================================================================= simulation

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # ======================================================================= robot

    robot_cfg: ArticulationCfg = DELTO_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace()
    # robot_cfg: ArticulationCfg = DELTO_CFG

    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    hand_joint_names = [
        "rj_dg_1_1",
        "rj_dg_1_2",
        "rj_dg_1_3",
        "rj_dg_1_4",
        "rj_dg_2_1",
        "rj_dg_2_2",
        "rj_dg_2_3",
        "rj_dg_2_4",
        "rj_dg_3_1",
        "rj_dg_3_2",
        "rj_dg_3_3",
        "rj_dg_3_4",
        "rj_dg_4_1",
        "rj_dg_4_2",
        "rj_dg_4_3",
        "rj_dg_4_4",
        "rj_dg_5_1",
        "rj_dg_5_2",
        "rj_dg_5_3",
        "rj_dg_5_4"
    ]

    start_position = [-90.0, -75.0, 120.0, -50.0, 90.0, -90.0]
    lift_delta_deg = [-0.0, -20.0, 0.0, -20.0, 0.0, 0.0]

    # ======================================================================= sensors

    ft_names = [
        "rl_dg_1_4",
        "rl_dg_2_4",
        "rl_dg_3_4",
        "rl_dg_4_4",
        "rl_dg_5_4",
    ]

    # fingers_names = [
    #     "rl_dg_1_4/rl_dg_1_tip",
    #     "rl_dg_2_4/rl_dg_2_tip",
    #     "rl_dg_3_4/rl_dg_3_tip",
    #     "rl_dg_4_4/rl_dg_4_tip",
    #     "rl_dg_5_4/rl_dg_5_tip",
    # ]

    contact_sensors = {}
    for name in ft_names:
        contact_sensors[name] = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/dg5f_my/{name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            # filter_prim_paths_expr=["/World/envs/env_.*/Cube"],
            track_air_time=False,
        )

    # ======================================================================= objects

    # in-manipulator object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",  # один куб на среду
        spawn=sim_utils.CylinderCfg(  # используем Box вместо Cone
            radius=0.03,  # кубик 10 см
            height=0.15,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # красный
                metallic=0.1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.24, -0.80, 0.1),
            # pos=(0.22, -1.0, 0.1),
        ),
    )
    
    table_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",  # один куб на среду
            spawn=sim_utils.CuboidCfg(  # используем Box вместо Cone
                size=(0.5, 0.5, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.1),
                    metallic=0.1,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.30, -0.85, 0.05),
            ),
        )

    # ======================================================================= scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_env, 
        env_spacing= 3.0, 
        replicate_physics=True
        )
    
    # === двухфазная логика
    grasp_contact_min = 3            # минимум активных пальцев
    grasp_fc_angle_deg = 110.0       # «распора» (force closure) — угол
    grasp_hold_steps = 20            # сколько тактов подряд держать хват
    grasp_max_dist = 0.07            # м: COM объекта близко к центру хвата

    lift_steps = 120                 # длительность подъёма (шагов симуляции)
    lift_delta_deg = [-0.0, -20.0, 0.0, -20.0, 0.0, 0.0]  # дельта к [shoulder_pan, shoulder_lift, ...]
    success_height = 0.12            # м: на столько поднять выше поверхности стола
    slip_grace = 10                  # допускаем кратковременную потерю контакта, шагов

    tilt_fail_deg = 55.0 
    
    #  ======================================================================= rewards

   
# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

class DeltoEnv(DirectRLEnv):
    cfg: DeltoEnvCfg

    def __init__(self, cfg: DeltoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.scene.num_envs

        self.env_origins = self.scene.env_origins

        self.hand = self.scene.articulations["hand"]

        self.hand_joint_ids, _ = self.hand.find_joints(cfg.hand_joint_names)
        self.arm_joint_ids, _ = self.hand.find_joints(cfg.arm_joint_names)

        # self.ft_ids = [self.hand.body_names.index(n) for n in self.cfg.ft_names]
        
        self.object = self.scene.rigid_objects["object"]
        self.table = self.scene.rigid_objects["table"]

        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel   

        # линейная траектория для сустава руки во время подъема
        self.arm_start = torch.zeros(N, len(self.arm_joint_ids), device=self.device)
        self.arm_goal  = torch.zeros_like(self.arm_start)

        with torch.no_grad():
            self.arm_start= self.joint_pos[:, self.arm_joint_ids]
            delta = torch.tensor(self.cfg.lift_delta_deg, device=self.device) * torch.pi/180.0
            self.arm_goal = self.arm_start + delta


# =====================================================================================================

    def _setup_scene(self):
        # prim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        self.hand = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["hand"] = self.hand

        self.object = RigidObject(self.cfg.object_cfg)
        self.scene.rigid_objects["object"] = self.object

        self.table = RigidObject(self.cfg.table_cfg)
        self.scene.rigid_objects["table"] = self.table

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # self.force_thresh = 1.0  # Н
        # self.ema = 0.9               # сглаживание флагов/сил, чтобы не мигало

        # print(self.cfg.contact_sensors)

        for name in self.cfg.ft_names:
            self.scene.sensors[name] = ContactSensor(self.cfg.contact_sensors[name])    # доступ: self.scene.sensors["robot0_ffdistal"]

# =====================================================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

        # self._update_and_apply_disturbances()

# =====================================================================================================

    def _apply_action(self):
        # --- контакт/геометрия для фазовой машины
        self.hand.set_joint_position_target(self.actions, joint_ids=self.hand_joint_ids)

# =====================================================================================================

    def _get_observations(self):

        object_pos = self.object.data.body_link_pose_w[:,0,:3]
        object_pos -= self.env_origins
        object_quat = self.object.data.body_link_pose_w[:,0,3:]

        Fw, flags = self._read_contacts(threshold=1.0, frame="w")  # силы и флаги в world frame
        obs = {
            "joints_pos": self.joint_pos,
            "joints_vel": self.joint_vel,
            "contact_forces_w": Fw,   # (N,5,3)
            "contact_flags": flags,   # (N,5) True/False
            "object_pos": object_pos,
            "object_quat": object_quat
        }
        return {"state": obs}
    
# =====================================================================================================

    def _get_rewards(self) -> torch.Tensor:
        
        return torch.ones(self.num_envs, device=self.device)
    
# =====================================================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        return torch.zeros(self.num_envs, dtype=torch.bool), self.episode_length_buf >= self.max_episode_length
    
# =====================================================================================================

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # joint_pos = torch.zeros((len(env_ids), 7), device=self.device)
        # joint_vel = torch.zeros((len(env_ids), 7), device=self.device)

        joint_pos = torch.zeros((len(env_ids), 26), device=self.device)
        joint_vel = torch.zeros((len(env_ids), 26), device=self.device)

        joint_pos[:, :6] = torch.tensor(self.cfg.start_position, device=self.device)*np.pi/180
 
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        with torch.no_grad():
            self.arm_start[env_ids] = self.joint_pos[env_ids][:, self.arm_joint_ids]
            delta = torch.tensor(self.cfg.lift_delta_deg, device=self.device) * torch.pi/180.0
            self.arm_goal[env_ids]  = self.arm_start[env_ids] + delta

        self.hand.set_joint_position_target(self.arm_start[env_ids], joint_ids=self.arm_joint_ids, env_ids=env_ids)

    
# ===========================================================================
# ===========================================================================
# ===========================================================================

    def _read_contact_forces(self, frame: str = "w") -> torch.Tensor:
        """
        Возвращает тензор сил контакта размера (num_envs, num_fingers, 3).
        frame: 'w' — в мировых координатах, 'b' — в body frame (локально для линка).
        """
        assert frame in ("w", "b")
        forces_per_finger = []
        attr = f"net_forces_{frame}"

        for name in self.cfg.ft_names:
            s = self.scene.sensors[name]
            f = getattr(s.data, attr)  # shape: (N, 3) или (N, H, 3) при history_length>1
            if f.ndim == 3:
                f = f[:, -1, :]  # берём последний сэмпл истории
            forces_per_finger.append(f)

        return torch.stack(forces_per_finger, dim=1)  # (N, 5, 3)
    

    def _read_contact_flags(
        self,
        threshold: float = 1.0,
        frame: str = "w",
    ) -> torch.Tensor:
        """
        Булевы флаги контакта (norm(force) > threshold) размера (num_envs, num_fingers).
        threshold — порог в Ньютонах.
        """
        F = self._read_contact_forces(frame=frame)          # (N, 5, 3)
        norms = torch.linalg.norm(F, dim=-1)                # (N, 5)
        return norms > threshold
    

    def _read_contacts(
        self,
        threshold: float = 1.0,
        frame: str = "w",
    ):
        """
        Удобный комбинированный вызов: возвращает (forces, flags)
        forces: (N, 5, 3), flags: (N, 5)
        """
        F = self._read_contact_forces(frame=frame)
        flags = torch.linalg.norm(F, dim=-1) > threshold
        return F, flags