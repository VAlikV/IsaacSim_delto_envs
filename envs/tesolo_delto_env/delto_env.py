
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

from isaaclab.sim.schemas import modify_articulation_root_properties

# from isaaclab.utils.math import quat_to_rot_mats
from isaaclab.utils.math import quat_apply


##
# Configuration
##

# isaac/Props/YCB/Axis_Aligned/035_power_drill.usd

DELTO_CFG = ArticulationCfg(
    # prim_path = "/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"robots/dg5f_right/dg5f_my.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            # max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    # spawn = sim_utils.UsdFileCfg(
    #     usd_path=f"robots/dg5f_right/dg5f_my.usd",
    #     activate_contact_sensors=True,   # <-- Включаем отчетность о контактах
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         retain_accelerations=False,
    #         # можно задать ограничение скоростей:
    #         max_linear_velocity=100.0,
    #         max_angular_velocity=1000.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=True,
    #         solver_position_iteration_count=16,   # больше итераций решателя
    #         solver_velocity_iteration_count=0,
    #         # stabilization_threshold=1e-4,
    #         # sleep_threshold=1e-3,
    #     ),
    #     collision_props=sim_utils.CollisionPropertiesCfg(
    #         contact_offset=0.002,        # уменьшить генерацию ранних контактных точек
    #         rest_offset=0.0,             # убрать «воздушную» прослойку
    #     ),
    #     joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    # ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"rj_dg_.*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rj_dg_.*"],
            effort_limit={
                "rj_dg_.*":30
            },
            stiffness={
                "rj_dg_1_1": 0.8294,
                "rj_dg_1_2": 0.62859, 
                "rj_dg_1_3": 0.45859, 
                "rj_dg_1_4": 0.16977, 

                "rj_dg_2_1": 2.43808, 
                "rj_dg_2_2": 0.82762, 
                "rj_dg_2_3": 0.41117, 
                "rj_dg_2_4": 0.09574, 

                "rj_dg_3_1": 2.45355, 
                "rj_dg_3_2": 0.82643,
                "rj_dg_3_3": 0.41118, 
                "rj_dg_3_4": 0.09575, 

                "rj_dg_4_1": 2.3093,
                "rj_dg_4_2": 0.82729,
                "rj_dg_4_3": 0.41094,
                "rj_dg_4_4": 0.09573,

                "rj_dg_5_1": 1.73423,
                "rj_dg_5_2": 1.10747,
                "rj_dg_5_3": 0.45515,
                "rj_dg_5_4": 0.16936
            },
            damping={
         
                "rj_dg_1_1": 0.05314332,
                "rj_dg_1_2": 0.02990215, 
                "rj_dg_1_3": 0.01657904, 
                "rj_dg_1_4": 0.00270917, 

                "rj_dg_2_1": 0.26783944, 
                "rj_dg_2_2": 0.0451749, 
                "rj_dg_2_3": 0.0120614, 
                "rj_dg_2_4": 0.00088871, 

                "rj_dg_3_1": 0.2703927,
                "rj_dg_3_2": 0.04507756,
                "rj_dg_3_3": 0.0125066, 
                "rj_dg_3_4": 0.00088885, 

                "rj_dg_4_1": 0.24690116,
                "rj_dg_4_2": 0.04514794,
                "rj_dg_4_3": 0.01249565,
                "rj_dg_4_4": 0.00088857,

                "rj_dg_5_1": 0.16547778,
                "rj_dg_5_2": 0.0699276,
                "rj_dg_5_3": 0.01639284,
                "rj_dg_5_4": 0.0026993,

                # "rj_dg_1_1": 0.00033,
                # "rj_dg_1_2": 0.00025, 
                # "rj_dg_1_3": 0.00018, 
                # "rj_dg_1_4": 0.00007, 

                # "rj_dg_2_1": 0.00093, 
                # "rj_dg_2_2": 0.00033, 
                # "rj_dg_2_3": 0.00016, 
                # "rj_dg_2_4": 0.00004, 

                # "rj_dg_3_1": 0.00098, 
                # "rj_dg_3_2": 0.00033,
                # "rj_dg_3_3": 0.00016, 
                # "rj_dg_3_4": 0.00004, 

                # "rj_dg_4_1": 0.00092,
                # "rj_dg_4_2": 0.00033,
                # "rj_dg_4_3": 0.00016,
                # "rj_dg_4_4": 0.00004,

                # "rj_dg_5_1": 0.00069,
                # "rj_dg_5_2": 0.00044,
                # "rj_dg_5_3": 0.00018,
                # "rj_dg_5_4": 0.00007
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

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

    # ======================================================================= sensors

    # ft_names = [
    #     "robot0_ffdistal",
    #     "robot0_mfdistal",
    #     "robot0_rfdistal",
    #     "robot0_lfdistal",
    #     "robot0_thdistal",
    # ]

    # contact_sensors = {}
    # for name in ft_names:
    #     contact_sensors[name] = ContactSensorCfg(
    #         prim_path=f"/World/envs/env_.*/Robot/{name}",
    #         update_period=0.0,
    #         history_length=1,
    #         debug_vis=False,
    #         # filter_prim_paths_expr=["/World/envs/env_.*/Cube"],
    #         track_air_time=False,
    #     )

    # ======================================================================= objects

    # in-manipulator object
    # object_cfg: RigidObjectCfg = RigidObjectCfg(
    #         prim_path="/World/envs/env_.*/Object",
    #         spawn=sim_utils.UsdFileCfg(
    #             # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/035_power_drill.usd",
    #             # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mugs/SM_Mug_D1.usd",
    #             usd_path="robots/objects/Mug.usd",
    #             rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                 disable_gravity=False,
    #                 retain_accelerations=True,
    #                 max_depenetration_velocity=1000.0,
    #             ),
    #         ),
    #         init_state=RigidObjectCfg.InitialStateCfg(
    #             pos=(0.32, -1.2, 0.1),
    #             # pos=(0.22, -1.0, 0.1),
    #         ),
    #     )
    
    # table_cfg: RigidObjectCfg = RigidObjectCfg(
    #         prim_path="/World/envs/env_.*/Table",  # один куб на среду
    #         spawn=sim_utils.CuboidCfg(  # используем Box вместо Cone
    #             size=(0.5, 0.5, 0.1),
    #             rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                 disable_gravity=True,
    #                 kinematic_enabled=True
    #             ),
    #             mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
    #             collision_props=sim_utils.CollisionPropertiesCfg(),
    #             visual_material=sim_utils.PreviewSurfaceCfg(
    #                 diffuse_color=(0.1, 0.1, 0.1),
    #                 metallic=0.1,
    #             ),
    #         ),
    #         init_state=RigidObjectCfg.InitialStateCfg(
    #             pos=(0.30, -1.0, 0.05),
    #         ),
    #     )

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

        self.env_origins = self.scene.env_origins

        self.hand = self.scene.articulations["hand"]
        self.hand_joint_ids, _ = self.hand.find_joints(cfg.hand_joint_names)
        # self.arm_joint_ids, _ = self.hand.find_joints(cfg.arm_joint_names)
        # self.ft_ids = [self.hand.body_names.index(n) for n in self.cfg.ft_names]
        
        # self.object = self.scene.rigid_objects["object"]
        # self.table = self.scene.rigid_objects["table"]

        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel   


# =====================================================================================================

    def _setup_scene(self):
        # prim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        self.hand = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["hand"] = self.hand

        # self.object = RigidObject(self.cfg.object_cfg)
        # self.scene.rigid_objects["object"] = self.object

        # self.table = RigidObject(self.cfg.table_cfg)
        # self.scene.rigid_objects["table"] = self.table

        # modify_articulation_root_properties(
        #     "/World/envs/env_0/Robot",   # ваш корневой путь спавна
        #     sim_utils.ArticulationRootPropertiesCfg()
        # )

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # self.force_thresh = 1.0  # Н
        # self.ema = 0.9               # сглаживание флагов/сил, чтобы не мигало

        # for name in self.cfg.ft_names:
        #     self.scene.sensors[name] = ContactSensor(self.cfg.contact_sensors[name])    # доступ: self.scene.sensors["robot0_ffdistal"]

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

        obs = {
            "joints_pos": self.joint_pos,
            "joints_vel": self.joint_vel,
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

        joint_pos = torch.zeros((len(env_ids), 20), device=self.device)
        joint_vel = torch.zeros((len(env_ids), 20), device=self.device)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)