"""Configuration for the Delto direct RL environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass

from .ur_delto_cfg import DELTO_CFG


_OBJECT_PRIM_PATH_EXPR = "/World/envs/env_.*/Object"


@configclass
class RobotCfg:
    """Robot asset, joint layout, initial pose, and joint limits."""

    asset: ArticulationCfg = DELTO_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace()
    arm_joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    hand_joint_names: tuple[str, ...] = (
        "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4",
        "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4",
        "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4",
        "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4",
        "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4",
    )
    arm_start_deg: tuple[float, ...] = (
        -90.0,
        -90.0,
        100.0,
        -10.0,
        90.0,
        180.0,
    )
    hand_start_deg: tuple[float, ...] = (
        30.0, 0.0, 0.0, 0.0, 0.0,
        -50.0, 30.0, 30.0, 30.0, 0.0,
        0.0, 30.0, 30.0, 30.0, 30.0,
        0.0, 30.0, 30.0, 30.0, 30.0,
    )
    arm_lower_limits_deg: tuple[float, ...] = (
        -360.0,
        -360.0,
        -180.0,
        -360.0,
        -360.0,
        -360.0,
    )
    arm_upper_limits_deg: tuple[float, ...] = (
        360.0,
        360.0,
        180.0,
        360.0,
        360.0,
        360.0,
    )
    hand_lower_limits_deg: tuple[float, ...] = (
        -22.0, -20.0, -30.0, -32.0, 0.0,
        -155.0, 0.0, 0.0, 0.0, -15.0,
        -90.0, -90.0, -90.0, -90.0, -90.0,
        -90.0, -90.0, -90.0, -90.0, -90.0,
    )
    hand_upper_limits_deg: tuple[float, ...] = (
        70.0, 31.0, 30.0, 15.0, 60.0,
        0.0, 115.0, 115.0, 110.0, 90.0,
        90.0, 90.0, 90.0, 90.0, 90.0,
        90.0, 90.0, 90.0, 90.0, 90.0,
    )


@configclass
class ObjectCfg:
    """Manipulated object assets and domain-randomization parameters."""

    asset: RigidObjectCfg = RigidObjectCfg(
        prim_path=_OBJECT_PRIM_PATH_EXPR,
        # Objects are spawned individually in setup_scene() so every environment
        # can use a different primitive and color.
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.83, 0.15), rot=(0.7071, 0.0, 0.0, -0.7071)
        ),
    )

    randomize_shape: bool = True
    shape_names: tuple[str, ...] = ("cuboid", "sphere", "cylinder", "capsule")
    randomize_size: bool = True
    cuboid_size: tuple[float, float, float] = (0.07, 0.07, 0.1)
    cuboid_size_min: tuple[float, float, float] = (0.03, 0.03, 0.04)
    cuboid_size_max: tuple[float, float, float] = (0.1, 0.1, 0.12)
    sphere_radius: float = 0.05
    sphere_radius_range: tuple[float, float] = (0.025, 0.06)
    cylinder_radius: float = 0.04
    cylinder_height: float = 0.1
    cylinder_radius_range: tuple[float, float] = (0.02, 0.055)
    cylinder_height_range: tuple[float, float] = (0.03, 0.12)
    capsule_radius: float = 0.035
    capsule_height: float = 0.04
    capsule_radius_range: tuple[float, float] = (0.02, 0.045)
    capsule_height_range: tuple[float, float] = (0.01, 0.06)

    randomize_mass: bool = True
    mass_range: tuple[float, float] = (0.1, 0.4)
    default_mass: float = 0.2

    randomize_color: bool = True
    default_color: tuple[float, float, float] = (0.0, 1.0, 0.0)
    color_min: tuple[float, float, float] = (0.1, 0.1, 0.1)
    color_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    static_friction: float = 2.0
    dynamic_friction: float = 2.0
    restitution: float = 0.0

    randomize_friction: bool = True
    static_friction_range: tuple[float, float] = (0.5, 2.0)
    dynamic_friction_range: tuple[float, float] = (0.4, 1.5)
    restitution_range: tuple[float, float] = (0.0, 0.1)
    friction_num_buckets: int = 32

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.5, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=15.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1), metallic=0.1
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.85, 0.05)),
    )


@configclass
class SensorCfg:
    """Fingertip contact sensors and point-cloud visualization."""

    fingertip_names: tuple[str, ...] = (
        "rl_dg_1_tip",
        "rl_dg_2_tip",
        "rl_dg_3_tip",
        "rl_dg_4_tip",
        "rl_dg_5_tip",
    )
    contact_threshold: float = 1.5
    point_cloud_size: int = 25
    contacts: dict[str, ContactSensorCfg] = {
        name: ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/dg5f/{name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            # ``force_matrix_w`` will contain only fingertip-object contacts.
            filter_prim_paths_expr=[_OBJECT_PRIM_PATH_EXPR],
            track_air_time=False,
        )
        for name in fingertip_names
    }
    point_cloud_marker = RAY_CASTER_MARKER_CFG.replace(
        prim_path="/Visuals/ObservationPointCloud"
    )
    point_cloud_marker.markers["hit"].radius = 0.0025


@configclass
class ActionCfg:
    """Incremental joint-position action parameters."""

    scale: tuple[float, ...] = (0.05,) * 6 + (0.01,) * 20


@configclass
class ObservationCfg:
    """Policy observation history parameters."""

    history_length: int = 5
    flatten_history: bool = True
    asymmetric_critic: bool = True


@configclass
class RewardCfg:
    """Reward parameters."""

    approach_scale: float = 1.0
    approach_distance_scale: float = 0.4
    contact_scale: float = 0.3
    lift_scale: float = 1.0
    force_scale: float = 0.3
    force_threshold: float = 10.0
    action_rate_scale: float = 0.001
    action_scale: float = 0.001
    success_bonus: float = 30.0
    success_height: float = 0.4
    grasp_contact_min: int = 3


@configclass
class ResetCfg:
    """Episode reset randomization parameters."""

    joint_noise_deg: int = 10
    timeout_fraction_range: tuple[float, float] = (0.7, 1.0)
    object_xy_range: float = 0.15
    object_z_range: float = 0.2


@configclass
class TerminationCfg:
    """Object workspace limits [m]."""

    object_position_min: tuple[float, float, float] = (-2.5, -3.1, 0.0)
    object_position_max: tuple[float, float, float] = (2.5, 0.0, 2.0)


@configclass
class CurriculumCfg:
    """Success-driven observation-noise and disturbance curriculum."""

    enabled: bool = True
    success_rate_threshold: float = 0.25
    success_consecutive_steps: int = 10
    level_cooldown_steps: int = 600
    max_level: int = 10

    joint_pos_noise_std_max: float = 0.02
    contact_force_noise_std_max: float = 1.0
    spatial_noise_std_max: float = 0.005

    external_force_max: float = 5.0
    external_force_probability_max: float = 0.25
    external_force_lift_height: float = 0.03


# =======================================================================


@configclass
class DeltoEnvCfg(DirectRLEnvCfg):
    """Top-level configuration for :class:`DeltoEnv`."""

    decimation: int = 2
    episode_length_s: float = 5.0
    action_space: int = 26
    observation_space: int = 121 * 5
    # Derived from the policy observation and privileged-state layout at init.
    state_space: int = 121 * 5 + 30

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=3.0, replicate_physics=True
    )

    robot: RobotCfg = RobotCfg()
    object: ObjectCfg = ObjectCfg()
    sensors: SensorCfg = SensorCfg()
    action: ActionCfg = ActionCfg()
    observation: ObservationCfg = ObservationCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


__all__ = [
    "ActionCfg",
    "CurriculumCfg",
    "DeltoEnvCfg",
    "ObjectCfg",
    "ObservationCfg",
    "ResetCfg",
    "RewardCfg",
    "RobotCfg",
    "SensorCfg",
    "TerminationCfg",
]
