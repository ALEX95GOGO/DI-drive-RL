'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''
from gym.envs.registration import register, registry
from core import SIMULATORS
from .base_drive_env import BaseDriveEnv
from .drive_env_wrapper import DriveEnvWrapper, BenchmarkEnvWrapper

envs = []
env_map = {}

if 'carla' in SIMULATORS:
    from .simple_carla_env import SimpleCarlaEnv
    from .scenario_carla_env import ScenarioCarlaEnv
    from .scenario_carla_env_ori import ScenarioCarlaEnv2
    from .scenario_carla_env_parallel import ScenarioCarlaEnv3
    from .scenario_carla_env_parallel_cam import ScenarioCarlaEnvCam
    from .scenario_carla_env_1cam import ScenarioCarlaEnv_1cam
    from .scenario_carla_env_fixed_action import ScenarioCarlaEnv_fixed
    from .scenario_carla_env_fixed_action_n1 import ScenarioCarlaEnv_fixed_n1
    from .scenario_carla_env_fixed_action_n2 import ScenarioCarlaEnv_fixed_n2
    from .scenario_carla_env_fixed_action_n3 import ScenarioCarlaEnv_fixed_n3
    from .scenario_carla_env_fixed_action_n4 import ScenarioCarlaEnv_fixed_n4
    from .scenario_carla_env_fixed_action_n4 import ScenarioCarlaEnv_fixed_n4
    from .scenario_carla_env_cam_eval import ScenarioCarlaEnvCamEval
    from .scenario_carla_env_HAGRL import ScenarioCarlaEnvHAGRL
    from .scenario_carla_env_leaderboard import ScenarioCarlaEnvLeaderboard

    from .env_sb3_3cam import CarFollowing
    from .env_leftturn_gym_3cam import LeftTurn
    from .scenario_carla_env_steer import ScenarioCarlaEnv4
    env_map.update(
        {
            "SimpleCarla-v1": 'core.envs.simple_carla_env.SimpleCarlaEnv',
            "ScenarioCarla-v1": 'core.envs.scenario_carla_env.ScenarioCarlaEnv'
        }
    )

if 'metadrive' in SIMULATORS:
    from .md_macro_env import MetaDriveMacroEnv
    from .md_traj_env import MetaDriveTrajEnv
    env_map.update(
        {
            "Macro-v1": 'core.envs.md_macro_env:MetaDriveMacroEnv',
            "Traj-v1": 'core.envs.md_traj_env:MetaDriveTrajEnv'
        }
    )

for k, v in env_map.items():
    if k not in registry.env_specs:
        envs.append(k)
        register(id=k, entry_point=v)

if len(envs) > 0:
    print("[ENV] Register environments: {}.".format(envs))
