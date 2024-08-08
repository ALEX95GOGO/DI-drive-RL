import os
import time
import numpy as np
from datetime import datetime
from typing import Any, Dict
from gym import spaces

from .base_drive_env import BaseDriveEnv
from core.simulators import CarlaScenarioSimulator
from core.utils.others.visualizer import Visualizer
from core.utils.simulator_utils.carla_utils import visualize_birdview
import matplotlib.pyplot as plt
from core.policy import AutoPIDPolicy
from core.simulators.srunner.tools.route_parser import RouteParser
import random
from core.policy import AutoPIDPolicy

class ScenarioCarlaEnv3(BaseDriveEnv):
    """
    Carla Scenario Environment with a single hero vehicle. It uses ``CarlaScenarioSimulator`` to load scenario
    configurations and interacts with Carla server to get running status. The Env is initialized with a scenario
    config, which could be a route with scenarios or a single scenario. The observation, sensor settings and visualizer
    are the same with `SimpleCarlaEnv`. The reward is derived based on the scenario criteria in each tick. The criteria
    is also related to success and failure judgement which is used to end an episode.

    When created, it will initialize environment with config and Carla TCP host & port. This method will NOT create
    the simulator instance. It only creates some data structures to store information when running env.

    :Arguments:
        - cfg (Dict): Env config dict.
        - host (str, optional): Carla server IP host. Defaults to 'localhost'.
        - port (int, optional): Carla server IP port. Defaults to 9000.
        - tm_port (Optional[int], optional): Carla Traffic Manager port. Defaults to None.

    :Interfaces: reset, step, close, is_success, is_failure, render, seed

    :Properties:
        - hero_player (carla.Actor): Hero vehicle in simulator.
    """

    #action_space = spaces.Dict({})
    #observation_space = spaces.Dict({})
    HEIGHT = 100; WIDTH=300;
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float16)
        
    # Example for using image as input (you can change these dimensions according to your needs):
    #observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH,3), dtype=np.uint8)
    observation_space = spaces.Box(low=0, high=255, shape=(3,HEIGHT, WIDTH), dtype=np.uint8)


    reward_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(1, ))
    config = dict(
        simulator=dict(),
        # reward value if success
        success_reward=10,
        # whether open visualize
        visualize=None,
        # outputs of scenario conclusion
        outputs=[],
        output_dir='',
    )

    def __init__(
            self,
            cfg: Dict,
            cfg_policy: Dict,
            #setupconfig: Dict,
            routes: str,
            scenario_file: str,
            single_route: int = None,
            host: str = 'localhost',
            port: int = None,
            tm_port: int = None,
            **kwargs,
    ) -> None:
        """
        Initialize environment with config and Carla TCP host & port.
        """
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
        self.cfg_policy = cfg_policy
        self._simulator_cfg = self._cfg.simulator
        self._carla_host = host
        self._carla_port = port
        self._carla_tm_port = tm_port

        self._use_local_carla = False
        if self._carla_host != 'localhost':
            self._use_local_carla = True
        self._simulator = None

        self._output_dir = self._cfg.output_dir
        self._outputs = self._cfg.outputs

        self._success_reward = self._cfg.success_reward
        self._is_success = False
        self._is_failure = False
        self._collided = False
        self.collision = False

        self._tick = 0
        self._timeout = float('inf')
        self._launched_simulator = False
        self._config = None

        self._visualize_cfg = self._cfg.visualize
        self._simulator_databuffer = dict()
        self._visualizer = None
        #self.setup_config = setupconfig
        self.setup_config = []
        self.setup_config += RouteParser.parse_routes_file(routes, scenario_file, single_route)
        self.auto_policy = AutoPIDPolicy(self.cfg_policy).eval_mode
        self.scenario = int((port-9010)//2+20)

    def _init_carla_simulator(self) -> None:
        if not self._use_local_carla:
            print("------ Run Carla on Port: %d, GPU: %d ------" % (self._carla_port, 0))
            #self.carla_process = subprocess.Popen()
            self._simulator = CarlaScenarioSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port
            )
        else:
            print('------ Using Remote Carla @ {}:{} ------'.format(self._carla_host, self._carla_port))
            self._simulator = CarlaScenarioSimulator(
                cfg=self._simulator_cfg,
                client=None,
                host=self._carla_host,
                port=self._carla_port,
                tm_port=self._carla_tm_port
            )
        self._launched_simulator = True

    #def reset(self, config: Any) -> Dict:
    def reset(self):
        """
        Reset environment to start a new episode, with provided reset params. If there is no simulator, this method will
        create a new simulator instance. The reset param is sent to simulator's ``init`` method to reset simulator,
        then reset all statues recording running states, and create a visualizer if needed. It returns the first frame
        observation.

        :Arguments:
            - config (Any): Configuration instance of the scenario

        :Returns:
            Dict: The initial observation.
        """
        #import pdb; pdb.set_trace()
        if not self._launched_simulator:
            self._init_carla_simulator()
        #random.seed(42)  # You can use any number as the seed
        #random_integer = random.randint(20, 24)
        #random_integer = (self._carla_port-9000)//2
        #config = self.setup_config[random_integer]
        #self._config = self.setup_config[random_integer]
        i = self.scenario
        config = self.setup_config[i]
        self._config = self.setup_config[i]
        self._simulator.init(self._config)
        self.last_acceleration = [0,0,0]
        if self._visualize_cfg is not None:
            if self._visualizer is not None:
                self._visualizer.done()
            else:
                self._visualizer = Visualizer(self._visualize_cfg)

            if 'Route' in config.name:
                config_name = os.path.splitext(os.path.split(config.scenario_file)[-1])[0]
            else:
                config_name = config.name
            vis_name = "{}_{}".format(config_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

            self._visualizer.init(vis_name)

        self._simulator_databuffer.clear()
        self._collided = False
        self._criteria_last_value = dict()
        self._is_success = False
        self._is_failure = False
        self.collision = False
        self.outside_route = False
        self._reward = 0
        self._tick = 0
        self._timeout = self._simulator.end_timeout
        self.auto_policy.reset([0])
        segmentation1 = self.get_observations()['segmentation1']
        segmentation2 = self.get_observations()['segmentation2']
        segmentation3 = self.get_observations()['segmentation3']
        segmentations = np.concatenate([segmentation3,segmentation1,segmentation2], axis=1)
        segmentations = segmentations.transpose(2,0,1)
        print(segmentations.shape)
        return segmentations

    def step(self, action):
        """
        Run one time step of environment, get observation from simulator and calculate reward. The environment will
        be set to 'done' only at success or failure. And if so, all visualizers will end. Its interfaces follow
        the standard definition of ``gym.Env``.

        :Arguments:
            - action (Dict): Action provided by policy.

        :Returns:
            Tuple[Any, float, bool, Dict]: A tuple contains observation, reward, done and information.
        """
        #print(action)
        #auto_policy = AutoPIDPolicy(cfg.policy).eval_mode
        #steer_ori = action['steer']
        
        
        obs = self.get_observations()

        actions = self.auto_policy.forward({0: obs})
        #print(actions)

        if action[0] > 0:
            throttle = action[0]
            brake = 0
        else:
            brake = -action[0]
            throttle = 0
        action = {
            'throttle': throttle,
            'brake': brake,
            'steer': actions[0]['action']['steer']
        }
        
        #action = actions[0]['action']
        if action is not None:
            self._simulator.apply_control(action)
            self._simulator_databuffer.update({'action': action})
        self._simulator.run_step()
        self._tick += 1

        #import pdb; pdb.set_trace()
        self._collided = self._simulator.collided

        res = self._simulator.scenario_manager.get_scenario_status()
        if res == 'SUCCESS':
            self._is_success = True
        elif res in ['FAILURE', 'INVALID']:
            self._is_failure = True

        self._reward, reward_info = self.compute_reward()        
                

        segmentation1 = self.get_observations()['segmentation1']
        segmentation2 = self.get_observations()['segmentation2']
        segmentation3 = self.get_observations()['segmentation3']
        segmentations = np.concatenate([segmentation3,segmentation1,segmentation2], axis=1)
        
        done = self.is_success() or self.is_failure() or self.collision or self.outside_route
        if done:
            self._simulator.end_scenario()
            self._conclude_scenario(self._config)
            if self._visualizer is not None:
                self._visualizer.done()

        info = self._simulator.get_information()
        info.update(reward_info)
        info.update({
            'failure': self.is_failure(),
            'success': self.is_success(),
        })
        #print(self._reward)
        #plt.imshow(obs['rgb'])
        #plt.show(block=False)
        #plt.pause(0.01)
        #self.render()

        state_space = segmentations[:,:,0]
        state = np.repeat(np.expand_dims(state_space,2), 3, axis=2)
        state_ = state.copy()
        state_1 = state_[:,:,2].copy()
        state_[:,:,0] = state_[:,:,1].copy()
        state_[:,:,1] = state_1
        state_[:,:,2] = state_space.copy()
        state_ = state_.transpose(2,0,1)
        return state_, self._reward, done, info

    def close(self):
        """
        Delete simulator & visualizer instances and close environment.
        """
        if self._launched_simulator:
            self._simulator.clean_up()
            self._simulator._set_sync_mode(False)
            del self._simulator
            self._launched_simulator = False
        if self._visualizer is not None:
            self._visualizer.done()

    def is_success(self) -> bool:
        """
        Check if the task succeed. It only happens when behavior tree ends successfully.

        :Returns:
            bool: Whether success.
        """
        res = self._simulator.scenario_manager.get_scenario_status()
        if res == 'SUCCESS':
            return True
        return False

    def is_failure(self) -> bool:
        """
        Check if the task fails. It may happen when behavior tree ends unsuccessfully or some criteria trigger.

        :Returns:
            bool: Whether failure.
        """
        res = self._simulator.scenario_manager.get_scenario_status()
        if res in ['FAILURE', 'INVALID']:
            return True
        return False

    def get_observations(self):
        """
        Get observations from simulator. The sensor data, navigation, state and information in simulator
        are used, while not all these are added into observation dict.

        :Returns:
            Dict: Observation dict.
        """
        obs = dict()
        state = self._simulator.get_state()
        sensor_data = self._simulator.get_sensor_data()
        navigation = self._simulator.get_navigation()
        information = self._simulator.get_information()

        self._simulator_databuffer['state'] = state
        self._simulator_databuffer['navigation'] = navigation
        self._simulator_databuffer['information'] = information

        obs.update(sensor_data)
        obs.update(
            {
                'tick': information['tick'],
                'timestamp': np.float32(information['timestamp']),
                'agent_state': navigation['agent_state'],
                'node': navigation['node'],
                'node_forward': navigation['node_forward'],
                'target': np.float32(navigation['target']),
                'target_forward': np.float32(navigation['target_forward']),
                'command': navigation['command'],
                'speed': np.float32(state['speed']),
                'speed_limit': np.float32(navigation['speed_limit']),
                'location': np.float32(state['location']),
                'forward_vector': np.float32(state['forward_vector']),
                'acceleration': np.float32(state['acceleration']),
                'velocity': np.float32(state['velocity']),
                'angular_velocity': np.float32(state['angular_velocity']),
                'rotation': np.float32(state['rotation']),
                'is_junction': np.float32(state['is_junction']),
                'tl_state': state['tl_state'],
                'tl_dis': np.float32(state['tl_dis']),
                'waypoint_list': navigation['waypoint_list'],
                'direction_list': navigation['direction_list'],
            }
        )

        if self._visualizer is not None:
            if self._visualize_cfg.type not in sensor_data:
                raise ValueError("visualize type {} not in sensor data!".format(self._visualize_cfg.type))
            self._render_buffer = sensor_data[self._visualize_cfg.type].copy()
            if self._visualize_cfg.type == 'birdview':
                self._render_buffer = visualize_birdview(self._render_buffer)
        return obs

    def compute_reward(self):
        """
        Compute reward for current frame, and return details in a dict. In short, it contains goal reward,
        route following reward calculated by criteria in current and last frame, and failure reward by checking criteria
        in each frame.

        :Returns:
            Tuple[float, Dict]: Total reward value and detail for each value.
        """
        goal_reward = 0
        if self._is_success:
            goal_reward += self._success_reward

        elif self._is_failure:
            goal_reward -= self._success_reward
        
        criteria_dict = self._simulator.get_criteria()

        failure_reward = 0
        complete_reward = 0
        
        speed = self._simulator_databuffer['state']['speed']
        acceleration = self._simulator_databuffer['state']['acceleration']
        jerk = ((acceleration[0]-self.last_acceleration[0])**2+(acceleration[1]-self.last_acceleration[1])**2+(acceleration[2]-self.last_acceleration[2])**2)**0.5
        self.last_acceleration = acceleration
        #print(speed)
        
        #print(jerk)
        speed_reward = 0
        if speed < 0.2:
            speed_reward = -0.2
        elif speed > 25:
            speed_reward = 25-speed
        elif speed <12.5:
            speed_reward = speed/25
        #else:
        #    speed_reward = speed/25
            
        for info, value in criteria_dict.items():
            if value[0] == 'FAILURE':
                if info in self._criteria_last_value and value[1] != self._criteria_last_value[info][1]:
                    if 'Collision' in info:
                        failure_reward -= 100
                        self.collision = True
                    elif 'RunningRedLight' in info:
                        failure_reward -= 10
                    elif 'OutsideRouteLanes' in info:
                        failure_reward -= 10
                        self.outside_route = True
            if 'RouteCompletion' in info and info in self._criteria_last_value:
                complete_reward = 10 / 100 * (value[1] - self._criteria_last_value[info][1])
            #    #complete_reward = self._finish_reward / 100 * (value[1] - self._criteria_last_value[info][1])
            self._criteria_last_value[info] = value

        reward_info = dict()
        reward_info['goal_reward'] = goal_reward
        reward_info['complete_reward'] = complete_reward
        reward_info['failure_reward'] = failure_reward
        #reward_info['route_completion']=criteria_dict['RouteCompletion']

        #total_reward = goal_reward + failure_reward + complete_reward + speed_reward
        total_reward = goal_reward + failure_reward + complete_reward
        #total_reward = failure_reward + complete_reward + speed_reward - 0.1*jerk

        return total_reward, reward_info

    def render(self):
        """
        Render a runtime visualization on screen, save a gif video file according to visualizer config.
        The main canvas is from a specific sensor data. It only works when 'visualize' is set in config dict.
        """
        #import pdb; pdb.set_trace()
        if self._visualizer is None:
            return

        render_info = {
            'collided': self._collided,
            'reward': self._reward,
            'tick': self._tick,
            'end_timeout': self._simulator.end_timeout,
            'end_distance': self._simulator.end_distance,
            'total_distance': self._simulator.total_distance,
        }
        render_info.update(self._simulator_databuffer['state'])
        render_info.update(self._simulator_databuffer['navigation'])
        render_info.update(self._simulator_databuffer['information'])
        render_info.update(self._simulator_databuffer['action'])

        self._visualizer.paint(self._render_buffer, render_info)
        self._visualizer.run_visualize()

    def seed(self, seed: int) -> None:
        """
        Set random seed for environment.

        :Arguments:
            - seed (int): Random seed value.
        """
        print('[ENV] Setting seed:', seed)
        np.random.seed(seed)

    def __repr__(self) -> str:
        return "ScenarioCarlaEnv with host %s, port %s." % (self._carla_host, self._carla_port)

    def _conclude_scenario(self, config: Any) -> None:
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        filename = None
        config_name = config.name

        if self._output_dir != '':
            os.makedirs(self._output_dir, exist_ok=True)
            config_name = os.path.join(self._output_dir, config_name)
        if 'junit' in self._outputs:
            junit_filename = config_name + '_' + current_time + ".xml"
        if 'file' in self._outputs:
            filename = config_name + '_' + current_time + "txt"

        if self._simulator.scenario_manager.analyze_scenario(True, filename, junit_filename):
            self._is_failure = True
        else:
            self._is_success = True

    @property
    def hero_player(self):
        return self._simulator.hero_player
