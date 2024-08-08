import os
import argparse
from argparse import RawTextHelpFormatter
from easydict import EasyDict

from core.envs import DriveEnvWrapper, ScenarioCarlaEnv2
from core.policy import AutoPIDPolicy
from ding.utils import set_pkg_seed
from core.simulators.srunner.tools.route_parser import RouteParser
from core.simulators.srunner.tools.scenario_parser import ScenarioConfigurationParser

import argparse
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta
from collections import deque
import numpy as np
import pytorch_lightning as pl
#from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.plugins import DDPPlugin

from TCP.model import TCP
#from TCP.data import CARLA_Data
from TCP.config import GlobalConfig
import torchvision.transforms as T

casezoo_config = dict(
    env=dict(
        simulator=dict(
            planner=dict(type='behavior', ),
            n_vehicles=20,
            #n_pedestrians=25,
            disable_two_wheels=True,
            obs=(
                #dict(
                #    name='rgb',
                #    type='rgb',
                #    size=[400, 400],
                #    position=[-5.5, 0, 2.8],
                #    rotation=[-15, 0, 0],
                #),
                # dict(
                #    name='rgb',
                #    type='rgb',
                #    size=[640, 360],
                #    position=[2.5, 0, 1],
                #    rotation=[0, 30, 0],
                #),
                
                dict(
                    name='rgb',
                    type='rgb',
                    size=[900, 256],
                    position=[2.5, 0, 1],
                    rotation=[0, 0, 0],
                    fov=100
                ),
                dict(
                    name='birdview',
                    type='bev',
                    size=[320, 320],
                    pixels_per_meter=6,
                ),
            ),
            waypoint_num=50,
            #debug=True,
        ),
        #no_rendering=True,
        visualize=dict(
            type='rgb',
            outputs=['show']
        ),
    ),
    policy=dict(target_speed=40, ),
)

main_config = EasyDict(casezoo_config)


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
    

def control_pid(waypoints, velocity, target):
    ''' Predicts vehicle control with a PID controller.
    Args:
        waypoints (tensor): output of self.plan()
        velocity (tensor): speedometer input
    '''
    # Controller
    turn_KP = 0.75
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size
    
    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size
 
    turn_controller = PIDController(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
    speed_controller = PIDController(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
    assert(waypoints.size(0)==1)
    waypoints = waypoints[0].data.cpu().numpy()
    target = target.squeeze().data.cpu().numpy()

    # flip y (forward is negative in our waypoints)
    waypoints[:,1] *= -1
    target[1] *= -1

    # iterate over vectors between predicted waypoints
    num_pairs = len(waypoints) - 1
    best_norm = 1e5
    desired_speed = 0
    aim = waypoints[0]
    aim_dist = 4.0 # distance to search around for aim point
    for i in range(num_pairs):
        # magnitude of vectors, used for speed
        desired_speed += np.linalg.norm(
                waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

        # norm of vector midpoints, used for steering
        norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
        if abs(aim_dist-best_norm) > abs(aim_dist-norm):
            aim = waypoints[i]
            best_norm = norm

    aim_last = waypoints[-1] - waypoints[-2]

    angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
    angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
    angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

    # choice of point to aim for steering, removing outlier predictions
    # use target point if it has a smaller angle or if error is large
    # predicted point otherwise
    # (reduces noise in eg. straight roads, helps with sudden turn commands)
    use_target_to_aim = np.abs(angle_target) < np.abs(angle)

    angle_thresh = 0.3 # outlier control detection angle
    #angle_thresh = 1 # outlier control detection angle
    dist_thresh = 10 # target point y-distance for outlier filtering
    use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > angle_thresh and target[1] < dist_thresh)
    
    #use_target_to_aim = False
    if use_target_to_aim:
        angle_final = angle_target
        #print(angle_final)
    else:
        angle_final = angle
    
    angle_final = angle_target
    #angle_final = angle_target
    #print(angle_final)
    #if angle_final <-0.3:
    #    angle_final=angle_final+0.4
    #if np.abs(angle_target) > np.abs(angle):
    #    angle_final = angle
    #    print(angle)

    steer = turn_controller.step(angle_final)
    steer = np.clip(steer, -1.0, 1.0)

    speed = velocity[0].data.cpu().numpy()
    
    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller
 
    brake = desired_speed < brake_speed or (speed / desired_speed) > brake_ratio

    delta = np.clip(desired_speed - speed, 0.0, clip_delta)
    throttle = speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, max_throttle)
    throttle = throttle if not brake else 0.0

    metadata = {
        'speed': float(speed.astype(np.float64)),
        'steer': float(steer),
        'throttle': float(throttle),
        'brake': float(brake),
        'wp_4': tuple(waypoints[3].astype(np.float64)),
        'wp_3': tuple(waypoints[2].astype(np.float64)),
        'wp_2': tuple(waypoints[1].astype(np.float64)),
        'wp_1': tuple(waypoints[0].astype(np.float64)),
        'aim': tuple(aim.astype(np.float64)),
        'target': tuple(target.astype(np.float64)),
        'desired_speed': float(desired_speed.astype(np.float64)),
        'angle': float(angle.astype(np.float64)),
        'angle_last': float(angle_last.astype(np.float64)),
        'angle_target': float(angle_target.astype(np.float64)),
        'angle_final': float(angle_final.astype(np.float64)),
        'delta': float(delta.astype(np.float64)),
    }

    return steer, throttle, brake, metadata
        

def number_to_one_hot(label, num_classes):
    # Create an array of zeros of length num_classes
    one_hot_vector = np.zeros(num_classes, dtype=int)
    # Set the position corresponding to the label to 1
    one_hot_vector[label] = 1
    return one_hot_vector
        
def main(args, cfg, seed=0):
    configs = []
    if args.route is not None:
        routes = args.route[0]
        scenario_file = args.route[1]
        single_route = None
        if len(args.route) > 2:
            single_route = args.route[2]

        configs += RouteParser.parse_routes_file(routes, scenario_file, single_route)

    if args.scenario is not None:
        configs += ScenarioConfigurationParser.parse_scenario_configuration(args.scenario)

    #import pdb; pdb.set_trace()
    carla_env = DriveEnvWrapper(ScenarioCarlaEnv2(cfg.env, args.host, args.port))
    carla_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = AutoPIDPolicy(cfg.policy).eval_mode
    #env = carla_env  # Replace with your environment's class name
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=100000)
    
    
    
    
    # Config
    config = GlobalConfig()
    
    # Data
    #train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug = config.img_aug)
    #print(len(train_set))
    #val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,)
    #print(len(val_set))
    
    #dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    #dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    config.ratio=2
    config.backbone='resnet18'
    #TCP_model = TCP_planner(config, args.lr)
    model = TCP(config=config)
    #state_dict = torch.load('tcp-13_9M.ckpt')['state_dict']
    # Create a new state dictionary without the 'model.' prefix
    #new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # Load the new state dictionary into your model
    #model.load_state_dict(new_state_dict)
    #model.cuda()
    # The size of the input image is 900 x 256 and the FOV of the camera is set as 100
    # K = 4 (future steps)
    #self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    #front_img = batch['front_img']
		#speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		#target_point = batch['target_point'].to(dtype=torch.float32)
		#command = batch['target_command']
		# VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6

    for config in configs:
        auto_policy.reset([0])
        obs = carla_env.reset(config)
        while True:
            actions = auto_policy.forward({0: obs})
            action = actions[0]['action']

               
            speed = obs['speed']
            #target_point = obs['target_forward']
            #target_point = obs['target']
            front_img = obs['rgb'] #numpy float32, 256x900x3
            # Ensure that the image data is in the range [0, 1] if it isn't already
            if front_img.max() > 1.0:
                front_img /= 255.0
            
            # Define the transformation pipeline
            _im_transform = T.Compose([
                T.ToTensor(),  # Converts the NumPy image (H x W x C) in the range [0, 1] to a PyTorch tensor (C x H x W) in the range [0, 1]
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Apply the transformation
            front_img = _im_transform(front_img)
            command = obs['command']
            if command < 0:
                command = 4
            command -= 1
            assert command in [0, 1, 2, 3, 4, 5]
            cmd_one_hot = [0] * 6
            cmd_one_hot[int(command)] = 1
            cmd_one_hot = np.array(cmd_one_hot)
            #import pdb; pdb.set_trace()
            #print(obs['rotation'])
            #theta = np.arctan(obs['forward_vector'][1]/obs['forward_vector'][0])
            theta = np.radians(-obs['rotation'][1])
            #theta = 0
            #if np.abs(obs['rotation'][1]) <90:
            #    theta = 0
            #else:
            #    theta = np.pi
            R = np.array([\
        			[np.cos(np.pi/2+theta), -np.sin(np.pi/2+theta)],\
        			[np.sin(np.pi/2+theta),  np.cos(np.pi/2+theta)]\
        			])
             
            #target_point = obs['target_forward'] - obs['forward_vector']
            local_command_point_aim = np.array([(obs['target'][1]-obs['location'][1]), obs['target'][0]-obs['location'][0]])
            local_command_point_aim = R.T.dot(local_command_point_aim)
                    
            speed = torch.from_numpy(speed).unsqueeze(0).unsqueeze(0)/12
            #target_point = torch.from_numpy(np.array([-target_point[1],-target_point[0]])).unsqueeze(0)
            #target_point = torch.from_numpy(np.array([target_point[1],target_point[0]])).unsqueeze(0)
            #target_point = torch.from_numpy(target_point).unsqueeze(0)
            #target_point = torch.from_numpy(local_command_point_aim.astype(np.float32)).unsqueeze(0)
            #if local_command_point_aim[0] > 0:
            #    target_point = torch.from_numpy(np.array([-local_command_point_aim[1],local_command_point_aim[0]]).astype(np.float32)).unsqueeze(0)
            #else:
            #    target_point = torch.from_numpy(np.array([local_command_point_aim[1],-local_command_point_aim[0]]).astype(np.float32)).unsqueeze(0)
            target_point = torch.from_numpy(np.array([-local_command_point_aim[1],np.abs(local_command_point_aim[0])]).astype(np.float32)).unsqueeze(0)
            
            #import pdb; pdb.set_trace()
            command = torch.from_numpy(cmd_one_hot).unsqueeze(0)
            front_img = front_img.unsqueeze(0)
            
            state = torch.cat([speed, target_point, command], 1)
            #import pdb; pdb.set_trace()
            #pred = model(front_img.cuda(), state.cuda(), target_point.cuda())

            #control_pid(pred['pred_wp'], speed, target_point)[0]
            #steer_control = control_pid(pred['pred_wp'], speed, target_point)[0].copy()
            
            angle_target = np.degrees(np.pi / 2 - np.arctan2(target_point[0,1], target_point[0,0])) / 90
            
            # Controller
            turn_KP = 0.75
            turn_KI = 0.75
            turn_KD = 0.3
            #turn_KP = 0.7
            #turn_KI = 1.0
            #turn_KD = 0
            turn_n = 40 # buffer size
            
            speed_KP = 5.0
            speed_KI = 0.5
            speed_KD = 1.0
            speed_n = 40 # buffer size
         
            turn_controller = PIDController(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
    
            steer = turn_controller.step(angle_target)
            #steer = steer_control
            steer = np.clip(steer, -1.0, 1.0)
            print(obs['rotation'])
            #print(obs['target'])
            #print(target_point)
            #print(steer_control)
            action['steer']=steer
            if obs['command'] == -1:
                action['brake']=0
                action['throttle']=0.3
                action['steer']=0
                
            #if action['throttle']>0.5:
            #    action['throttle']=0.5

            action['brake']=0
            action['throttle']=0.75
            #print(action)
            timestep = carla_env.step(action)
            obs = timestep.obs
            #print(obs.shape)
            if timestep.info.get('abnormal', False):
                # If there is an abnormal timestep, reset all the related variables(including this env).
                auto_policy.reset([0])
                obs = carla_env.reset(config)
            carla_env.render()
            if timestep.done:
                break


if __name__ == "__main__":
    description = ("DI-drive CaseZoo Environment")

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--route', help='Run a route as a scenario (input:(route_file,scenario_file,[route id]))', nargs='+', type=str)
    parser.add_argument('--scenario', help='Run a single scenario (input: scenario name)', type=str)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=9000,
                        help='TCP port to listen to (default: 9000)', type=int)
    parser.add_argument('--tm-port', default=None,
                        help='Port to use for the TrafficManager (default: None)', type=int)

    args = parser.parse_args()

    main(args, main_config)
