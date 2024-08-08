import os
import argparse
from argparse import RawTextHelpFormatter
from easydict import EasyDict

from core.envs import DriveEnvWrapper, ScenarioCarlaEnv
from core.policy import AutoPIDPolicy
from ding.utils import set_pkg_seed
from core.simulators.srunner.tools.route_parser import RouteParser
from core.simulators.srunner.tools.scenario_parser import ScenarioConfigurationParser
from stable_baselines3 import PPO,SAC

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import csv
        
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        nb_actions = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            #nn.Flatten(),
        )
        #256x19x8
        #self.conv1 = nn.Conv2d(n_input_channels, 6, 6)
        #self.conv2 = nn.Conv2d(6, 16, 6)
        
        # Compute shape by doing one forward pass
        #with torch.no_grad():
        #    n_flatten = self.cnn(
        #        torch.as_tensor(observation_space.sample()[None]).float()
        #    ).flatten().shape[1]


        #self.fc1 = nn.Linear(16*16*7, 256)
        #self.fc1 = nn.Linear(39216, 256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #self.fc4 = nn.Linear(64, nb_actions)
        #self.sig = nn.Tanh()
        features_dim = 128
        #self.linear = nn.Sequential(nn.Linear(152*34, features_dim), nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(306*34, features_dim), nn.ReLU())
        
        self.node_size = 12
        self.proj_shape = (34,self.node_size)  
        #self.proj_shape = (34,36)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        #self.N = 19*8
        self.N = 34*9
        self.norm_shape = (self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        '''
        with torch.no_grad():
            self.model = Model().cuda().eval()
            checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/CDNN_scenario1/ckpts/cdnn/model_epoch_2.tar')
            self.model.load_state_dict(checkpoint['state_dict'])
    
            #self.model = Model().cuda().eval()
            #checkpoint = torch.load('/data/zhuzhuan/CDNN-traffic-saliency/ckpts/cdnn/model_epoch_26.tar')
            #checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/eye_TD3/algo/checkpoints/car_following/model/eye/actor1707651533.pkl')
            #import pdb; pdb.set_trace()
            #self.model.load_state_dict(checkpoint['model'])
            rgb_image = torch.cat([observations[:,1:2],observations[:,1:2],observations[:,1:2]], axis=1)
            # Desired new size
            #new_size = (216//2, 384//2)
            new_size = (72,128)
            #import pdb; pdb.set_trace()
            # Resize/Interpolate the tensor to the new size
            rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
            self.observation = observations[0,0]
            
            output = self.model(rgb_image)
            self.human_map = F.interpolate(output, size=(45, 80), mode='bilinear',align_corners=False).clone()
        '''
        #x = inp.unsqueeze(0) if len(inp.shape)==3 else inp
        N, Cin, H, W = observations.shape
        #import pdb; pdb.set_trace()
        #x = F.max_pool2d( self.conv1(x), 2)
        #x = F.max_pool2d( self.conv2(x), 2)
        x = self.cnn(observations)
        
        _,_,cH,cW = x.shape
        xcoords = torch.arange(cW).repeat(cH,1).float() / cW               
        ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
        spatial_coords = torch.stack([xcoords,ycoords],dim=0).cuda()
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N,1,1,1) 
        x = torch.cat([x,spatial_coords],dim=1)
        x = x.permute(0,2,3,1)
        x = x.flatten(1,2)

        K = self.k_proj(x)                                                 
        K = self.k_norm(K) 
        
        Q = self.q_proj(x)
        Q = self.q_norm(Q) 
        
        V = self.v_proj(x)
        V = self.v_norm(V) 
        A = torch.einsum('bfe,bge->bfg',Q,K)                               
        A = A / np.sqrt(self.node_size)
        A = torch.nn.functional.softmax(A,dim=2) 
        #print(A.shape)
        self.att = A
        E = torch.einsum('bfc,bcd->bfd',A,V)                               
                	
        x = x.reshape(E.size(0),-1)
        
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.sig(self.fc4(x))
        return self.linear(x)

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        nb_actions = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            #nn.Flatten(),
        )
        #256x19x8
        #self.conv1 = nn.Conv2d(n_input_channels, 6, 6)
        #self.conv2 = nn.Conv2d(6, 16, 6)
        
        # Compute shape by doing one forward pass
        #with torch.no_grad():
        #    n_flatten = self.cnn(
        #        torch.as_tensor(observation_space.sample()[None]).float()
        #    ).flatten().shape[1]


        #self.fc1 = nn.Linear(16*16*7, 256)
        #self.fc1 = nn.Linear(39216, 256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 64)
        #self.fc4 = nn.Linear(64, nb_actions)
        #self.sig = nn.Tanh()
        features_dim = 128
        #self.linear = nn.Sequential(nn.Linear(152*34, features_dim), nn.ReLU())
        #306=9*34
        self.linear = nn.Sequential(nn.Linear(306*34, features_dim), nn.ReLU())
        
        self.node_size = 12
        self.proj_shape = (34,self.node_size)  
        #self.proj_shape = (34,36)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)
        #self.N = 19*8
        self.N = 34*9
        self.norm_shape = (self.N,self.node_size)
        self.k_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.norm_shape, elementwise_affine=True)
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            model = Model().cuda().eval()
            checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/CDNN_scenario1/ckpts/cdnn/model_epoch_2.tar')
            model.load_state_dict(checkpoint['state_dict'])
    
            #self.model = Model().cuda().eval()
            #checkpoint = torch.load('/data/zhuzhuan/CDNN-traffic-saliency/ckpts/cdnn/model_epoch_26.tar')
            #checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/eye_TD3/algo/checkpoints/car_following/model/eye/actor1707651533.pkl')
            #import pdb; pdb.set_trace()
            #self.model.load_state_dict(checkpoint['model'])
            rgb_image = torch.cat([observations[:,1:2],observations[:,1:2],observations[:,1:2]], axis=1)
            # Desired new size
            #new_size = (216//2, 384//2)
            new_size = (72,128)
            #import pdb; pdb.set_trace()
            # Resize/Interpolate the tensor to the new size
            rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
            self.observation = observations[0,0]
            
            output = model(rgb_image)
            self.human_map = F.interpolate(output, size=(45, 80), mode='bilinear',align_corners=False).clone()
        
        #x = inp.unsqueeze(0) if len(inp.shape)==3 else inp
        N, Cin, H, W = observations.shape
        #import pdb; pdb.set_trace()
        #x = F.max_pool2d( self.conv1(x), 2)
        #x = F.max_pool2d( self.conv2(x), 2)
        x = self.cnn(observations)
        
        _,_,cH,cW = x.shape
        xcoords = torch.arange(cW).repeat(cH,1).float() / cW               
        ycoords = torch.arange(cH).repeat(cW,1).transpose(1,0).float() / cH
        spatial_coords = torch.stack([xcoords,ycoords],dim=0).cuda()
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N,1,1,1) 
        x = torch.cat([x,spatial_coords],dim=1)
        x = x.permute(0,2,3,1)
        x = x.flatten(1,2)

        K = self.k_proj(x)                                                 
        K = self.k_norm(K) 
        
        Q = self.q_proj(x)
        Q = self.q_norm(Q) 
        
        V = self.v_proj(x)
        V = self.v_norm(V) 
        A = torch.einsum('bfe,bge->bfg',Q,K)                               
        A = A / np.sqrt(self.node_size)
        A = torch.nn.functional.softmax(A,dim=2) 
        #print(A.shape)
        self.att = A
        E = torch.einsum('bfc,bcd->bfd',A,V)                               
                	
        x = x.reshape(E.size(0),-1)
        
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = self.sig(self.fc4(x))
        return self.linear(x)
         
casezoo_config = dict(
    env=dict(
        simulator=dict(
            planner=dict(type='behavior', ),
            n_vehicles=20,
            #n_pedestrians=25,
            disable_two_wheels=True,
            obs=(
#                dict(
#                    name='rgb',
#                    type='rgb',
#                    size=[400, 400],
#                    position=[-5.5, 0, 2.8],
#                    rotation=[-15, 0, 0],
#                ),
#                dict(
#                    name='rgb',
#                    type='rgb',
#                    size=[400, 400],
#                    position=[2, 0, 1],
#                    rotation=[0, 0, 0],
#                    fov=60,
#                ),
                dict(
                    name='rgb',
                    type='rgb',
                    size=[600, 150],
                    position=[2, 0, 1],
                    rotation=[0, 0, 0],
                    fov=150,
                ),
#                dict(
#                    name='segmentation',
#                    type='segmentation',
#                    size=[180, 90],
#                    position=[2, 0, 1],
#                    rotation=[0, 0, 0],
#                ),
                dict(
                    name='segmentation1',
                    type='segmentation',
                    size=[100, 100],
                    position=[2, 0, 1],
                    rotation=[0, 0, 0],
                    fov=60,
                ),
                dict(
                    name='segmentation2',
                    type='segmentation',
                    size=[100, 100],
                    position=[2, 0.5, 1],
                    rotation=[0, 60, 0],
                    fov=60,
                ),
                dict(
                    name='segmentation3',
                    type='segmentation',
                    size=[100, 100],
                    position=[2, -0.5, 1],
                    rotation=[0, -60, 0],
                    fov=60,
                ),
                #dict(
                #    name='birdview',
                #    type='bev',
                #    size=[320, 320],
                #    pixels_per_meter=6,
                #),
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
    
    print(configs)
    
    #config_i = [configs[i]]
    carla_env = DriveEnvWrapper(ScenarioCarlaEnv(cfg.env, cfg.policy, configs, args.host, args.port))
    carla_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = AutoPIDPolicy(cfg.policy).eval_mode
    env = carla_env  # Replace with your environment's class name
    #import pdb; pdb.set_trace()
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    model = PPO("CnnPolicy", env, verbose=1,policy_kwargs=policy_kwargs, tensorboard_log="./ppo_tensorboard/")
    #model.learn(total_timesteps=300000)
    #model = PPO.load("ppo_carla_model_mix")
    #model = PPO.load("ppo_carla_model_mix_old")
    #model = PPO.load("ppo_carla_modelmixed")
    #model = PPO.load("log/n1")
    #model = PPO.load("log/n4")
    #model = PPO.load("ppo_carla_model_20240513-185519_n1")
    #model = PPO.load("ppo_carla_model_20240513-095302_mixed")
    #model = PPO.load("new_checkpoints/ppo_carla_model_20240514-163852_n2")
    model = PPO.load("human_guided_carla_model_20240516-063414_n3")
    #model = PPO.load("/projects/CIBCIGroup/00DataUploading/HAT/DI-drive/demo/auto_run/ppo_carla_model_20240516-002738_n4")
        
    #for i in range(0,25):

        
    total_timesteps = 50000
    #T = 1000  # Number of steps per episode
    #n_episodes = total_timesteps // T
    i = 0
    obs = env.reset(i)
    for k in range(total_timesteps):
        actions, _ = model.predict(obs)
        obs, rewards, dones, infos = env.step(actions)
        att = model.policy.features_extractor.att
        #max_saliency = att[0].mean(dim=0).view(8,19)
        max_saliency = att[0].mean(dim=0).view(9,34)
        flattened_actor = max_saliency.cuda().unsqueeze(0).unsqueeze(0).reshape(1, -1)
        prob_distribution_actor = F.softmax(flattened_actor.view(1, -1), dim=1).view_as(flattened_actor)
        machine_att = prob_distribution_actor.view(9,34)
        
        # Since this is a 2D tensor, we'll add a dimension to it to represent the channels (grayscale in this case)
        machine_att = machine_att.cpu().numpy()
        machine_att = np.expand_dims(machine_att, axis=-1)
        
        # Convert to a format compatible with cv2 (float32)
        machine_att = (machine_att * 255).astype(np.float32)
        
        # Upsample by 8 times using bicubic interpolation
        upsampled_att = cv2.resize(machine_att, (machine_att.shape[1] * 8, machine_att.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
        
        # Since we're working with matplotlib which expects the image in (height, width) for grayscale,
        # and we added a dimension for compatibility with cv2, we need to squeeze that last dimension out
        upsampled_att = np.squeeze(upsampled_att)
        
        #import pdb; pdb.set_trace()
        rgb_image = obs
        # Display the RGB image
        #import pdb; pdb.set_trace()
        
        if k%10 == 0:
            plt.imshow(rgb_image[0])
            # Overlay the attention map with a specific alpha
            # Here, we use the 'jet' colormap for the attention map, but you can choose any other
            plt.imshow(upsampled_att, cmap='jet', alpha=0.6)  # Alpha controls the transparency
            #plt.imshow(upsampled_att)
            plt.show(block=False)
            plt.pause(0.001)
        
        #import pdb; pdb.set_trace()
        env.render()
        if dones:
            #import pdb; pdb.set_trace()
            i = i+1
            obs = env.reset(i)

            # Set up CSV reader and writer
    
            output_file_path='output_dos3_new.csv'
            # Open the input file in read mode and the output file in write mode
            
            with open(output_file_path, mode='a', newline='') as outfile:
                fieldnames = ['status']  # Adjust fieldnames based on your output requirements
                writer = csv.DictWriter(outfile, fieldnames=infos.keys())
                writer.writeheader()
                
                # Write the updated row to the output file
                writer.writerow(infos)
            
            #env.close()
            #break
    '''   
    #import pdb; pdb.set_trace()
    for config in configs:
    #while True:
        #config = configs[0]
        auto_policy.reset([0])
        obs = carla_env.reset()
        #obs = carla_env.reset(config)
        while True:
            actions = auto_policy.forward({0: obs})
            action = actions[0]['action']
            print(action)
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
    '''
    env.close()


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
