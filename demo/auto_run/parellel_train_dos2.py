import os
import argparse
from argparse import RawTextHelpFormatter
from easydict import EasyDict

from core.envs import DriveEnvWrapper, ScenarioCarlaEnv3
from core.policy import AutoPIDPolicy
from ding.utils import set_pkg_seed
from core.simulators.srunner.tools.route_parser import RouteParser
from core.simulators.srunner.tools.scenario_parser import ScenarioConfigurationParser
from stable_baselines3 import PPO,SAC,TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
            
class Model(nn.Module):
    def __init__(self):
        n, m = 24, 3

        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)


        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd4 = conv3x3(4*n, 4*n)

        self.convu3 = conv3x3(8*n, 4*n)
        self.convu2 = conv3x3(6*n, 2*n)
        self.convu1 = conv3x3(3*n, 1*n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        #import pdb; pdb.set_trace()
       
        x1 = x
        x1 = self.convd1(x1)
        # print(x1.size())

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)
        # print(x2.size())

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)
        # print(x3.size())

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)
        # print(x4.size())

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)
        # print(y3.size())

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)
        # print(y2.size())

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)
        # print(y1.size())

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)
        # print(y1.size())
        # exit(0)
        #y1 = F.softmax(y1, dim=1)
        return y1

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./logs/")

class RewardLoggerCallback(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.save_path = './log'
        self.best_mean_reward = -np.inf
        self.best_model_path = os.path.join(self.save_path, best_model_path)
        self.check_freq = 100000
        self.episode_count=0

    def _on_step(self):
        episode_rewards = self.locals.get('rewards', [])
        
        # Assuming episode_rewards is a list of rewards for the current step or episode
        # For an array-like object, you'd aggregate with .sum() or similar
        reward_sum = sum(episode_rewards) if isinstance(episode_rewards, list) else episode_rewards.sum()
        dones = self.locals.get('dones', [])  # Assuming 'dones' indicates terminal states

        latest_reward = reward_sum
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
        # Optionally, save the trained model
        #file_name = "ppo_carla_model_{}_{}".format(current_time,scenario_file[:-18:-12])
        #model.save(file_name)
        #import pdb; pdb.set_trace()
        
        #if self.n_calls % self.check_freq == 0:
        #if self.n_calls % self.check_freq == 0:
        #if latest_reward > self.best_mean_reward:
        #      self.best_mean_reward = latest_reward
        #      self.model.save(self.best_model_path+current_time)
        #      if self.verbose > 0:
        #        print(f"New best mean reward: {self.best_mean_reward:.2f} - model saved!")

        if reward_sum:
            # Log the rewards using the SB3 logger for TensorBoard compatibility
            self.logger.record("step/reward", reward_sum)
            
        # Check for episode completion
        if any(dones):  # If any 'done' flag is True, an episode has ended
            self.episode_count += 1
            self.logger.record("episode/number", self.episode_count)
            self.logger.record("episode/reward", reward_sum)  # Log the reward at the end of the episode

        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model when it achieves a better cumulative reward
    during training than any previous model.
    """
    def __init__(self, check_freq, save_path, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.best_model_path = os.path.join(save_path, 'best_model03')

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve the current cumulative reward from the environment
          x, y = self.training_env.get_attr("episode_rewards", 0)
          latest_reward = np.mean([y[-1] for x, y in zip(x, y) if len(y) > 0])

          if latest_reward > self.best_mean_reward:
              self.best_mean_reward = latest_reward
              self.model.save(self.best_model_path)
              if self.verbose > 0:
                print(f"New best mean reward: {self.best_mean_reward:.2f} - model saved!")

        return True
        
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
            checkpoint = torch.load('human_model/DOS_0520/dos2_model_epoch_5.tar')
            #checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/CDNN_scenario2/ckpts/cdnn/model_epoch_5.tar')
            model.load_state_dict(checkpoint['state_dict'])
    
            #self.model = Model().cuda().eval()
            #checkpoint = torch.load('/data/zhuzhuan/CDNN-traffic-saliency/ckpts/cdnn/model_epoch_26.tar')
            #checkpoint = torch.load('/data/zhuzhuan/VR_driving_data/data/script/eye_TD3/algo/checkpoints/car_following/model/eye/actor1707651533.pkl')
            #import pdb; pdb.set_trace()
            #self.model.load_state_dict(checkpoint['model'])
            rgb_image = torch.cat([observations[:,1:2],observations[:,1:2],observations[:,1:2]], axis=1)
            # Desired new size
            #new_size = (216//2, 384//2)
            #new_size = (72,256)
            new_size = (72,272)
            #new_size = (100,300)
            #import pdb; pdb.set_trace()
            # Resize/Interpolate the tensor to the new size
            #rgb_image = F.interpolate(rgb_image, size=new_size, mode='bilinear', align_corners=False)
            rgb_image = F.interpolate(rgb_image, size=new_size, mode='nearest')
            self.observation = observations[0,0]
            #print(self.observation.max())
            #print(self.observation.min())
            #import matplotlib.pyplot as plt
            #plt.imshow(self.observation.cpu())
            #plt.show(block=False)
            #plt.pause(0.01)
            output = model(rgb_image)
            #self.human_map = F.interpolate(output, size=(45, 80), mode='bilinear',align_corners=False).clone()
            #self.human_map = F.interpolate(output, size=(100, 300), mode='nearest').clone()
            self.human_map = output
            
#            import matplotlib.pyplot as plt
#            plt.subplot(2,1,1)
#            plt.imshow(output[0,0].cpu())
#            plt.subplot(2,1,2)
#            plt.imshow(rgb_image[0,0].cpu())
#            plt.show()
#            plt.show(block=False)
#            plt.pause(0.001)
            
        
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
        

class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq  # Render the environment every `render_freq` steps

    def _on_step(self) -> bool:
        # Check if it's time to render
        #import pdb; pdb.set_trace()
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()

        return True  # Returning True means training will continue

     
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
                    position=[2, 1, 1],
                    rotation=[0, 60, 0],
                    fov=60,
                ),
                dict(
                    name='segmentation3',
                    type='segmentation',
                    size=[100, 100],
                    position=[2, -1, 1],
                    rotation=[0, -60, 0],
                    fov=60,
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
            type='segmentation1',
            outputs=['show']
        ),
    ),
    policy=dict(target_speed=40, ),
)

main_config = EasyDict(casezoo_config)


def main(args, cfg, seed=6):
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
    # Define ports for each environment
    #ports = [9000, 9002, 9004, 9006, 9008]
    ports = [9010, 9012, 9014, 9016, 9018]
    #ports = [9020, 9022, 9024, 9026, 9028]
    #ports = [9030, 9032, 9034, 9036, 9038]
    #ports = [9040, 9042, 9044, 9046, 9048]
    #ports = [9050, 9052, 9054, 9056, 9058]
    #ports = [9060, 9062, 9064, 9066, 9068]
    #ports = [9070, 9072, 9074, 9076, 9078]
    
    # Create a list of environments
    #envs = []
    #for port in ports:
    #    env = ScenarioCarlaEnv(cfg.env, configs[0], args.host, port)
    #    envs.append(env)
    
    # Assuming cfg.env, configs, args.host, seed, and ports are defined
    #configs[0]
   
    # import pdb; pdb.set_trace()
    # Function to create and seed each environment
    def make_env(port):
        def _init():
            #env = ScenarioCarlaEnv(cfg.env, trigger_points, ego_vehicles, other_actors, town, name, type, route, agent, cloudiness, precipitation, sun_altitude_angle, friction, subtype, route_var_name, args.host, port)
            #env = ScenarioCarlaEnv(cfg.env, configs[0], args.host, port)
            env = ScenarioCarlaEnv3(cfg.env, cfg.policy ,routes, scenario_file, single_route, args.host, port)
            env.seed(seed)
            
            return env
        return _init
    
    #import pdb; pdb.set_trace()
    # Create SubprocVecEnv with environments
    envs = SubprocVecEnv([make_env(port) for port in ports])
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # Initialize PPO model with CnnPolicy
    model = PPO("CnnPolicy", envs, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_tensorboard/")
    #model = TD3("CnnPolicy", envs, verbose=1, buffer_size=38400,train_freq=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_tensorboard/")

    #render_callback = RenderCallback(render_freq=1)  # Adjust `render_freq` as needed
    callback = RewardLoggerCallback(scenario_file[:-18:-12])
    
    # Evaluation callback
    #eval_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path='./logs/')
    
    # Combine callbacks if you have more, or just use a single callback
    #callback = CallbackList([callback, checkpoint_callback])
    #model.learn(total_timesteps=500, callback=callback)
    #model.learn(total_timesteps500000, callback=callback)
    #model.learn(total_timesteps=300, callback=callback)
    model.learn(total_timesteps=300000, callback=callback)


    # Train the model
    #model.learn(total_timesteps=50, callback=render_callback)
    #model.learn(total_timesteps=500000, callback=callback)
    
    # Optionally, save the trained model
    #model.save("ppo_carla_model{}".format(scenario_file[:-18:-12]))
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Optionally, save the trained model
    file_name = "dos2_human_guided_carla_model_{}_{}".format(current_time,scenario_file[:-18:-12])
    model.save(file_name)

    #model = PPO.load("ppo_carla_modeln1")
    #import pdb; pdb.set_trace()
    # Train the model
    total_timesteps = 5000
    T = 1000  # Number of steps per episode
    n_episodes = total_timesteps // T
    
    for _ in range(n_episodes):
        obs = envs.reset()
        print(obs.shape)
        for _ in range(T):
            actions, _ = model.predict(obs)
            obs, rewards, dones, infos = envs.step(actions)
    
    '''
    auto_policy = AutoPIDPolicy(cfg.policy).eval_mode
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
    carla_env.close()


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
